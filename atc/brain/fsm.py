"""ATC state machine: phases, positions, clearances, readback checks.

All decisions (runway, route, altitude, squawk, frequencies) are made here from
sim/world data. The LLM only supplies the PilotIntent.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from ..phase_tracker import PhaseTracker, on_runway
from ..phraseology import Phraseology, freq_text, wind
from ..state import Phase, PilotIntent, Position, SimState, Transmission
from ..world import taxi
from ..world.aptdat import Airport, AptIndex, distance_m
from ..world.cifp import cifp_file, parse_cifp, sids_for_runway
from ..world.runways import select_runway

log = logging.getLogger("brain")

VHF = (118000, 137000)

PHASE_POSITION = {
    Phase.PARKED: Position.DELIVERY,
    Phase.CLEARANCE: Position.GROUND,
    Phase.PUSH_START: Position.GROUND,
    Phase.TAXI_OUT: Position.GROUND,
    Phase.HOLDING: Position.TOWER,
    Phase.TAKEOFF: Position.TOWER,
    Phase.DEPARTURE: Position.DEPARTURE,
    Phase.ENROUTE: Position.CENTER,
    Phase.ARRIVAL: Position.CENTER,
    Phase.APPROACH: Position.APPROACH,
    Phase.LANDING: Position.TOWER,
    Phase.TAXI_IN: Position.GROUND,
}
ARRIVAL_PHASES = (Phase.ARRIVAL, Phase.APPROACH, Phase.LANDING, Phase.TAXI_IN)


@dataclass
class FlightConfig:
    callsign: str = "N123AB"
    telephony: str | None = None
    destination: str | None = None
    cruise_ft: int = 35000
    initial_alt_ft: int = 5000
    sid: str | None = None
    first_fix: str | None = None
    handoff_departure_agl: int = 1000
    handoff_center_ft: int = 10000
    center_khz: int = 127900
    region: str = "ICAO"
    transition_alt: int = 5000
    strict_readback: bool = False
    confirm_readback: bool = False   # say "readback correct" (real ATC is usually silent)
    arrival_gate: str | None = None


@dataclass
class Pending:
    position: Position
    expected: dict
    required: set
    tx: Transmission
    on_ok: Callable[[], list[Transmission]] | None = None
    t: float = field(default_factory=time.time)
    repeats: int = 0
    pilot_t_at_issue: float = 0.0   # brain.last_pilot_t when issued; unchanged = no reply yet


class Brain:
    def __init__(self, fc: FlightConfig, airport: Airport, index: AptIndex | None = None,
                 xp_root=None):
        self.fc = fc
        self.dep = airport
        self.dest: Airport | None = None
        self.index = index
        self.xp_root = xp_root
        self.phr = Phraseology(fc.callsign, fc.telephony, fc.region, fc.transition_alt)
        self.phase = Phase.PARKED
        self.tracker = PhaseTracker()
        self.runway: str | None = None
        self.arr_runway: str | None = None
        self.sid: str | None = None
        self.squawk: str | None = None
        self.assigned_alt: int | None = None
        self.alt_captured = False
        self.assigned_hdg: int | None = None   # magnetic, while under radar vectors
        self.hdg_captured = False
        self.route: taxi.TaxiRoute | None = None
        self.route_joined = False
        self.leg_idx = 0                       # current clearance limit in route.legs
        self.cleared_runways: set[str] = set()  # runways cleared to cross on this route
        self.cleared_takeoff = False
        self.cleared_land = False
        self.pending: Pending | None = None
        self.last_tx: Transmission | None = None
        self.last_pilot_t = 0.0
        self.sim = SimState()
        self.gate: str | None = None
        # VATSIM first call "good day, A320 Lufthansa 4AB on stand 1": who is calling
        self.identified = False                 # callsign taken from the pilot's first call
        self.called_type: str | None = None
        self.stand: str | None = None
        self.dest_choices: list[str] = []       # several airports in the requested city
        self._rng = random.Random(fc.callsign)

    # ------------------------------------------------------------------ helpers
    @property
    def field(self) -> Airport:
        if self.phase in ARRIVAL_PHASES and self.dest is not None:
            return self.dest
        return self.dep

    @property
    def acf_type(self) -> str:
        return self.sim.acf_icao or self.called_type or "A320"

    @staticmethod
    def _vhf(khz: int | None) -> bool:
        return bool(khz) and VHF[0] <= khz < VHF[1]

    def _apt_freq(self, ap: Airport, *kinds: str) -> int | None:
        for k in kinds:
            for f in ap.freqs.get(k, []):
                if self._vhf(f.khz):
                    return f.khz
        return None

    def freq_of(self, pos: Position, ap: Airport | None = None) -> int | None:
        ap = ap or self.field
        if pos == Position.DELIVERY:
            return self._apt_freq(ap, "delivery", "ground", "tower", "unicom")
        if pos == Position.GROUND:
            return self._apt_freq(ap, "ground", "tower", "unicom")
        if pos == Position.TOWER:
            return self._apt_freq(ap, "tower", "unicom")
        if pos == Position.DEPARTURE:
            return self._apt_freq(ap, "departure", "approach") or self.fc.center_khz
        if pos == Position.APPROACH:
            return self._apt_freq(ap, "approach", "departure") or self.fc.center_khz
        return self.fc.center_khz

    def expected_position(self) -> Position:
        return PHASE_POSITION[self.phase]

    def position_on(self, khz: int) -> Position | None:
        """Which position is listening on this frequency (expected one wins ties)."""
        def same(a, b):
            return a and b and abs(a - b) <= 5
        exp = self.expected_position()
        if same(self.freq_of(exp), khz):
            return exp
        for pos in Position:
            for ap in filter(None, (self.field, self.dep, self.dest)):
                if same(self.freq_of(pos, ap), khz):
                    return pos
        return None

    def _tx(self, pos: Position, pair: tuple[str, str], expects: bool = False) -> Transmission:
        tx = Transmission(pos, pair[0], pair[1], expects, self.freq_of(pos) or 0)
        self.last_tx = tx
        log.info("ATC %s: %s", pos.value, pair[0])
        return tx

    def _expect(self, tx: Transmission, expected: dict, required: set, on_ok=None) -> None:
        self.pending = Pending(tx.position, expected, required, tx, on_ok,
                               pilot_t_at_issue=self.last_pilot_t)

    def _set_phase(self, p: Phase) -> None:
        if p != self.phase:
            log.info("phase %s -> %s", self.phase.value, p.value)
            self.phase = p

    def context(self) -> dict:
        """Context given to the intent parser / STT prompt."""
        ap = self.field
        ctx = {
            "callsign": self.fc.callsign, "phase": self.phase.value,
            "expected_station": f"{ap.icao} {self.expected_position().value}",
            "runways": [e.name for r in ap.runways for e in r.ends],
        }
        if self.pending:
            ctx["awaiting_readback_of"] = self.pending.tx.text
        return ctx

    def stt_prompt(self) -> str:
        ap = self.field
        twys = sorted({e.name for e in ap.edges if e.name and e.kind == "taxiway"})[:30]
        stations = ", ".join(self.station_name(p) for p in
                             (Position.DELIVERY, Position.GROUND, Position.TOWER, Position.DEPARTURE))
        from ..phraseology import taxiway as spoken_twy
        # taxiways as they are said ("Golf, Mike"): Whisper heard "G M" as "go with mic"
        twys_spoken = ", ".join(spoken_twy(t) for t in twys)
        return (f"{stations}, good day, {self.phr.cs_spoken}, radio check. {ap.icao} {ap.name}. Runway "
                + ", ".join(e.name for r in ap.runways for e in r.ends)
                + f". Taxi via {twys_spoken}, hold short. Squawk, QNH, flight level, cleared, readback, "
                  "decimal, on stand, request IFR clearance.")

    # --------------------------------------------------------------- readbacks
    def _check_readback(self, intent: PilotIntent) -> list[Transmission]:
        p = self.pending
        rb = intent.readback
        wrong: list[tuple[str, str, str]] = []
        for slot, val in p.expected.items():
            got = rb.get(slot)
            if slot == "altitude" and rb.get("flight_level"):
                got = int(rb["flight_level"]) * 100  # a stated flight level wins over a stray "altitude"
            if got is None:
                if slot in p.required or (self.fc.strict_readback and slot != "frequency"):
                    wrong.append(self._label(slot, val))
                continue
            if not self._same(slot, got, val):
                wrong.append(self._label(slot, val))
        if wrong:
            p.t = time.time()
            p.pilot_t_at_issue = self.last_pilot_t
            return [self._tx(p.position, self.phr.readback_wrong(wrong), True)]
        self.pending = None
        out = p.on_ok() if p.on_ok else []
        if not out and self.fc.confirm_readback:
            out = [self._tx(p.position, self.phr.readback_correct())]
        return out

    @staticmethod
    def _same(slot: str, got, val) -> bool:
        if slot == "altitude":
            return abs(int(got) - int(val)) < 50
        if slot == "heading":
            return (int(got) - int(val)) % 360 == 0
        if slot == "frequency":
            try:
                return abs(float(got) * 1000 - int(val)) <= 5
            except ValueError:
                return False
        if slot in ("cleared_takeoff", "cleared_land", "line_up"):
            return bool(got) == bool(val)
        return str(got).upper().lstrip("0") == str(val).upper().lstrip("0")

    def _label(self, slot: str, val) -> tuple[str, str, str]:
        from ..phraseology import digits, runway as rwy_s, freq
        if slot == "squawk":
            return "squawk", str(val), digits(val)
        if slot in ("runway", "hold_short"):
            lab = "runway" if slot == "runway" else "hold short runway"
            return lab, str(val), rwy_s(str(val))
        if slot == "altitude":
            t, s = self.phr.alt(int(val))
            return "altitude", t, s
        if slot == "frequency":
            return "frequency", freq_text(int(val)), freq(int(val))
        if slot == "heading":
            from ..phraseology import heading as hdg_s
            return "heading", f"{int(val):03d}", hdg_s(int(val))
        if slot == "cleared_takeoff":
            return "cleared for takeoff", "", ""
        if slot == "cleared_land":
            return "cleared to land", "", ""
        return slot, str(val), str(val)

    # ------------------------------------------------------------ pilot input
    def handle(self, intent: PilotIntent, sim: SimState) -> list[Transmission]:
        self.sim = sim
        self.last_pilot_t = time.time()
        pos = self.position_on(sim.tx_khz)
        log.info("pilot on %s (%s): %s -> %s %s", freq_text(sim.tx_khz) if sim.tx_khz else "?",
                 pos.value if pos else "nobody", intent.raw_text, intent.intent, intent.readback)
        if pos is None:
            return []
        self._identify(intent)
        if pos == Position.DELIVERY and self.dest_choices and (code := self._chosen_destination(intent)):
            intent = PilotIntent("request_clearance", callsign=intent.callsign, raw_text=intent.raw_text,
                                 data={"destination": code})
        if intent.unclear or intent.intent == "other":
            return [self._tx(pos, self.phr.say_again())]
        if intent.intent == "say_again":
            if self.last_tx and self.last_tx.position == pos:
                return [self._tx(pos, (self.last_tx.text, self.last_tx.spoken), self.last_tx.expects_readback)]
            return [self._tx(pos, self.phr.say_again())]
        if intent.intent == "radio_check":
            return [self._tx(pos, self.phr.radio_check(self.station_name(pos)))]
        if self.pending and pos == self.pending.position and (
                intent.intent in ("readback", "acknowledge") or self._mentions_expected(intent)):
            # "climbing flight level 100" is a readback, whatever the classifier called it
            return self._check_readback(intent)
        if intent.intent in ("readback", "acknowledge"):
            self.pending = None
            return []  # nothing outstanding: silence is the realistic answer
        if self._acknowledges_handoff(intent, pos):
            return []  # "126.3 for push and start, thank you" on the old frequency
        exp = self.expected_position()
        handler = getattr(self, f"_on_{pos.name.lower()}", None)
        out = handler(intent) if handler else None
        if out is not None:
            return out
        if pos != exp:
            return [self._tx(pos, self.phr.wrong_frequency(exp.value.lower(), self.freq_of(exp)))]
        if intent.intent in ("check_in", "report_position"):
            # initial call without a request ("Delivery, DLH4AB, good day")
            return [self._tx(pos, self.phr.go_ahead(self.station_name(pos)))]
        return [self._tx(pos, self.phr.say_again())]

    def _identify(self, it: PilotIntent) -> None:
        """First call on the ground sets callsign, type and stand; the config callsign is the fallback."""
        if self.identified or self.phase != Phase.PARKED:
            return
        if t := it.data.get("acf_type"):
            self.called_type = t
        if st := it.data.get("stand"):
            self.stand = self._match_stand(st)
        if it.callsign:
            self.identified = True
            if it.callsign != self.fc.callsign:
                log.info("callsign from first call: %s (config %s)", it.callsign, self.fc.callsign)
                self.fc.callsign, self.fc.telephony = it.callsign, None
                self.phr = Phraseology(self.fc.callsign, None, self.fc.region, self.fc.transition_alt)

    def _match_stand(self, said: str) -> str:
        """'1' -> the apt.dat start location named '1' / 'Stand 1' / 'Gate 1', else as said."""
        for g in self.dep.gates:
            name = g.name.upper()
            if name == said or name.split()[-1:] == [said] or name.replace(" ", "") == said:
                return g.name
        return said

    def _chosen_destination(self, it: PilotIntent) -> str | None:
        """Answer to 'say destination ICAO': a code from the call, preferably one of the city's."""
        from ..llm.intent_parser import spoken_codes
        codes = spoken_codes(it.raw_text)
        if d := it.data.get("destination"):
            codes.insert(0, d)
        for c in codes:
            if c in self.dest_choices:
                return c
        for c in codes:
            if self._resolve_airport(c, city=False):
                return c
        return None

    def station_name(self, pos: Position) -> str:
        """'Burlington Delivery': city from the apt.dat frequency names, else airport name."""
        if pos == Position.CENTER:
            return "Center"
        ap = self.field
        city = ap.meta.get("city", "")
        if not city:
            for lst in ap.freqs.values():
                for f in lst:
                    words = f.name.split()
                    if len(words) >= 2:
                        city = words[0]
                        break
                if city:
                    break
        city = (city or ap.name.split()[0]).title()
        return f"{city} {pos.value}"

    # Each _on_<position> returns transmissions, or None when that station would not
    # handle this request now (-> "contact <expected>").

    def _on_delivery(self, it: PilotIntent):
        if it.intent != "request_clearance":
            return None
        if self.phase not in (Phase.PARKED, Phase.CLEARANCE, Phase.PUSH_START):
            return None
        dest = (it.data.get("destination") or it.readback.get("destination") or self.fc.destination or "")
        if not dest:
            return [self._tx(Position.DELIVERY, self.phr.custom("say destination"))]
        icao = self._resolve_airport(dest)
        if icao is None:
            from ..phraseology import letters
            d = dest.upper()
            city, choices = self._city_airports(d)
            if len(choices) > 1:
                self.dest_choices = choices
                return [self._tx(Position.DELIVERY, self.phr.custom(
                    f"several airports in {city}, say destination ICAO"))]
            spoken = letters(d) if len(d) <= 4 else d.title()
            return [self._tx(Position.DELIVERY, self.phr.custom(
                f"unable to find {d}, say again destination", f"unable to find {spoken}, say again destination"))]
        self.dest_choices = []
        self.fc.destination = icao
        if self.runway is None:
            self.runway = self._select_dep_runway()
        self.sid = self._select_sid(self.runway)
        self.squawk = self.squawk or self._new_squawk()
        self.assigned_alt = self.fc.initial_alt_ft
        dep_khz = self.freq_of(Position.DEPARTURE, self.dep)
        tx = self._tx(Position.DELIVERY, self.phr.ifr_clearance(
            self.fc.destination, self.runway, self.sid, self.assigned_alt, self.squawk, dep_khz), True)
        gnd = self.freq_of(Position.GROUND, self.dep)

        def ok():
            self._set_phase(Phase.CLEARANCE)
            if gnd and gnd != self.freq_of(Position.DELIVERY, self.dep):
                return [self._tx(Position.DELIVERY, self.phr.readback_correct(
                    "ground", gnd, "for push and start"))]
            return [self._tx(Position.DELIVERY, self.phr.readback_correct())]

        self._expect(tx, {"runway": self.runway, "altitude": self.assigned_alt,
                          "squawk": self.squawk}, {"squawk"}, ok)
        return [tx]

    def _on_ground(self, it: PilotIntent):
        G = Position.GROUND
        if self.phase in (Phase.TAXI_IN, Phase.LANDING) or (
                self.phase in (Phase.APPROACH, Phase.ARRIVAL) and self.sim.on_ground):
            if it.intent in ("request_taxi", "check_in", "report_position"):
                return self._taxi_in()
            return None
        if it.intent == "request_clearance":
            dl = self.freq_of(Position.DELIVERY)
            if dl and dl != self.freq_of(G):
                return [self._tx(G, self.phr.contact("delivery", dl, "for clearance"))]
            return self._on_delivery(it)
        if self.phase == Phase.PARKED and it.intent in ("request_pushback", "request_startup", "request_taxi"):
            dl = self.freq_of(Position.DELIVERY)
            if dl and dl != self.freq_of(G):
                return [self._tx(G, self.phr.contact("delivery", dl, "for clearance first"))]
        if it.intent == "request_pushback":
            self._set_phase(Phase.PUSH_START)
            return [self._tx(G, self.phr.push_start(self._push_facing()))]
        if it.intent == "request_startup":
            return [self._tx(G, self.phr.startup())]
        if it.intent == "request_taxi" or (it.intent == "check_in" and self.phase in (Phase.CLEARANCE, Phase.PUSH_START)):
            return self._taxi_out()
        return None

    def _taxi_out(self) -> list[Transmission]:
        s = self.sim
        if self.runway is None:
            self.runway = self._select_dep_runway()
        cls = taxi.aircraft_class(self.acf_type)
        self._new_route(taxi.route_to_runway(self.dep, s.lat, s.lon, self.runway, cls))
        if not self.route.ok:
            log.warning("taxi route not found, using fallback instruction")
        hold = self._hold_word(self.route.legs[0].hold_short) if self.route.legs else self.runway
        log.info("taxi route legs: %s", [(lg.taxiways, lg.hold_short, lg.cross) for lg in self.route.legs])
        tx = self._tx(Position.GROUND, self.phr.taxi(self.route, self.runway, s.qnh_hpa, hold), True)
        self._set_phase(Phase.TAXI_OUT)
        self._expect(tx, {"hold_short": hold}, {"hold_short"})
        return [tx]

    def _new_route(self, route: taxi.TaxiRoute | None) -> None:
        self.route = route
        self.leg_idx = 0
        self.route_joined = False
        self.cleared_runways = set()

    def _hold_word(self, runway_name: str | None) -> str | None:
        """'15/33' -> the end in use (most headwind); a single end is returned as is."""
        if not runway_name or "/" not in runway_name:
            return runway_name
        from ..world.runways import wind_components
        ends = [self.field.runway_end(n) for n in runway_name.split("/")]
        ends = [e for e in ends if e is not None]
        if not ends:
            return runway_name.split("/")[0]
        return max(ends, key=lambda e: wind_components(e, self.sim.wind_dir_true, self.sim.wind_speed_kt)[0]).name

    def has_next_leg(self) -> bool:
        return bool(self.route and self.route.ok and self.leg_idx < len(self.route.legs) - 1)

    def continue_taxi(self) -> list[Transmission]:
        """At a crossing hold point: clear the crossing and the next leg."""
        self.leg_idx += 1
        leg = self.route.legs[self.leg_idx]
        self.cleared_runways.add(leg.cross)
        last = self.leg_idx == len(self.route.legs) - 1
        hold = self._hold_word(leg.hold_short)
        tx = self._tx(Position.GROUND, self.phr.taxi_continue(
            leg, self._hold_word(leg.cross), hold, gate=self.gate if self.phase == Phase.TAXI_IN else None,
            intersection=self.route.intersection if last else None), bool(hold))
        if hold:
            self._expect(tx, {"hold_short": hold}, {"hold_short"})
        return [tx]

    def _taxi_in(self) -> list[Transmission]:
        s = self.sim
        ap = self.field
        gate = self._pick_gate(ap)
        cls = taxi.aircraft_class(self.acf_type)
        if gate is None:
            self.route = None
            return [self._tx(Position.GROUND, self.phr.custom("taxi to parking at your discretion"))]
        self.gate = gate.name
        self._new_route(taxi.route_to_point(ap, s.lat, s.lon, gate.lat, gate.lon, cls, gate.name))
        self._set_phase(Phase.TAXI_IN)
        hold = self._hold_word(self.route.legs[0].hold_short) if self.route.legs else None
        tx = self._tx(Position.GROUND, self.phr.taxi_to_gate(self.route, gate.name, hold), bool(hold))
        if hold:
            self._expect(tx, {"hold_short": hold}, {"hold_short"})
        return [tx]

    def _on_tower(self, it: PilotIntent):
        T = Position.TOWER
        if it.intent in ("ready_for_departure", "check_in") and self.phase in (
                Phase.TAXI_OUT, Phase.HOLDING, Phase.TAKEOFF):
            if self.runway is None:
                self.runway = self._select_dep_runway()
            self._set_phase(Phase.HOLDING)
            return self._takeoff_clearance()
        if it.intent in ("report_final", "check_in", "report_position") and self.phase in (
                Phase.APPROACH, Phase.LANDING, Phase.ARRIVAL):
            self._set_phase(Phase.LANDING)
            rwy = self.arr_runway or self._select_arr_runway()
            self.arr_runway = rwy
            self.cleared_land = True
            tx = self._tx(T, self.phr.cleared_land(rwy, wind(self.sim.wind_dir_true, self.sim.wind_speed_kt)), True)
            self._expect(tx, {"runway": rwy, "cleared_land": True}, set())
            return [tx]
        return None

    def _takeoff_clearance(self) -> list[Transmission]:
        rwy = self.runway
        on = on_runway(self.dep, self.sim.lat, self.sim.lon)
        self.cleared_takeoff = True
        tx = self._tx(Position.TOWER, self.phr.cleared_takeoff(
            rwy, wind(self.sim.wind_dir_true, self.sim.wind_speed_kt)), True)
        self._set_phase(Phase.TAKEOFF)
        self._expect(tx, {"runway": rwy, "cleared_takeoff": True}, {"cleared_takeoff"} if not on else set())
        return [tx]

    def _on_departure(self, it: PilotIntent):
        D = Position.DEPARTURE
        if self.phase in (Phase.TAKEOFF, Phase.DEPARTURE) and it.intent in ("check_in", "report_position", "request_climb"):
            self._set_phase(Phase.DEPARTURE)
            new_alt = min(self.fc.cruise_ft, max(self.assigned_alt or 0, 10000)) \
                if self.fc.cruise_ft > (self.assigned_alt or 0) else self.assigned_alt
            self._assign_alt(new_alt)
            expected: dict = {"altitude": new_alt}
            hdg, turn = (None, None)
            if self.sid is None:  # "radar vectors" departure: give a heading
                hdg = self._vector_heading()
                turn = self._turn_direction(hdg)
                self._assign_heading(hdg)
                expected["heading"] = hdg
            tx = self._tx(D, self.phr.radar_contact(new_alt, hdg, turn), True)
            self._expect(tx, expected, set())
            return [tx]
        return self._on_center(it) if self.phase in (Phase.ENROUTE, Phase.ARRIVAL, Phase.APPROACH) else None

    def _on_center(self, it: PilotIntent):
        C = Position.CENTER
        if it.intent in ("check_in", "request_climb", "report_position") and self.phase in (
                Phase.DEPARTURE, Phase.ENROUTE):
            self._set_phase(Phase.ENROUTE)
            alt = self.fc.cruise_ft
            self._assign_alt(alt)
            verb = "identified, climb" if self.sim.altitude_ft < alt - 300 else "identified, maintain"
            t, s = self.phr.climb(alt, verb)
            if self.assigned_hdg is not None:  # end of radar vectors
                self._assign_heading(None)
                nav_t, nav_s = self.phr.end_vectors(self.fc.destination)
                t, s = f"{t}, {nav_t}", f"{s}, {nav_s}"
            tx = self._tx(C, (t, s), True)
            self._expect(tx, {"altitude": alt}, set())
            return [tx]
        if it.intent == "request_descent" and self.phase in (Phase.ENROUTE, Phase.ARRIVAL):
            self._set_phase(Phase.ARRIVAL)
            self._load_dest()
            want = it.data.get("altitude") or (it.data.get("flight_level") or 0) * 100
            alt = int(want) if want else 10000
            self._assign_alt(alt)
            tx = self._tx(C, self.phr.climb(alt, "descend"), True)
            self._expect(tx, {"altitude": alt}, set())
            return [tx]
        if it.intent in ("request_approach", "check_in", "report_position") and self.phase in (
                Phase.ARRIVAL, Phase.APPROACH):
            return self._approach_clearance(C if it.intent != "check_in" else C)
        return None

    def _on_approach(self, it: PilotIntent):
        if self.phase in (Phase.ARRIVAL, Phase.APPROACH, Phase.ENROUTE) and it.intent in (
                "check_in", "request_approach", "report_position", "request_descent"):
            if self.phase == Phase.ENROUTE:
                self._set_phase(Phase.ARRIVAL)
            return self._approach_clearance(Position.APPROACH)
        return self._on_departure(it) if self.phase in (Phase.TAKEOFF, Phase.DEPARTURE) else None

    def _approach_clearance(self, pos: Position) -> list[Transmission]:
        self._load_dest()
        ap = self.dest or self.dep
        self.arr_runway = self.arr_runway or self._select_arr_runway()
        alt = int(round((ap.elevation_ft + 2500) / 500.0) * 500)
        self._assign_alt(alt)
        self._set_phase(Phase.APPROACH)
        twr = self._apt_freq(ap, "tower", "unicom")
        tx = self._tx(pos, self.phr.cleared_approach("ILS", self.arr_runway, alt, twr), True)
        self._expect(tx, {"runway": self.arr_runway, "altitude": alt}, set())
        return [tx]

    # ------------------------------------------------------------ decisions
    def _assign_alt(self, alt: int) -> None:
        self.assigned_alt = alt
        self.alt_captured = False

    def _assign_heading(self, hdg: int | None) -> None:
        self.assigned_hdg = hdg
        self.hdg_captured = False

    @property
    def magvar(self) -> float:
        """True minus magnetic, from the sim's own two headings."""
        return ((self.sim.heading_true - self.sim.heading_mag + 180) % 360) - 180

    def _vector_heading(self) -> int:
        """Magnetic heading towards the destination, else runway heading; rounded to 10°."""
        from ..world.aptdat import bearing_deg
        true = None
        if self.index and self.fc.destination and self.fc.destination in self.index.entries:
            _, _, lat, lon = self.index.entries[self.fc.destination][:4]
            if lat is not None:
                true = bearing_deg(self.sim.lat, self.sim.lon, lat, lon)
        if true is None:
            end = self.dep.runway_end(self.runway or "")
            true = end.heading_true if end else self.sim.heading_true
        mag = (true - self.magvar) % 360
        return int(round(mag / 10.0) * 10) % 360 or 360

    def _turn_direction(self, hdg: int) -> str | None:
        diff = (hdg - self.sim.heading_mag + 540) % 360 - 180
        if abs(diff) < 10:
            return None  # "fly heading"
        return "right" if diff > 0 else "left"

    def _select_dep_runway(self) -> str:
        e = select_runway(self.dep, self.sim.wind_dir_true, self.sim.wind_speed_kt,
                          acf_type=self.acf_type)
        return e.name if e else "??"

    def _select_arr_runway(self) -> str:
        ap = self.dest or self.dep
        e = select_runway(ap, self.sim.wind_dir_true, self.sim.wind_speed_kt,
                          acf_type=self.acf_type)
        return e.name if e else "??"

    def _select_sid(self, rwy: str) -> str | None:
        if self.fc.sid:
            return self.fc.sid
        if not (self.xp_root and self.fc.first_fix):
            return None
        f = cifp_file(self.xp_root, self.dep.icao)
        if not f:
            return None
        for p in sids_for_runway(parse_cifp(f), rwy):
            if any(self.fc.first_fix.upper() in fixes for fixes in p.transitions.values()):
                return p.ident
        return None

    def _mentions_expected(self, it: PilotIntent) -> bool:
        """The call repeats values of the instruction we are waiting a readback for."""
        if it.intent not in ("check_in", "report_position", "request_climb", "request_descent", "other"):
            return False
        exp = set(self.pending.expected)
        got = set(it.readback)
        if "altitude" in exp and "flight_level" in got:
            got.add("altitude")
        return bool(exp & got)

    def _acknowledges_handoff(self, it: PilotIntent, pos: Position) -> bool:
        """Pilot reads back another station's frequency on this one = acknowledging a handoff."""
        f = it.readback.get("frequency")
        if not f:
            return False
        try:
            khz = round(float(f) * 1000)
        except ValueError:
            return False
        own = self.freq_of(pos)
        for other in Position:
            k = self.freq_of(other)
            if other != pos and k and k != own and abs(k - khz) <= 5:
                return True
        return False

    def _push_facing(self) -> str | None:
        """Direction the nose should point after pushback: along the start of the taxi route."""
        if self.runway is None:
            self.runway = self._select_dep_runway()
        s = self.sim
        rt = taxi.route_to_runway(self.dep, s.lat, s.lon, self.runway,
                                  taxi.aircraft_class(self.acf_type))
        if not rt.ok or len(rt.nodes) < 2:
            return None
        from ..world.aptdat import bearing_deg
        a = self.dep.nodes[rt.nodes[0]]
        b = self.dep.nodes[rt.nodes[-1]]
        for nid in rt.nodes[1:]:  # first route point ~60 m away; short connectors give odd bearings
            b = self.dep.nodes[nid]
            if distance_m(a.lat, a.lon, b.lat, b.lon) >= 60:
                break
        brg = bearing_deg(a.lat, a.lon, b.lat, b.lon)
        names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
        return names[int((brg + 22.5) // 45) % 8]

    def _resolve_airport(self, code: str, city: bool = True) -> str | None:
        """Destination against apt.dat: 'KBOS', a city with one airport ('Boston', 'New York Kennedy'
        -> longest city prefix), 'BOS' -> 'KBOS'. Without an index, trust 4 letters."""
        code = code.upper().strip()
        if self.index is None:
            return code if len(code) == 4 and code.isalpha() else None
        if code in self.index.entries:
            return code
        if city:
            _, found = self._city_airports(code)
            if found:
                return found[0] if len(found) == 1 else None  # several: the caller asks for the ICAO
        if len(code) == 3 and "K" + code in self.index.entries:
            return "K" + code
        return None

    def _city_airports(self, name: str) -> tuple[str, list[str]]:
        """Longest city prefix of what was said: 'NEW YORK KENNEDY' -> ('New York', [...])."""
        words = name.split() if self.index else []
        for n in range(len(words), 0, -1):
            city = " ".join(words[:n])
            if found := self.index.in_city(city):
                return city.title(), found
        return name.title(), []

    def _new_squawk(self) -> str:
        while True:
            code = "".join(str(self._rng.randint(0, 7)) for _ in range(4))
            if code not in ("7500", "7600", "7700", "2000", "1200", "7000") and code[0] != "0":
                return code

    def _load_dest(self) -> None:
        if self.dest is None and self.index and self.fc.destination:
            try:
                self.dest = self.index.load(self.fc.destination)
            except KeyError:
                log.warning("destination %s not in apt.dat", self.fc.destination)

    def _pick_gate(self, ap: Airport):
        if self.fc.arrival_gate:
            for g in ap.gates:
                if g.name.upper() == self.fc.arrival_gate.upper():
                    return g
        gates = [g for g in ap.gates if g.kind == "gate"] or ap.gates
        if not gates:
            return None
        jets = [g for g in gates if "jets" in g.types] if taxi.aircraft_class(self.acf_type) >= "C" else gates
        return self._rng.choice(jets or gates)

    # -------------------------------------------------------- physical events
    def update(self, sim: SimState) -> None:
        self.sim = sim
        ev = self.tracker.update(sim)
        if ev.took_off and self.phase in (Phase.PARKED, Phase.CLEARANCE, Phase.PUSH_START,
                                          Phase.TAXI_OUT, Phase.HOLDING):
            self._set_phase(Phase.TAKEOFF)  # departed without clearance; carry on
        if ev.landed and self.phase in (Phase.ARRIVAL, Phase.APPROACH, Phase.LANDING, Phase.ENROUTE):
            if self.dest is None:
                self._load_dest()
            self._set_phase(Phase.LANDING)
        if ev.shutdown_parked and self.phase == Phase.TAXI_IN:
            self.reset_for_next_flight()
        if self.assigned_alt and abs(sim.altitude_ft - self.assigned_alt) < 200:
            self.alt_captured = True
        if self.assigned_hdg is not None and abs((sim.heading_mag - self.assigned_hdg + 540) % 360 - 180) < 10:
            self.hdg_captured = True

    # ------------------------------------------------------------ persistence
    STATE_FIELDS = ("runway", "arr_runway", "sid", "squawk", "assigned_alt", "assigned_hdg",
                    "cleared_takeoff", "cleared_land", "gate", "identified", "called_type", "stand")

    def snapshot(self) -> dict:
        """What survives a program restart (the taxi route is recomputed on request)."""
        d = {k: getattr(self, k) for k in self.STATE_FIELDS}
        d.update(phase=self.phase.value, dep=self.dep.icao, destination=self.fc.destination,
                 callsign=self.fc.callsign)
        return d

    def restore(self, d: dict) -> None:
        for k in self.STATE_FIELDS:
            if k in d:
                setattr(self, k, d[k])
        self.fc.destination = d.get("destination") or self.fc.destination
        if d.get("identified") and d.get("callsign") and d["callsign"] != self.fc.callsign:
            self.fc.callsign, self.fc.telephony = d["callsign"], None
            self.phr = Phraseology(self.fc.callsign, None, self.fc.region, self.fc.transition_alt)
        self.phase = Phase(d.get("phase", Phase.PARKED.value))
        if self.phase in ARRIVAL_PHASES:
            self._load_dest()
        log.info("restored state: %s", d)

    def skip_clearance(self) -> Transmission:
        """Testing shortcut: issue the IFR clearance without the radio exchange."""
        if self.runway is None:
            self.runway = self._select_dep_runway()
        self.sid = self._select_sid(self.runway)
        self.squawk = self.squawk or self._new_squawk()
        self.assigned_alt = self.fc.initial_alt_ft
        tx = self._tx(Position.DELIVERY, self.phr.ifr_clearance(
            self.fc.destination or "destination", self.runway, self.sid, self.assigned_alt,
            self.squawk, self.freq_of(Position.DEPARTURE, self.dep)))
        self._set_phase(Phase.CLEARANCE)
        return tx

    def reset_for_next_flight(self) -> None:
        if self.dest is not None:
            self.dep, self.dest = self.dest, None
        self.fc.destination = None
        self.runway = self.arr_runway = self.sid = self.squawk = None
        self.assigned_alt = None
        self._new_route(None)
        self.cleared_takeoff = self.cleared_land = False
        self.pending = None
        self.identified, self.stand, self.dest_choices = False, None, []  # next flight: new first call
        self._set_phase(Phase.PARKED)

    def hold_point(self) -> tuple[float, float] | None:
        """Hold short point of the current leg (the final one is the end of the route)."""
        if self.route and self.route.ok and self.route.nodes:
            idx = len(self.route.nodes) - 1
            if self.route.legs:
                idx = min(self.route.legs[self.leg_idx].end, idx)
            n = self.field.nodes.get(self.route.nodes[idx])
            if n:
                return n.lat, n.lon
        return None

    def distance_to_hold(self) -> float | None:
        hp = self.hold_point()
        if hp is None:
            return None
        return distance_m(self.sim.lat, self.sim.lon, *hp)
