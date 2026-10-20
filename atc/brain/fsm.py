"""ATC state machine: phases, positions, clearances, readback checks.

All decisions (runway, route, altitude, squawk, frequencies) are made here from
sim/world data. The LLM only supplies the PilotIntent.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from ..phase_tracker import PhaseTracker, on_runway
from ..phraseology import Phraseology, freq_text, wind
from ..state import Phase, PilotIntent, Position, SimState, Transmission
from ..world import navdata, taxi
from ..world.aptdat import Airport, AptIndex, distance_m
from ..world.cifp import (cifp_file, final_for_runway, parse_cifp, parse_finals, sids_for_runway,
                          star_entries, star_path, stars_for_runway)
from ..world.runways import runway_requirements, select_runway
from . import vectors

log = logging.getLogger("brain")

VHF = (118000, 137000)
ROUTE_POINT_PASSED_NM = 3.0   # this close to a route point: "direct" goes to the next one
STAR_END_NM = 5.0             # this close to the last fix of the STAR: vectors to final
DESCENT_FT = 10000            # first descent clearance from cruise
TOD_NM_PER_1000FT = 3.0       # top of descent: 3 NM per 1000 ft to lose
STAR_MIN_FT = 6000            # below this the arrival goes straight to vectors, no STAR

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
STEP_DOWN_FT = 2000   # vectors: next descent once the 3° path is this far below the assigned altitude
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
    acf_type: str | None = None         # from the flight plan
    filed_destination: str | None = None  # flight plan loaded (SimBrief): "flight planned route"
    route_points: list = field(default_factory=list)  # simbrief.RoutePoint: SID exit .. STAR entry
    star: str | None = None             # planned STAR (SimBrief)


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
        self.missed_turn = False               # "missed your turn, hold position and advise" outstanding
        self.leg_idx = 0                       # current clearance limit in route.legs
        self.cleared_runways: set[str] = set()  # runways cleared to cross on this route
        self.cleared_takeoff = False
        self.cleared_land = False
        self.pending: Pending | None = None
        self.last_tx: Transmission | None = None
        self.last_pilot_t = 0.0
        self.last_pilot_pos: Position | None = None   # station the pilot last talked to
        self.sim = SimState()
        self.gate: str | None = None
        # VATSIM first call "good day, A320 Lufthansa 4AB on stand 1": who is calling
        self.identified = False                 # callsign taken from the pilot's first call
        self.called_type: str | None = None
        self.stand: str | None = None
        self.dest_choices: list[str] = []       # several airports in the requested city
        # vectors to final (APPROACH phase)
        self.final: vectors.Final | None = None
        self.appr_kind, self.appr_variant = "ILS", ""
        self.vector: vectors.Vector | None = None
        self.vector_pos = Position.APPROACH
        self.appr_cleared = False
        self.established = False
        self.approach_handoff = False   # Center already sent us to Approach
        # STAR from the CIFP: flown to its last fix, then vectors to final
        self.star: str | None = None
        self.star_fixes: list[str] = []
        self._star_end: tuple[float, float] | None = None
        self._rng = random.Random(fc.callsign)

    # ------------------------------------------------------------------ helpers
    @property
    def field(self) -> Airport:
        if self.phase in ARRIVAL_PHASES and self.dest is not None:
            return self.dest
        return self.dep

    @property
    def acf_type(self) -> str:
        return self.sim.acf_icao or self.called_type or self.fc.acf_type or "A320"

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
        order = list(Position)
        if self.phase == Phase.ENROUTE or self.phase in ARRIVAL_PHASES:
            # Departure and Approach often share one frequency (KBTV 121.1): inbound, it is Approach
            order.remove(Position.APPROACH)
            order.insert(order.index(Position.DEPARTURE), Position.APPROACH)
        for pos in order:
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
        # before the first call the callsign is unknown: airline names, not the config fallback
        # (Whisper heard "Aeroflot" as "Airflot")
        who = self.phr.cs_spoken if self.identified else "Aeroflot, Rossiya, Lufthansa, Speedbird"
        if not self.identified and self.fc.filed_destination:  # callsign from the flight plan
            who = f"{self.phr.cs_spoken}, {who}"
        if self.star:   # "Buku three Echo arrival"
            from ..phraseology import procedure
            who = f"{who}, {procedure(self.star)} arrival"
        return (f"{stations}, good day, A320 {who}, radio check. {ap.icao} {ap.name}. Runway "
                + ", ".join(e.name for r in ap.runways for e in r.ends)
                + f". Taxi via {twys_spoken}, hold short. Squawk, QNH, flight level, cleared, readback, "
                  "decimal, on stand, request IFR clearance.")

    # --------------------------------------------------------------- readbacks
    def _check_readback(self, intent: PilotIntent) -> list[Transmission]:
        p = self.pending
        rb = dict(intent.readback)
        if "gate" in p.expected and not rb.get("gate") and intent.data.get("stand"):
            rb["gate"] = intent.data["stand"]  # "taxi via S5 to stand 39": the parser files it as a stand
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
        if slot == "gate":  # "stand 39" / "gate 039" / "39"
            def norm(x):
                return re.sub(r"STAND|GATE|PARKING|[^A-Z0-9]", "", str(x).upper()).lstrip("0")
            return norm(got) == norm(val)
        if slot == "taxiways":
            def names(x):
                return [str(n).upper().replace(" ", "") for n in (x if isinstance(x, list) else [x])]
            return names(got) == names(val)
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
        if slot == "gate":
            return "stand", str(val), str(val)
        if slot == "taxiways":
            from ..phraseology import taxiway
            return "via", ", ".join(val), ", ".join(taxiway(n) for n in val)
        if slot == "fix":
            from ..phraseology import letters
            return "direct", str(val), letters(str(val))
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
        self.last_pilot_pos = pos
        self._identify(intent)
        if pos == Position.DELIVERY and self.dest_choices and (code := self._chosen_destination(intent)):
            intent = PilotIntent("request_clearance", callsign=intent.callsign, raw_text=intent.raw_text,
                                 data={"destination": code})
        if intent.intent == "report_parked" and self.phase in (Phase.TAXI_IN, Phase.PARKED):
            if self.phase == Phase.TAXI_IN:
                self.reset_for_next_flight()  # the flight ends here; the next one starts from this field
            return [self._tx(pos, self.phr.custom("roger, good day"))]
        if (self.missed_turn and pos == Position.GROUND and not intent.unclear
                and self.phase in (Phase.TAXI_OUT, Phase.TAXI_IN)
                and intent.intent not in ("say_again", "radio_check", "request_qnh")):
            return self._reroute()  # "holding position" / "request taxi": the pilot advises
        if intent.unclear or (intent.intent == "other" and not (
                self.pending and pos == self.pending.position and self._mentions_expected(intent))):
            return [self._tx(pos, self.phr.say_again())]
        if intent.intent == "say_again":
            if self.last_tx and self.last_tx.position == pos:
                return [self._tx(pos, (self.last_tx.text, self.last_tx.spoken), self.last_tx.expects_readback)]
            return [self._tx(pos, self.phr.say_again())]
        if intent.intent == "radio_check":
            return [self._tx(pos, self.phr.radio_check(self.station_name(pos)))]
        if intent.intent == "unable":  # "negative" / "unable": drop what we were waiting for
            self.pending = None
            return [self._tx(pos, self.phr.roger())]
        if intent.intent == "request_qnh":
            return [self._tx(pos, self.phr.custom(*self.phr.qnh(self.sim.qnh_hpa)))]
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

    _GENERIC_FREQ_WORDS = {"CLEARANCE", "CLR", "CLD", "DEL", "DELIVERY", "GND", "GROUND", "TWR", "TOWER",
                           "APP", "APCH", "APPROACH", "DEP", "DEPARTURE", "RADAR", "CTR", "CENTER", "ATIS",
                           "CTAF", "UNICOM", "INTL", "INTERNATIONAL", "AIRPORT", "GENERAL", "THE"}

    def station_name(self, pos: Position) -> str:
        """'Pulkovo Delivery', 'Peterburg Approach': the name from the apt.dat frequency of this
        position ('PULKOVO DEL'), else of any position, else the airport name, else the city."""
        if pos == Position.CENTER:
            return "Center"
        ap = self.field

        def name_of(freqs) -> str | None:
            for f in freqs:
                word = re.split(r"[\s/-]", f.name.strip().upper())[0] if f.name.strip() else ""
                if word.isalpha() and len(word) > 2 and word not in self._GENERIC_FREQ_WORDS:
                    return word
            return None

        name = (name_of(ap.freqs.get(pos.name.lower(), []))
                or name_of(f for lst in ap.freqs.values() for f in lst)
                or name_of([type("F", (), {"name": ap.name})()])
                or ap.meta.get("city", "") or ap.name.split()[0])
        return f"{name.title()} {pos.value}"

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
            if self.fc.filed_destination in choices:  # "to Moscow" with UUEE filed
                choices = [self.fc.filed_destination]
            if len(choices) == 1:
                icao = choices[0]
            elif len(choices) > 1:
                self.dest_choices = choices
                return [self._tx(Position.DELIVERY, self.phr.custom(
                    f"several airports in {city}, say destination ICAO"))]
            else:
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
            self.fc.destination, self.runway, self.sid, self.assigned_alt, self.squawk, dep_khz,
            self._filed()), True)
        gnd = self.freq_of(Position.GROUND, self.dep)

        def ok():  # "readback correct" only with confirm_readback; silence otherwise
            self._set_phase(Phase.CLEARANCE)
            if gnd and gnd != self.freq_of(Position.DELIVERY, self.dep):
                return [self._tx(Position.DELIVERY, self.phr.readback_correct(
                    "ground", gnd, "for push and start", confirm=self.fc.confirm_readback))]
            return []

        self._expect(tx, {"runway": self.runway, "altitude": self.assigned_alt,
                          "squawk": self.squawk}, {"squawk"}, ok)
        return [tx]

    def _on_ground(self, it: PilotIntent):
        G = Position.GROUND
        if self.phase in (Phase.TAXI_IN, Phase.LANDING) or (
                self.phase in (Phase.APPROACH, Phase.ARRIVAL) and self.sim.on_ground):
            if it.intent in ("request_taxi", "check_in", "report_position"):
                return self._taxi_in(keep_gate=self.route is not None)
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
        # a bare "Ground, good day" after the clearance is not a taxi request (-> go ahead);
        # after push and start it is
        if it.intent == "request_taxi" or (it.intent == "check_in" and self.phase == Phase.PUSH_START):
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
        self._expect(tx, self._taxi_expected(None, hold), {"hold_short"})
        return [tx]

    def _reroute(self) -> list[Transmission]:
        """After a missed turn: a new route from where the aircraft is now, same destination."""
        log.info("missed turn: new taxi route from %.5f %.5f", self.sim.lat, self.sim.lon)
        self.pending = None
        return self._taxi_in(keep_gate=True) if self.phase == Phase.TAXI_IN else self._taxi_out()

    def _new_route(self, route: taxi.TaxiRoute | None) -> None:
        self.route = route
        self.leg_idx = 0
        self.route_joined = False
        self.missed_turn = False
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
        return max(ends, key=lambda e: wind_components(e, *self.sim.surface_wind)[0]).name

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

    def _taxi_in(self, keep_gate: bool = False) -> list[Transmission]:
        s = self.sim
        ap = self.field
        gate = next((g for g in ap.gates if g.name == self.gate), None) if keep_gate else None
        gate = gate or self._pick_gate(ap)
        cls = taxi.aircraft_class(self.acf_type)
        if gate is None:
            self.route = None
            return [self._tx(Position.GROUND, self.phr.custom("taxi to parking at your discretion"))]
        self.gate = gate.name
        self._new_route(taxi.route_to_point(ap, s.lat, s.lon, gate.lat, gate.lon, cls, gate.name))
        self._set_phase(Phase.TAXI_IN)
        hold = self._hold_word(self.route.legs[0].hold_short) if self.route.legs else None
        tx = self._tx(Position.GROUND, self.phr.taxi_to_gate(self.route, gate.name, hold), True)
        self._expect(tx, self._taxi_expected(gate.name, hold), {"hold_short"} if hold else set())
        return [tx]

    def _taxi_expected(self, gate: str | None, hold: str | None) -> dict:
        """Readback of a taxi clearance: stand, taxiways of the first leg, hold short."""
        exp: dict = {}
        if gate:
            exp["gate"] = gate
        if self.route and self.route.ok and self.route.legs and self.route.legs[0].taxiways:
            exp["taxiways"] = list(self.route.legs[0].taxiways)
        if hold:
            exp["hold_short"] = hold
        return exp

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
            tx = self._tx(T, self.phr.cleared_land(rwy, self._wind()), True)
            self._expect(tx, {"runway": rwy, "cleared_land": True}, set())
            return [tx]
        return None

    def _takeoff_clearance(self) -> list[Transmission]:
        rwy = self.runway
        on = on_runway(self.dep, self.sim.lat, self.sim.lon)
        self.cleared_takeoff = True
        tx = self._tx(Position.TOWER, self.phr.cleared_takeoff(
            rwy, self._wind()), True)
        self._set_phase(Phase.TAKEOFF)
        self._expect(tx, {"runway": rwy, "cleared_takeoff": True}, {"cleared_takeoff"} if not on else set())
        return [tx]

    def _on_departure(self, it: PilotIntent):
        D = Position.DEPARTURE
        if it.intent == "request_direct" and self.phase == Phase.DEPARTURE:
            return self._direct(D, it)
        if self.phase in (Phase.TAKEOFF, Phase.DEPARTURE) and it.intent in ("check_in", "report_position", "request_climb"):
            self._set_phase(Phase.DEPARTURE)
            new_alt = self.next_climb_alt() or self.assigned_alt
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
        if (out := self._runway_request(C, it)) is not None:
            return out
        if it.intent in ("check_in", "request_climb", "report_position") and self.phase in (
                Phase.DEPARTURE, Phase.ENROUTE):
            self._set_phase(Phase.ENROUTE)
            alt = self.fc.cruise_ft
            self._assign_alt(alt)
            verb = "identified, climb" if self.sim.altitude_ft < alt - 300 else "identified, maintain"
            t, s = self.phr.climb(alt, verb)
            if self.assigned_hdg is not None:  # end of radar vectors
                self._assign_heading(None)
                nxt = self.next_route_point()
                nav_t, nav_s = self.phr.end_vectors(nxt.ident if nxt else self.fc.destination)
                t, s = f"{t}, {nav_t}", f"{s}, {nav_s}"
            tx = self._tx(C, (t, s), True)
            self._expect(tx, {"altitude": alt}, set())
            return [tx]
        if it.intent == "request_direct" and self.phase in (Phase.ENROUTE, Phase.ARRIVAL):
            return self._direct(C, it)
        if it.intent == "request_descent" and self.phase in (Phase.ENROUTE, Phase.ARRIVAL):
            want = it.data.get("altitude") or (it.data.get("flight_level") or 0) * 100
            return self._descend(int(want) if want else DESCENT_FT)
        if it.intent == "request_approach" and self.phase == Phase.ENROUTE:
            self._set_phase(Phase.ARRIVAL)
        if it.intent in ("request_approach", "check_in", "report_position") and self.phase == Phase.ARRIVAL:
            return self.handoff_approach() or self._arrival(C)
        if self.phase == Phase.APPROACH and self.vector_pos == C:
            return self._on_vectors(it)
        return None

    def _descend(self, alt: int) -> list[Transmission]:
        """Center: 'descend FL100[, cleared BUKU3E arrival, expect ILS approach runway 24L]'."""
        C = Position.CENTER
        self._set_phase(Phase.ARRIVAL)
        self._load_dest()
        self._assign_alt(alt)
        t, sp = self.phr.climb(alt, "descend")
        if self.star is None and self._assign_star():
            self._assign_heading(None)   # the STAR ends any radar vectors
            st, ss = self.phr.star_clearance(self.star, self._appr_name(), self.arr_runway)
            t, sp = f"{t}, {st}", f"{sp}, {ss}"
        tx = self._tx(C, (t, sp), True)
        self._expect(tx, {"altitude": alt}, set())
        return [tx]

    def descent_tick(self) -> list[Transmission]:
        """Monitor: Center starts the descent at the top of descent, 3 NM per 1000 ft to the field,
        or (with a planned STAR) to its entry point at DESCENT_FT, whichever comes first."""
        s = self.sim
        if self.phase != Phase.ENROUTE or self.pending is not None or not self.tracker.airborne:
            return []
        if s.altitude_ft < DESCENT_FT + 2000 or s.vs_fpm > 500:
            return []   # low already, or still climbing
        if self.position_on(s.tx_khz) != Position.CENTER:
            return []
        if self.dest is not None:
            dest, elev = (self.dest.lat, self.dest.lon), self.dest.elevation_ft
        elif self.index and self.fc.destination and self.fc.destination in self.index.entries:
            _, _, lat, lon = self.index.entries[self.fc.destination][:4]
            if lat is None:
                return []
            dest, elev = (lat, lon), 0
        else:
            return []
        nm = distance_m(s.lat, s.lon, *dest) / 1852
        tod = nm <= TOD_NM_PER_1000FT * (s.altitude_ft - elev) / 1000
        if not tod and self.fc.star and self.fc.route_points:
            entry = self.fc.route_points[-1]   # the plan's points end at the STAR entry
            tod = distance_m(s.lat, s.lon, entry.lat, entry.lon) / 1852 <= \
                TOD_NM_PER_1000FT * (s.altitude_ft - DESCENT_FT) / 1000
        if not tod:
            return []
        log.info("top of descent: %.0f NM to %s at %.0f ft", nm, self.fc.destination, s.altitude_ft)
        out = self._descend(DESCENT_FT)
        for tx in out:
            tx.proactive = True
        return out

    def _direct(self, pos: Position, it: PilotIntent) -> list[Transmission]:
        """'request direct VUZPU': approved when the point is in the nav data (ends radar vectors)."""
        fix = it.data.get("fix") or it.readback.get("fix")
        if not fix:
            return [self._tx(pos, self.phr.say_again())]
        if navdata.point_known(self.xp_root, fix) is False:
            return [self._tx(pos, self.phr.unknown_point(fix))]
        self._assign_heading(None)
        tx = self._tx(pos, self.phr.direct(fix), True)
        self._expect(tx, {"fix": fix}, set())
        return [tx]

    def _on_approach(self, it: PilotIntent):
        A = Position.APPROACH
        if (out := self._runway_request(A, it)) is not None:
            return out
        if self.phase in (Phase.ARRIVAL, Phase.ENROUTE) and it.intent in (
                "check_in", "request_approach", "report_position", "request_descent"):
            self._set_phase(Phase.ARRIVAL)
            return self._arrival(A)
        if self.phase == Phase.APPROACH:
            return self._on_vectors(it)
        return self._on_departure(it) if self.phase in (Phase.TAKEOFF, Phase.DEPARTURE) else None

    def _runway_request(self, pos: Position, it: PilotIntent) -> list[Transmission] | None:
        """'request runway 06R' / 'request ILS runway 06R' on the arrival: approved when the
        destination has it and it is long enough; the STAR and vectors are redone for it."""
        rwy = it.data.get("runway") or it.readback.get("runway")
        if it.intent not in ("request_runway", "request_approach") or not rwy:
            return None
        if self.phase not in (Phase.ENROUTE, Phase.ARRIVAL, Phase.APPROACH) or self.appr_cleared:
            return None
        self._load_dest()
        ap = self.dest
        if ap is None:
            return None
        end = ap.runway_end(rwy)
        need_len, _ = runway_requirements(self.acf_type)
        if end is None or end.length_m < need_len:
            return [self._tx(pos, self.phr.unable_runway(rwy))]
        if rwy == self.arr_runway:
            return None
        log.info("pilot requests runway %s instead of %s", rwy, self.arr_runway)
        self.arr_runway, self.final, self.vector = rwy, None, None
        ok_t, ok_s = self.phr.runway_approved(rwy)
        if self.phase == Phase.APPROACH:
            out = self._start_vectors(pos)
            tx = out[0]
            tx.text, tx.spoken = f"{ok_t}, {tx.text.split(', ', 1)[1]}", f"{ok_s}, {tx.spoken.split(', ', 1)[1]}"
            return out
        had_star, self.star, self.star_fixes = self.star is not None, None, []
        if had_star and self._assign_star():
            self._assign_heading(None)
            t, sp = self.phr.star_clearance(self.star, self._appr_name(), rwy)
        else:
            self.final = self._build_final()
            t, sp = self.phr.expect_approach(self._appr_name(), rwy)
        tx = self._tx(pos, (f"{ok_t}, {t}", f"{ok_s}, {sp}"), True)
        self._expect(tx, {"runway": rwy}, set())
        return [tx]

    # ------------------------------------------------------- route and STAR
    def next_route_point(self):
        """Next point of the planned route ahead of us (SimBrief), None without a plan or past its end."""
        pts = self.fc.route_points
        if not pts:
            return None
        s = self.sim
        d = [distance_m(s.lat, s.lon, p.lat, p.lon) for p in pts]
        i = min(range(len(pts)), key=d.__getitem__)
        if d[i] < ROUTE_POINT_PASSED_NM * 1852 or (
                i + 1 < len(pts) and distance_m(pts[i].lat, pts[i].lon, pts[i + 1].lat, pts[i + 1].lon) > d[i + 1]):
            i += 1   # at it, or already between it and the next one
        return pts[i] if i < len(pts) else None

    def _assign_star(self) -> bool:
        """STAR for the arrival runway from the destination's CIFP: the planned one if it serves the
        runway, else one joined at a point of the route (same entry as the planned one first)."""
        self._load_dest()
        ap = self.dest
        f = cifp_file(self.xp_root, ap.icao) if self.xp_root and ap else None
        if not f:
            return False
        self.arr_runway = self.arr_runway or self._select_arr_runway()
        self.final = self.final or self._build_final()   # approach kind for "expect ILS approach"
        cands = sorted(stars_for_runway(parse_cifp(f), self.arr_runway), key=lambda p: p.ident)
        route = [p.ident for p in self.fc.route_points]
        planned = next((p for p in cands if p.ident == self.fc.star), None)
        entry = None
        if planned is None:
            plan_entries = next((star_entries(p) for p in cands if p.ident[:-1] == (self.fc.star or "")[:-1]), [])

            def score(p):
                ents = star_entries(p)
                return (not any(e in plan_entries for e in ents), not any(e in route for e in ents))
            joined = [p for p in cands if any(e in route for e in star_entries(p)) or
                      any(e in plan_entries for e in star_entries(p))]
            planned = min(joined, key=score, default=None)
        if planned is None:
            log.info("no STAR for runway %s joined from the route %s", self.arr_runway, route)
            return False
        entry = next((e for e in star_entries(planned) if e in route), None)
        self.star = planned.ident
        self.star_fixes = star_path(planned, self.arr_runway, entry)
        self._star_end = None
        log.info("STAR %s runway %s: %s", self.star, self.arr_runway, " ".join(self.star_fixes))
        return bool(self.star_fixes)

    def star_end(self) -> tuple[float, float] | None:
        """Last fix of the STAR: from the flight plan, else the X-Plane nav data near the destination."""
        if self._star_end is None and self.star_fixes:
            last = self.star_fixes[-1]
            p = next((p for p in self.fc.route_points if p.ident == last), None)
            ap = self.dest or self.field
            self._star_end = (p.lat, p.lon) if p else navdata.point_near(self.xp_root, last, ap.lat, ap.lon)
        return self._star_end

    def star_done(self) -> bool:
        """Near the last fix of the STAR (or closer to the field than it is): time for vectors.
        A STAR whose last fix is not in the nav data cannot be followed: vectors right away."""
        end = self.star_end()
        if end is None:
            return True
        s, ap = self.sim, self.dest or self.field
        to_end = distance_m(s.lat, s.lon, *end)
        return to_end < STAR_END_NM * 1852 or \
            distance_m(s.lat, s.lon, ap.lat, ap.lon) < distance_m(*end, ap.lat, ap.lon) - 2 * 1852

    def _appr_name(self) -> tuple[str, str]:
        return self.phr.approach_name(self.appr_kind, self.appr_variant)

    def _arrival(self, pos: Position) -> list[Transmission]:
        """First call to the arrival controller: on the STAR "continue", otherwise vectors."""
        fresh = False
        if self.star is None and self.fc.route_points and self.sim.altitude_ft > STAR_MIN_FT:
            fresh = self._assign_star()   # nobody asked for descent: the STAR comes now
        if self.star is None or self.star_done():
            return self._start_vectors(pos)
        if fresh:
            self._assign_heading(None)
        f = self.final or self._build_final()
        alt = -(-(f.faf_alt + 2000) // 1000) * 1000 if f else None   # FAF + 2000, up to the next 1000
        if alt and self.sim.altitude_ft < alt + 200:
            alt = None
        if alt:
            self._assign_alt(alt)
        tx = self._tx(pos, self.phr.continue_star(self.star, self._appr_name(), self.arr_runway, alt, fresh), True)
        self._expect(tx, {"altitude": alt} if alt else {}, set())
        return [tx]

    def star_tick(self) -> list[Transmission]:
        """Monitor: on the STAR with the arrival controller, vectors start at its last fix."""
        if self.phase != Phase.ARRIVAL or self.star is None or self.pending is not None:
            return []
        pos = Position.APPROACH if self.approach_handoff else Position.CENTER
        if not self.tracker.airborne or self.last_pilot_pos != pos or self.position_on(self.sim.tx_khz) != pos:
            return []   # not checked in yet: the check-in starts the vectors
        if not self.star_done():
            return []
        out = self._start_vectors(pos)
        for tx in out:
            tx.proactive = True
        return out

    # ----------------------------------------------------- vectors to final
    def handoff_approach(self) -> list[Transmission]:
        """Center -> Approach, when Approach has a frequency of its own."""
        app, ctr = self.freq_of(Position.APPROACH), self.freq_of(Position.CENTER)
        if not app or app == ctr or self.approach_handoff:
            return []
        self.approach_handoff = True
        self._set_phase(Phase.ARRIVAL)
        tx = self._tx(Position.CENTER, self.phr.contact("approach", app), True)
        self._expect(tx, {"frequency": app}, set())
        return [tx]

    def _build_final(self) -> vectors.Final | None:
        """Final approach course from CIFP (FAF, altitude, course), else from the runway."""
        self._load_dest()
        ap = self.field
        self.arr_runway = self.arr_runway or self._select_arr_runway()
        end = ap.runway_end(self.arr_runway)
        if end is None:
            return None
        fa = None
        f = cifp_file(self.xp_root, ap.icao) if self.xp_root else None
        if f:
            fa = final_for_runway(parse_finals(f), self.arr_runway)
        self.appr_kind, self.appr_variant = (fa.kind, fa.variant) if fa else ("ILS", "")
        course_mag = fa.course_mag if fa and fa.course_mag is not None else (end.heading_true - self.magvar) % 360
        faf_alt = fa.faf_alt if fa and fa.faf_alt else int(round((ap.elevation_ft + 2000) / 500.0) * 500)
        faf_dist = fa.faf_dist_nm if fa and fa.faf_dist_nm else \
            vectors.faf_distance_from_altitude(faf_alt, ap.elevation_ft)
        log.info("final %s runway %s: course %03.0f, FAF %s %d ft %.1f NM", self.appr_kind, self.arr_runway,
                 course_mag, fa.faf if fa else "-", faf_alt, faf_dist)
        return vectors.Final(end.lat, end.lon, end.heading_true, course_mag, faf_dist, faf_alt, ap.elevation_ft)

    def _start_vectors(self, pos: Position) -> list[Transmission]:
        self.final = self._build_final()
        if self.final is None:
            return [self._tx(pos, self.phr.say_again())]
        self.vector_pos = pos
        self.appr_cleared = self.established = False
        self._set_phase(Phase.APPROACH)
        f = self.final
        x, y = f.frame(self.sim.lat, self.sim.lon)
        v = vectors.first_vector(f, x, y)
        alt = vectors.descent_target(f, v, x, y)
        if self.sim.altitude_ft < alt + 200:
            alt = None   # already low enough: keep what was assigned
        self.vector = v
        turn = self._turn_direction(v.heading)
        self._assign_heading(v.heading)
        expected: dict = {"heading": v.heading}
        if alt:
            self._assign_alt(alt)
            expected["altitude"] = alt
        appr = self.phr.approach_name(self.appr_kind, self.appr_variant)
        tx = self._tx(pos, self.phr.vectors_for_approach(appr, self.arr_runway, v.heading, turn, alt), True)
        self._expect(tx, expected, set())
        return [tx]

    def _on_vectors(self, it: PilotIntent):
        """Pilot calls while being vectored: restart after a program restart, else repeat."""
        if self.vector is None and not self.appr_cleared:
            return self._start_vectors(self.vector_pos)
        if it.intent in ("report_final", "report_position") and self.appr_cleared:
            return self._established_handoff() or None
        if it.intent in ("check_in", "request_approach", "report_position") and self.last_tx:
            if self.last_tx.position == self.vector_pos:
                return [self._tx(self.vector_pos, (self.last_tx.text, self.last_tx.spoken),
                                 self.last_tx.expects_readback)]
        return None

    def vector_tick(self) -> list[Transmission]:
        """Monitor: next turn, the approach clearance, then Tower once established."""
        f, v = self.final, self.vector
        if self.phase != Phase.APPROACH or f is None or v is None or self.pending is not None:
            return []
        if not self.tracker.airborne or self.sim.on_ground:
            return []
        x, y = f.frame(self.sim.lat, self.sim.lon)
        if self.appr_cleared:
            if not self.established and vectors.established(f, x, y, self.sim.heading_mag):
                return self._established_handoff()
            return []
        nv = vectors.next_vector(f, v, x, y)
        if nv and nv.leg == "intercept":
            return self._clear_approach(nv)
        if nv:
            self.vector = nv
            alt = vectors.descent_target(f, nv, x, y)
            alt = alt if (self.assigned_alt or 0) > alt else None
            turn = self._turn_direction(nv.heading)
            self._assign_heading(nv.heading)
            expected: dict = {"heading": nv.heading}
            if alt:
                self._assign_alt(alt)
                expected["altitude"] = alt
            tx = self._tx(self.vector_pos, self.phr.vector(nv.heading, turn, alt, leg=nv.leg), True)
            tx.proactive = True
            self._expect(tx, expected, set())
            return [tx]
        if v.leg == "intercept" and (abs(y) <= 2 * vectors.INTERCEPT_TURN_NM or x <= f.gate_nm + 8):
            return self._clear_approach(None, x, y)   # straight in: already on the intercept heading
        alt = vectors.descent_target(f, v, x, y)
        if (self.assigned_alt or self.sim.altitude_ft) - alt >= STEP_DOWN_FT:
            self._assign_alt(alt)   # step down along the 3° path, STEP_DOWN_FT or more at a time
            tx = self._tx(self.vector_pos, self.phr.climb(alt, "descend"), True)
            tx.proactive = True
            self._expect(tx, {"altitude": alt}, set())
            return [tx]
        return []

    def _clear_approach(self, nv: vectors.Vector | None, x: float | None = None,
                        y: float | None = None) -> list[Transmission]:
        f = self.final
        hdg = turn = None
        if nv is not None:
            self.vector = nv
            hdg, turn = nv.heading, self._turn_direction(nv.heading)
        if x is None:
            x, y = f.frame(self.sim.lat, self.sim.lon)
        # "maintain ... until established": below the glideslope where the localizer is joined,
        # never a climb over what is already assigned
        alt = vectors.intercept_altitude(f, vectors.join_distance(f, self.vector, x, y))
        alt = max(f.faf_alt, min(alt, self.assigned_alt or alt))
        self.appr_cleared = True
        self._assign_heading(None)   # the pilot now flies the localizer, not a heading
        self._assign_alt(alt)
        verb = "descend and maintain" if self.sim.altitude_ft > alt + 300 else "maintain"
        appr = self.phr.approach_name(self.appr_kind, self.appr_variant)
        tx = self._tx(self.vector_pos, self.phr.intercept_clearance(
            appr, self.appr_kind, self.arr_runway, hdg, turn, alt, verb), True)
        tx.proactive = True
        expected: dict = {"runway": self.arr_runway, "altitude": alt}
        if hdg is not None:
            expected["heading"] = hdg
        self._expect(tx, expected, set())
        return [tx]

    def _established_handoff(self) -> list[Transmission]:
        self.established = True
        self.assigned_alt = None   # descending on the glideslope is no altitude bust
        twr = self._apt_freq(self.field, "tower", "unicom")
        if not twr or twr == self.freq_of(self.vector_pos):
            return []
        tx = self._tx(self.vector_pos, self.phr.contact("tower", twr), True)
        tx.proactive = True
        self._expect(tx, {"frequency": twr}, set())
        return [tx]

    # ------------------------------------------------------------ decisions
    def next_climb_alt(self) -> int | None:
        """Next level on the climb: Departure takes us to the top of its airspace
        (handoff_center_ft), Center to cruise. None = nothing higher to give."""
        cur = self.assigned_alt or 0
        top = self.fc.handoff_center_ft
        alt = min(self.fc.cruise_ft, top) if self.phase in (Phase.TAKEOFF, Phase.DEPARTURE) and cur < top \
            else self.fc.cruise_ft
        return alt if alt > cur else None

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
        if nxt := self.next_route_point():
            true = bearing_deg(self.sim.lat, self.sim.lon, nxt.lat, nxt.lon)
        elif self.index and self.fc.destination and self.fc.destination in self.index.entries:
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

    def _wind(self) -> tuple[str, str]:
        """Surface wind as Tower gives it: magnetic."""
        return wind(*self.sim.surface_wind, self.magvar)

    def _select_dep_runway(self) -> str:
        e = select_runway(self.dep, *self.sim.surface_wind, acf_type=self.acf_type)
        return e.name if e else "??"

    def _select_arr_runway(self) -> str:
        ap = self.dest or self.dep
        e = select_runway(ap, *self.sim.surface_wind, acf_type=self.acf_type)
        log.info("arrival runway %s at %s, surface wind %03.0f/%.0f", e.name if e else "??", ap.icao,
                 *self.sim.surface_wind)
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
        if it.intent not in ("check_in", "report_position", "request_climb", "request_descent", "other",
                             "report_final", "request_approach"):
            return False
        exp = set(self.pending.expected)
        got = set(it.readback) | ({"gate"} if it.data.get("stand") else set())
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

    def _filed(self) -> bool:
        """The clearance follows the filed plan only when the pilot asked for its destination."""
        return bool(self.fc.filed_destination) and self.fc.filed_destination == self.fc.destination

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
        if ev.took_off:
            self._new_route(None)  # the taxi-out route must not be checked at the destination
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
    STATE_FIELDS = ("runway", "arr_runway", "sid", "star", "star_fixes", "squawk", "assigned_alt", "assigned_hdg",
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
            self.squawk, self.freq_of(Position.DEPARTURE, self.dep), self._filed()))
        self._set_phase(Phase.CLEARANCE)
        return tx

    def reset_for_next_flight(self) -> None:
        if self.dest is not None:
            self.dep, self.dest = self.dest, None
        self.fc.destination = self.fc.filed_destination = None  # the plan was for the flight just flown
        self.runway = self.arr_runway = self.sid = self.squawk = None
        self.assigned_alt = None
        self._new_route(None)
        self.cleared_takeoff = self.cleared_land = False
        self.pending = None
        self.last_pilot_pos = None
        self.identified, self.stand, self.dest_choices = False, None, []  # next flight: new first call
        self.final = self.vector = None
        self.appr_cleared = self.established = self.approach_handoff = False
        self.star, self.star_fixes, self._star_end = None, [], None
        self.fc.route_points, self.fc.star = [], None
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
