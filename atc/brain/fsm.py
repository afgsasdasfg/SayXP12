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
        self.route: taxi.TaxiRoute | None = None
        self.route_joined = False
        self.cleared_takeoff = False
        self.cleared_land = False
        self.pending: Pending | None = None
        self.last_tx: Transmission | None = None
        self.last_pilot_t = 0.0
        self.sim = SimState()
        self.gate: str | None = None
        self._rng = random.Random(fc.callsign)

    # ------------------------------------------------------------------ helpers
    @property
    def field(self) -> Airport:
        if self.phase in ARRIVAL_PHASES and self.dest is not None:
            return self.dest
        return self.dep

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
        return (f"{self.phr.cs_spoken}. {ap.icao} {ap.name}. Runway "
                + ", ".join(e.name for r in ap.runways for e in r.ends)
                + ". Taxiways " + ", ".join(twys) + ". Squawk, QNH, flight level, cleared, readback.")

    # --------------------------------------------------------------- readbacks
    def _check_readback(self, intent: PilotIntent) -> list[Transmission]:
        p = self.pending
        rb = intent.readback
        wrong: list[tuple[str, str, str]] = []
        for slot, val in p.expected.items():
            got = rb.get(slot)
            if slot == "altitude" and got is None and rb.get("flight_level"):
                got = int(rb["flight_level"]) * 100
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
        return out

    @staticmethod
    def _same(slot: str, got, val) -> bool:
        if slot == "altitude":
            return abs(int(got) - int(val)) < 50
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
        if intent.unclear or intent.intent == "other":
            return [self._tx(pos, self.phr.say_again())]
        if intent.intent == "say_again":
            if self.last_tx and self.last_tx.position == pos:
                return [self._tx(pos, (self.last_tx.text, self.last_tx.spoken), self.last_tx.expects_readback)]
            return [self._tx(pos, self.phr.say_again())]
        if self.pending and pos == self.pending.position and intent.intent in ("readback", "acknowledge"):
            return self._check_readback(intent)
        if intent.intent in ("readback", "acknowledge"):
            self.pending = None
            return []  # nothing outstanding: silence is the realistic answer
        exp = self.expected_position()
        handler = getattr(self, f"_on_{pos.name.lower()}", None)
        out = handler(intent) if handler else None
        if out is not None:
            return out
        if pos != exp:
            return [self._tx(pos, self.phr.wrong_frequency(exp.value.lower(), self.freq_of(exp)))]
        return [self._tx(pos, self.phr.say_again())]

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
        self.fc.destination = dest.upper()
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
            return [self._tx(G, self.phr.push_start())]
        if it.intent == "request_startup":
            return [self._tx(G, self.phr.startup())]
        if it.intent == "request_taxi" or (it.intent == "check_in" and self.phase in (Phase.CLEARANCE, Phase.PUSH_START)):
            return self._taxi_out()
        return None

    def _taxi_out(self) -> list[Transmission]:
        s = self.sim
        if self.runway is None:
            self.runway = self._select_dep_runway()
        cls = taxi.aircraft_class(s.acf_icao or "A320")
        self.route = taxi.route_to_runway(self.dep, s.lat, s.lon, self.runway, cls)
        self.route_joined = False
        if not self.route.ok:
            log.warning("taxi route not found, using fallback instruction")
        tx = self._tx(Position.GROUND, self.phr.taxi(self.route, self.runway, s.qnh_hpa), True)
        self._set_phase(Phase.TAXI_OUT)
        self._expect(tx, {"hold_short": self.runway}, {"hold_short"})
        return [tx]

    def _taxi_in(self) -> list[Transmission]:
        s = self.sim
        ap = self.field
        gate = self._pick_gate(ap)
        cls = taxi.aircraft_class(s.acf_icao or "A320")
        if gate is None:
            self.route = None
            return [self._tx(Position.GROUND, self.phr.custom("taxi to parking at your discretion"))]
        self.gate = gate.name
        self.route = taxi.route_to_point(ap, s.lat, s.lon, gate.lat, gate.lon, cls, gate.name)
        self.route_joined = False
        self._set_phase(Phase.TAXI_IN)
        return [self._tx(Position.GROUND, self.phr.taxi_to_gate(self.route, gate.name))]

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
            tx = self._tx(D, self.phr.radar_contact(new_alt), True)
            self._expect(tx, {"altitude": new_alt}, set())
            return [tx]
        return self._on_center(it) if self.phase in (Phase.ENROUTE, Phase.ARRIVAL, Phase.APPROACH) else None

    def _on_center(self, it: PilotIntent):
        C = Position.CENTER
        if it.intent in ("check_in", "request_climb", "report_position") and self.phase in (
                Phase.DEPARTURE, Phase.ENROUTE):
            self._set_phase(Phase.ENROUTE)
            alt = self.fc.cruise_ft
            self._assign_alt(alt)
            tx = self._tx(C, self.phr.climb(alt, "identified, climb") if self.sim.altitude_ft < alt - 300
                          else self.phr.climb(alt, "identified, maintain"), True)
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

    def _select_dep_runway(self) -> str:
        e = select_runway(self.dep, self.sim.wind_dir_true, self.sim.wind_speed_kt)
        return e.name if e else "??"

    def _select_arr_runway(self) -> str:
        ap = self.dest or self.dep
        e = select_runway(ap, self.sim.wind_dir_true, self.sim.wind_speed_kt)
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
        jets = [g for g in gates if "jets" in g.types] if taxi.aircraft_class(self.sim.acf_icao or "A320") >= "C" else gates
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

    def reset_for_next_flight(self) -> None:
        if self.dest is not None:
            self.dep, self.dest = self.dest, None
        self.fc.destination = None
        self.runway = self.arr_runway = self.sid = self.squawk = None
        self.assigned_alt = None
        self.route = None
        self.cleared_takeoff = self.cleared_land = False
        self.pending = None
        self._set_phase(Phase.PARKED)

    def hold_point(self) -> tuple[float, float] | None:
        if self.route and self.route.ok and self.route.nodes:
            n = self.field.nodes.get(self.route.nodes[-1])
            if n:
                return n.lat, n.lon
        return None

    def distance_to_hold(self) -> float | None:
        hp = self.hold_point()
        if hp is None:
            return None
        return distance_m(self.sim.lat, self.sim.lon, *hp)
