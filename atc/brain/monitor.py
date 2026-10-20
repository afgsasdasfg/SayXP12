"""Proactive ATC: rules checked about once a second; each decides *what* to say,
phraseology decides *how*. Every rule has its own cooldown so ATC does not spam.
"""
from __future__ import annotations

import logging
import time

from ..phase_tracker import on_runway
from ..state import Phase, Position, SimState, Transmission
from ..world.aptdat import distance_m
from ..world.taxi import distance_to_route
from .fsm import Brain

log = logging.getLogger("monitor")

NO_READBACK_S = 10.0
MAX_REPEATS = 2
HOLD_HANDOFF_M = 70.0
CROSSING_CLEARANCE_M = 120.0
TAXI_SPEED_KT = 30.0
HDG_DEV_DEG = 20.0
ROUTE_JOIN_M = 25.0
ROUTE_LOST_M = 60.0
ALT_DEV_FT = 300.0
APPROACH_HANDOFF_NM = 40.0
HANDOFF_LEAD_FT = 1500.0    # Departure hands over to Center this far below the handoff altitude
CLIMB_LEAD_FT = 1500.0      # further climb is given this far below the cleared level, before level-off
LEVEL_FT = 300.0


class Monitor:
    def __init__(self, brain: Brain):
        self.b = brain
        self._last: dict[str, float] = {}
        self._fast_ticks = 0

    def _ready(self, rule: str, cooldown: float) -> bool:
        now = time.time()
        if now - self._last.get(rule, 0.0) < cooldown:
            return False
        self._last[rule] = now
        return True

    def tick(self, s: SimState) -> list[Transmission]:
        b = self.b
        b.update(s)
        out: list[Transmission] = []
        for rule in (self._no_readback, self._handoff_departure, self._handoff_center,
                     self._further_climb, self._top_of_descent,
                     self._handoff_approach, self._star_vectors, self._approach_vectors,
                     self._hold_short_handoff, self._runway_incursion, self._route_deviation,
                     self._taxi_speed, self._heading_deviation,
                     self._altitude_deviation, self._runway_vacated):
            try:
                out += rule(s)
            except Exception:  # noqa: BLE001 - one bad rule must not stop the others
                log.exception("rule %s failed", rule.__name__)
        return out

    # ---------------------------------------------------------------- rules
    def _no_readback(self, s: SimState) -> list[Transmission]:
        p = self.b.pending
        if not p or self.b.last_pilot_t != p.pilot_t_at_issue or time.time() - p.t < NO_READBACK_S:
            return []
        if p.repeats >= MAX_REPEATS:
            log.info("no readback after %d repeats, dropping", p.repeats)
            self.b.pending = None
            return []
        p.repeats += 1
        p.t = time.time()
        tx = self.b._tx(p.position, self.b.phr.how_do_you_read((p.tx.text, p.tx.spoken)), True)
        p.tx = tx
        return [tx]

    def _handoff_departure(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase != Phase.TAKEOFF or not b.tracker.airborne or s.agl_ft < b.fc.handoff_departure_agl:
            return []
        b._set_phase(Phase.DEPARTURE)
        b.pending = None
        dep = b.freq_of(Position.DEPARTURE)
        if not dep or dep == b.freq_of(Position.TOWER):
            return []
        tx = b._tx(Position.TOWER, b.phr.contact("departure", dep), True)
        b._expect(tx, {"frequency": dep}, set())
        return [tx]

    def _handoff_center(self, s: SimState) -> list[Transmission]:
        b = self.b
        # before the cleared level, not at it: levelled at 9990 under FL100 it never fired
        top = b.fc.handoff_center_ft
        if b.assigned_alt and b.assigned_alt >= top:
            top = min(top, b.assigned_alt - HANDOFF_LEAD_FT)
        if b.phase != Phase.DEPARTURE or not b.tracker.airborne or s.altitude_ft < top:
            return []
        b._set_phase(Phase.ENROUTE)
        ctr = b.freq_of(Position.CENTER)
        if ctr == b.freq_of(Position.DEPARTURE):
            return []
        tx = b._tx(Position.DEPARTURE, b.phr.contact("center", ctr), True)
        b._expect(tx, {"frequency": ctr}, set())
        return [tx]

    def _further_climb(self, s: SimState) -> list[Transmission]:
        """Climbing towards a level below cruise: the next climb comes before the level-off,
        as a real controller gives it, not after minutes level at FL100."""
        b = self.b
        if b.phase not in (Phase.DEPARTURE, Phase.ENROUTE) or b.pending is not None or not b.tracker.airborne:
            return []
        if not b.assigned_alt or s.altitude_ft < b.assigned_alt - CLIMB_LEAD_FT:
            return []
        if b.phase == Phase.DEPARTURE and b.assigned_alt >= b.fc.handoff_center_ft:
            return []  # top of Departure's airspace: the Center handoff gives the climb
        alt = b.next_climb_alt()
        exp = b.expected_position()
        if alt is None or b.position_on(s.tx_khz) != exp:
            return []  # nothing higher, or the pilot is not on our frequency
        level = abs(s.altitude_ft - b.assigned_alt) < LEVEL_FT
        if b.last_pilot_pos != exp and not level:
            return []  # not checked in yet: the check-in call will get the climb
        if not self._ready("further_climb", 20.0):
            return []
        verb = "climb" if b.last_pilot_pos == exp else "identified, climb"
        b._assign_alt(alt)
        tx = b._tx(exp, b.phr.climb(alt, verb), True)
        tx.proactive = True
        b._expect(tx, {"altitude": alt}, set())
        return [tx]

    def _handoff_approach(self, s: SimState) -> list[Transmission]:
        """Inbound within APPROACH_HANDOFF_NM of the destination: Center hands over to Approach."""
        b = self.b
        if b.phase != Phase.ARRIVAL or b.approach_handoff or b.pending is not None or not b.tracker.airborne:
            return []
        ap = b.field
        if distance_m(s.lat, s.lon, ap.lat, ap.lon) > APPROACH_HANDOFF_NM * 1852:
            return []
        out = b.handoff_approach()
        for tx in out:
            tx.proactive = True
        return out

    def _top_of_descent(self, s: SimState) -> list[Transmission]:
        return self.b.descent_tick()

    def _star_vectors(self, s: SimState) -> list[Transmission]:
        return self.b.star_tick()

    def _approach_vectors(self, s: SimState) -> list[Transmission]:
        return self.b.vector_tick()

    def _hold_short_handoff(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase not in (Phase.TAXI_OUT, Phase.TAXI_IN) or b.pending is not None:
            return []
        d = b.distance_to_hold()
        if b.has_next_leg():
            # approaching a runway on the route: clear the crossing early so the pilot
            # doesn't have to stop (they still must not cross before hearing it)
            if d is not None and d <= CROSSING_CLEARANCE_M:
                return b.continue_taxi()
            return []
        if b.phase != Phase.TAXI_OUT:
            return []
        if d is None or d > HOLD_HANDOFF_M or s.groundspeed_kt > 15:
            return []
        b._set_phase(Phase.HOLDING)
        twr = b.freq_of(Position.TOWER)
        if twr == b.freq_of(Position.GROUND):
            return []
        tx = b._tx(Position.GROUND, b.phr.contact("tower", twr), True)
        b._expect(tx, {"frequency": twr}, set())
        return [tx]

    def _runway_incursion(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase not in (Phase.TAXI_OUT, Phase.HOLDING) or b.cleared_takeoff or not b.runway:
            return []
        if s.groundspeed_kt < 2 or not s.on_ground:
            return []
        rwy = on_runway(b.dep, s.lat, s.lon)
        if rwy is None or rwy.name in b.cleared_runways:
            return []  # off runways, or on a runway we were cleared to cross
        if not self._ready("incursion", 30):
            return []
        pos = Position.TOWER if b.phase == Phase.HOLDING else Position.GROUND
        word = b.runway if b.runway in (rwy.ends[0].name, rwy.ends[1].name) else b._hold_word(rwy.name)
        return [b._tx(pos, b.phr.stop_hold_position(word))]

    def _route_deviation(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase not in (Phase.TAXI_OUT, Phase.TAXI_IN) or not b.route or not b.route.ok:
            return []
        if not s.on_ground or s.groundspeed_kt < 3:
            return []
        d = distance_to_route(b.field, b.route, s.lat, s.lon)
        if d < ROUTE_JOIN_M:
            b.route_joined = True
            return []
        if b.route_joined and not b.missed_turn and d > ROUTE_LOST_M and self._ready("missed_turn", 60):
            b.missed_turn = True  # the next call from the pilot gets a new route
            return [b._tx(Position.GROUND, b.phr.missed_turn())]
        return []

    def _heading_deviation(self, s: SimState) -> list[Transmission]:
        """Under radar vectors: once on the assigned heading, more than 20° off -> remind."""
        b = self.b
        if b.assigned_hdg is None or not b.hdg_captured or not b.tracker.airborne:
            return []
        off = abs((s.heading_mag - b.assigned_hdg + 540) % 360 - 180)
        if off <= HDG_DEV_DEG or not self._ready("heading", 60):
            return []
        return [b._tx(b.expected_position(), b.phr.check_heading(b.assigned_hdg))]

    def _taxi_speed(self, s: SimState) -> list[Transmission]:
        """More than TAXI_SPEED_KT for a few seconds on taxiways (not on a runway)."""
        b = self.b
        fast = (b.phase in (Phase.TAXI_OUT, Phase.TAXI_IN, Phase.PUSH_START, Phase.CLEARANCE)
                and s.on_ground and s.groundspeed_kt > TAXI_SPEED_KT
                and on_runway(b.field, s.lat, s.lon) is None)
        self._fast_ticks = self._fast_ticks + 1 if fast else 0
        if self._fast_ticks < 3 or not self._ready("taxi_speed", 60):
            return []
        return [b._tx(Position.GROUND, b.phr.reduce_taxi_speed())]

    def _altitude_deviation(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase not in (Phase.DEPARTURE, Phase.ENROUTE, Phase.ARRIVAL, Phase.APPROACH):
            return []
        if not b.assigned_alt or not b.tracker.airborne:
            return []
        dev = s.altitude_ft - b.assigned_alt
        level_bust = b.alt_captured and abs(dev) > ALT_DEV_FT
        climb_through = dev > ALT_DEV_FT and s.vs_fpm > 300
        descend_through = dev < -ALT_DEV_FT and s.vs_fpm < -300 and b.phase in (Phase.ARRIVAL, Phase.APPROACH)
        busted = level_bust or climb_through or descend_through
        if not busted or not self._ready("altitude", 60):
            return []
        return [b._tx(b.expected_position(), b.phr.check_altitude(b.assigned_alt))]

    def _runway_vacated(self, s: SimState) -> list[Transmission]:
        b = self.b
        if b.phase != Phase.LANDING or b.tracker.airborne or not s.on_ground:
            return []
        if s.groundspeed_kt > 30 or on_runway(b.field, s.lat, s.lon) is not None:
            return []
        b._set_phase(Phase.TAXI_IN)
        b.pending = None
        b._new_route(None)  # no taxi route until Ground gives one
        gnd = b.freq_of(Position.GROUND)
        if gnd == b.freq_of(Position.TOWER):
            return b._taxi_in()
        tx = b._tx(Position.TOWER, b.phr.contact("ground", gnd), True)
        b._expect(tx, {"frequency": gnd}, set())
        return [tx]
