"""Physical flight-state tracking: airborne/landed edges, runway occupancy, stop/park.

The ATC phase (Phase) lives in atc.brain.fsm and advances on clearances; this module
only reports what the aircraft is physically doing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .state import SimState
from .world.aptdat import Airport, Runway


def _local_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    k = math.cos(math.radians(lat0)) * 111320.0
    return (lon - lon0) * k, (lat - lat0) * 110540.0


def runway_offset(rwy: Runway, lat: float, lon: float) -> tuple[float, float]:
    """(along-track m from end[0], cross-track m) relative to the runway centreline."""
    a, b = rwy.ends
    bx, by = _local_xy(a.lat, a.lon, b.lat, b.lon)
    px, py = _local_xy(a.lat, a.lon, lat, lon)
    L = math.hypot(bx, by) or 1.0
    ux, uy = bx / L, by / L
    along = px * ux + py * uy
    cross = -px * uy + py * ux
    return along, cross


def on_runway(ap: Airport, lat: float, lon: float, margin_m: float = 0.0) -> Runway | None:
    for r in ap.runways:
        along, cross = runway_offset(r, lat, lon)
        if -30 <= along <= r.length_m + 30 and abs(cross) <= r.width_m / 2 + margin_m:
            return r
    return None


@dataclass
class PhysicalEvents:
    took_off: bool = False
    landed: bool = False
    stopped: bool = False
    started_moving: bool = False
    shutdown_parked: bool = False


@dataclass
class PhaseTracker:
    airborne: bool = False
    moving: bool = False
    _air_count: int = 0
    _gnd_count: int = 0
    history: list[str] = field(default_factory=list)

    def update(self, s: SimState) -> PhysicalEvents:
        ev = PhysicalEvents()
        # hysteresis: a few consecutive samples before flipping air/ground
        if not s.on_ground and s.agl_ft > 20:
            self._air_count += 1
            self._gnd_count = 0
        elif s.on_ground:
            self._gnd_count += 1
            self._air_count = 0
        if not self.airborne and self._air_count >= 3:
            self.airborne = True
            ev.took_off = True
            self.history.append("took_off")
        elif self.airborne and self._gnd_count >= 3:
            self.airborne = False
            ev.landed = True
            self.history.append("landed")
        if s.on_ground:
            if not self.moving and s.groundspeed_kt > 3:
                self.moving = True
                ev.started_moving = True
            elif self.moving and s.groundspeed_kt < 0.5:
                self.moving = False
                ev.stopped = True
            if not self.moving and s.parking_brake > 0.5 and not s.any_engine_running:
                ev.shutdown_parked = True
        return ev
