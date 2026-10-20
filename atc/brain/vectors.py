"""Radar vectors to the final approach course: pure geometry, no radio.

Frame: origin at the landing threshold, x = NM out along the extended centreline
(on the approach side), y = NM right of the inbound course as the pilot sees it.
The plan joins the final at the gate (2 NM outside the FAF, FAA 7110.65 5-9-1)
with at most a 30° intercept:

  straight-in  -> intercept heading at once
  wide / abeam -> base leg (perpendicular), then intercept
  field side   -> downwind (reciprocal) past the gate, base, intercept
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..world.aptdat import bearing_deg, distance_m

NM = 1852.0
GATE_OUTSIDE_FAF_NM = 2.0
DOWNWIND_OFFSET_NM = 5.0     # downwind leg this far abeam the centreline
DOWNWIND_BAND_NM = 3.0       # ... give or take: wider than offset + band, cut in towards it
BASE_PAST_GATE_NM = 3.0      # base turn this far outside the gate
INTERCEPT_TURN_NM = 2.0      # turn onto the intercept heading this far from the centreline
INTERCEPT_DEG = 30
ESTABLISHED_NM = 0.5
GLIDE_FT_PER_NM = 318.0      # 3° glideslope


@dataclass
class Final:
    """Where the final approach course is (true course, threshold, FAF)."""
    lat: float
    lon: float
    course_true: float        # inbound
    course_mag: float
    faf_dist_nm: float
    faf_alt: int

    @property
    def gate_nm(self) -> float:
        return self.faf_dist_nm + GATE_OUTSIDE_FAF_NM

    def frame(self, lat: float, lon: float) -> tuple[float, float]:
        d = distance_m(self.lat, self.lon, lat, lon) / NM
        rel = math.radians(bearing_deg(self.lat, self.lon, lat, lon) - (self.course_true + 180))
        x = d * math.cos(rel)
        y = -d * math.sin(rel)   # +rel is left of the outbound direction = right of the inbound course
        return x, y


@dataclass
class Vector:
    leg: str                  # downwind / base / intercept
    heading: int              # magnetic, rounded to 10°
    side: int                 # +1 right of the inbound course, -1 left


def _round10(h: float) -> int:
    return int(round((h % 360) / 10.0) * 10) % 360 or 360


def side_of(y: float) -> int:
    return 1 if y >= 0 else -1


def intercept_heading(f: Final, side: int) -> int:
    return _round10(f.course_mag - side * INTERCEPT_DEG)


def straight_in(f: Final, x: float, y: float) -> bool:
    """Far enough out that a 30° cut from here still joins outside the gate."""
    return x > f.gate_nm and x - abs(y) / math.tan(math.radians(INTERCEPT_DEG)) >= f.gate_nm


def first_vector(f: Final, x: float, y: float) -> Vector:
    side = side_of(y)
    if straight_in(f, x, y):
        return Vector("intercept", intercept_heading(f, side), side)
    if x >= f.gate_nm + BASE_PAST_GATE_NM:
        return Vector("base", _round10(f.course_mag - side * 90), side)
    return Vector("downwind", downwind_heading(f, side, y), side)


def downwind_heading(f: Final, side: int, y: float) -> int:
    """Reciprocal of the final; cut 30° away from it while too close abeam, towards it while too wide."""
    cut = 0
    if abs(y) < DOWNWIND_OFFSET_NM - 1.0:
        cut = -side * INTERCEPT_DEG
    elif abs(y) > DOWNWIND_OFFSET_NM + DOWNWIND_BAND_NM:
        cut = side * INTERCEPT_DEG
    return _round10(f.course_mag + 180 + cut)


def next_vector(f: Final, v: Vector, x: float, y: float) -> Vector | None:
    """The next turn once the aircraft has flown far enough on the current leg."""
    level_off = _round10(f.course_mag + 180)
    if v.leg == "downwind" and v.heading != level_off and downwind_heading(f, v.side, y) == level_off:
        return Vector("downwind", level_off, v.side)   # reached the downwind offset: parallel to the final
    if v.leg == "downwind" and x >= f.gate_nm + BASE_PAST_GATE_NM and abs(y) >= INTERCEPT_TURN_NM:
        return Vector("base", _round10(f.course_mag - v.side * 90), v.side)
    if v.leg == "base" and abs(y) <= INTERCEPT_TURN_NM:
        return Vector("intercept", intercept_heading(f, v.side), v.side)
    return None


def established(f: Final, x: float, y: float, heading_mag: float) -> bool:
    off = abs((heading_mag - f.course_mag + 540) % 360 - 180)
    return abs(y) <= ESTABLISHED_NM and off <= 20 and 0 < x <= f.gate_nm + 10


def faf_distance_from_altitude(faf_alt: int, elevation_ft: float) -> float:
    """No CIFP: the FAF where a 3° glideslope meets its altitude."""
    return max(3.0, round((faf_alt - elevation_ft) / GLIDE_FT_PER_NM, 1))
