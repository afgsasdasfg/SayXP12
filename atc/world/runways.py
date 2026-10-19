"""Active runway selection by wind: minimum tailwind, then max headwind, then longest."""
from __future__ import annotations

import math

from .aptdat import Airport, RunwayEnd

CALM_KT = 3.0


def is_hard(surface: int) -> bool:
    """apt.dat surface codes: 1 asphalt, 2 concrete, 20-38 XP12 asphalt, 50-57 XP12 concrete."""
    return surface in (1, 2) or 20 <= surface <= 38 or 50 <= surface <= 57


def wind_components(end: RunwayEnd, wind_from_true: float, wind_kt: float) -> tuple[float, float]:
    """(headwind, crosswind) in kt; negative headwind = tailwind."""
    a = math.radians(wind_from_true - end.heading_true)
    return wind_kt * math.cos(a), wind_kt * math.sin(a)


# Rough minimum runway (length m, width m) by ICAO aerodrome code letter. Not a
# performance calculation: just enough to keep an A320 off a 1250 m GA runway.
CLASS_LIMITS = {"A": (600, 15), "B": (1000, 18), "C": (1800, 30), "D": (2400, 45),
                "E": (2600, 45), "F": (3000, 45)}
TYPE_LENGTH_M = {"A321": 2100, "A21N": 2000, "B738": 2000, "B739": 2100, "B38M": 1900,
                 "B39M": 2000, "CRJ2": 1700, "CRJ9": 1800, "E170": 1600, "E175": 1700,
                 "E190": 1800, "E195": 1900, "AT76": 1300, "AT75": 1300, "DH8D": 1400}


def runway_requirements(acf_type: str | None) -> tuple[float, float]:
    """(minimum length m, minimum width m) for an ICAO aircraft type."""
    if not acf_type:
        return 0.0, 0.0
    from .taxi import aircraft_class
    length, width = CLASS_LIMITS[aircraft_class(acf_type)]
    return float(TYPE_LENGTH_M.get(acf_type.upper(), length)), float(width)


def select_runway(ap: Airport, wind_from_true: float, wind_kt: float,
                  hard_only: bool = True, min_length_m: float = 0.0,
                  acf_type: str | None = None) -> RunwayEnd | None:
    """Wind-best end among runways long and wide enough for the aircraft. If none
    qualifies, only the longest runway is considered (never a shorter one)."""
    need_len, need_width = runway_requirements(acf_type)
    need_len = max(need_len, min_length_m)
    pool = [r for r in ap.runways if not hard_only or is_hard(r.surface)] or list(ap.runways)
    if not pool:
        return None
    fit = [r for r in pool if r.length_m >= need_len and r.width_m >= need_width]
    if not fit:
        longest = max(r.length_m for r in pool)
        fit = [r for r in pool if r.length_m >= longest - 1]
    ends: list[RunwayEnd] = [e for r in fit for e in r.ends]
    if wind_kt < CALM_KT:
        return max(ends, key=lambda e: e.length_m)

    def key(e: RunwayEnd):
        head, _ = wind_components(e, wind_from_true, wind_kt)
        tail = max(0.0, -head)
        return (round(tail), -round(head), -e.length_m)

    return min(ends, key=key)
