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


def select_runway(ap: Airport, wind_from_true: float, wind_kt: float,
                  hard_only: bool = True, min_length_m: float = 0.0) -> RunwayEnd | None:
    ends: list[RunwayEnd] = []
    for r in ap.runways:
        if hard_only and not is_hard(r.surface):
            continue
        if r.length_m < min_length_m:
            continue
        ends.extend(r.ends)
    if not ends:
        ends = [e for r in ap.runways for e in r.ends]
    if not ends:
        return None
    if wind_kt < CALM_KT:
        return max(ends, key=lambda e: e.length_m)

    def key(e: RunwayEnd):
        head, _ = wind_components(e, wind_from_true, wind_kt)
        tail = max(0.0, -head)
        return (round(tail), -round(head), -e.length_m)

    return min(ends, key=key)
