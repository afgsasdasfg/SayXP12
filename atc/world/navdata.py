"""Waypoints and navaids from X-Plane's earth_fix.dat / earth_nav.dat.

"request direct VUZPU" needs to know the point exists; a STAR needs to know where its last
fix is (CIFP names fixes, it has no coordinates). One name can exist in several places:
the one nearest to where we are asking about wins.
Custom Data (Navigraph etc.) is read together with default data.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

NAV_TYPES = {"2", "3", "12", "13"}   # NDB, VOR, DME(VOR), DME
NEAR_NM = 300.0                      # a same-named point further away is another one


def _files(xp_root: Path, name: str) -> list[Path]:
    return [p for p in (xp_root / "Custom Data" / name, xp_root / "Resources" / "default data" / name)
            if p.exists()]


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


@lru_cache(maxsize=4)
def point_positions(xp_root: Path) -> dict[str, tuple[tuple[float, float], ...]]:
    pts: dict[str, list[tuple[float, float]]] = {}
    for p in _files(xp_root, "earth_fix.dat"):
        for line in p.open(encoding="utf-8", errors="replace"):
            f = line.split()   # lat lon ident region area type
            if len(f) >= 3 and f[0][:1] in "-0123456789" and "." in f[0]:
                lat, lon = _float(f[0]), _float(f[1])
                if lat is not None and lon is not None:
                    pts.setdefault(f[2].upper(), []).append((lat, lon))
    for p in _files(xp_root, "earth_nav.dat"):
        for line in p.open(encoding="utf-8", errors="replace"):
            f = line.split()   # type lat lon elev freq range magvar ident ...
            if len(f) >= 8 and f[0] in NAV_TYPES:
                lat, lon = _float(f[1]), _float(f[2])
                if lat is not None and lon is not None:
                    pts.setdefault(f[7].upper(), []).append((lat, lon))
    return {k: tuple(v) for k, v in pts.items()}


def point_names(xp_root: Path) -> frozenset[str]:
    return frozenset(point_positions(xp_root))


def point_known(xp_root, ident: str) -> bool | None:
    """True/False from the nav data; None when there is no nav data to ask."""
    if xp_root is None:
        return None
    names = point_names(Path(xp_root))
    return ident.upper() in names if names else None


def point_near(xp_root, ident: str, lat: float, lon: float) -> tuple[float, float] | None:
    """Position of the point of that name nearest to (lat, lon), within NEAR_NM."""
    if xp_root is None:
        return None
    from .aptdat import distance_m
    cand = point_positions(Path(xp_root)).get(ident.upper(), ())
    best = min(cand, key=lambda p: distance_m(lat, lon, *p), default=None)
    if best is None or distance_m(lat, lon, *best) > NEAR_NM * 1852:
        return None
    return best
