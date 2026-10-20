"""Waypoint and navaid names from X-Plane's earth_fix.dat / earth_nav.dat.

Only names: "request direct VUZPU" needs to know the point exists, not where.
Custom Data (Navigraph etc.) is read together with default data.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

NAV_TYPES = {"2", "3", "12", "13"}   # NDB, VOR, DME(VOR), DME


def _files(xp_root: Path, name: str) -> list[Path]:
    return [p for p in (xp_root / "Custom Data" / name, xp_root / "Resources" / "default data" / name)
            if p.exists()]


@lru_cache(maxsize=4)
def point_names(xp_root: Path) -> frozenset[str]:
    names: set[str] = set()
    for p in _files(xp_root, "earth_fix.dat"):
        for line in p.open(encoding="utf-8", errors="replace"):
            f = line.split()
            if len(f) >= 3 and f[0][:1] in "-0123456789" and "." in f[0]:
                names.add(f[2].upper())
    for p in _files(xp_root, "earth_nav.dat"):
        for line in p.open(encoding="utf-8", errors="replace"):
            f = line.split()
            if len(f) >= 8 and f[0] in NAV_TYPES:
                names.add(f[7].upper())
    return frozenset(names)


def point_known(xp_root, ident: str) -> bool | None:
    """True/False from the nav data; None when there is no nav data to ask."""
    if xp_root is None:
        return None
    names = point_names(Path(xp_root))
    return ident.upper() in names if names else None
