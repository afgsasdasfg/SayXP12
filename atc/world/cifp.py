"""Minimal CIFP reader (ARINC-424-like CSV in Resources/default data/CIFP/<ICAO>.dat).

Row: <TYPE>:<seq>,<route type>,<proc ident>,<transition>,<fix>,...
Stage 1 only needs: which SIDs serve a runway, and their first fixes / top altitude.
Custom Data/CIFP overrides default data when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Procedure:
    kind: str                 # SID / STAR / APPCH
    ident: str
    transitions: dict[str, list[str]] = field(default_factory=dict)  # transition -> fixes


def cifp_file(xp_root: Path, icao: str) -> Path | None:
    for base in (xp_root / "Custom Data" / "CIFP", xp_root / "Resources" / "default data" / "CIFP"):
        p = base / f"{icao.upper()}.dat"
        if p.exists():
            return p
    return None


def parse_cifp(path: Path) -> dict[str, dict[str, Procedure]]:
    out: dict[str, dict[str, Procedure]] = {"SID": {}, "STAR": {}, "APPCH": {}}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        kind, rest = line.split(":", 1)
        if kind not in out:
            continue
        f = [x.strip() for x in rest.split(",")]
        if len(f) < 5:
            continue
        ident, trans, fix = f[2], f[3] or "ALL", f[4]
        proc = out[kind].setdefault(ident, Procedure(kind, ident))
        fixes = proc.transitions.setdefault(trans, [])
        if fix and (not fixes or fixes[-1] != fix):
            fixes.append(fix)
    return out


def sids_for_runway(procs: dict[str, dict[str, Procedure]], runway_end: str) -> list[Procedure]:
    key = f"RW{runway_end}"
    res = []
    for p in procs.get("SID", {}).values():
        if key in p.transitions or any(t.startswith("RW") and t.endswith("B") and t[2:4] == runway_end[:2]
                                       for t in p.transitions):
            res.append(p)
    return res


# ------------------------------------------------------------------ approaches
# APPCH row fields (after "APPCH:"), X-Plane CIFP / ARINC 424:
#  0 seq, 1 route type (A transition, I ILS, L LOC, R RNAV, ...), 2 ident (I15, R33-Y),
#  3 transition, 4 fix, 8 waypoint description (4th char: A IAF, B/I IF, F FAF, M MAP),
#  11 path/termination, 20 magnetic course (tenths), 21 leg distance (tenths NM),
#  22 altitude description, 23 altitude 1, 24 altitude 2
APPROACH_KINDS = {"I": "ILS", "L": "localizer", "R": "RNAV"}   # preference order


@dataclass
class FinalApproach:
    kind: str                 # ILS / localizer / RNAV
    ident: str                # CIFP ident, "I15"
    runway: str               # "15"
    variant: str = ""         # "Y" of R33-Y
    course_mag: float | None = None
    faf: str | None = None
    faf_alt: int | None = None
    faf_dist_nm: float | None = None   # FAF to threshold along the final legs


def _num(s: str, scale: float = 1.0) -> float | None:
    s = s.strip()
    if not s or not s.lstrip("-").isdigit():
        return None
    return int(s) / scale


def parse_finals(path: Path) -> list[FinalApproach]:
    """Final approach segments (FAF, its altitude, course, distance to the runway)."""
    import re
    rows: dict[str, list[list[str]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("APPCH:"):
            continue
        f = [x.strip() for x in line[6:].rstrip(";").split(",")]
        if len(f) < 25 or f[1] not in APPROACH_KINDS:
            continue
        rows.setdefault(f[2], []).append(f)
    out = []
    for ident, legs in rows.items():
        m = re.match(r"^[A-Z](\d{2}[LRC]?)(?:-([A-Z]))?$", ident)
        if not m:
            continue
        fa = FinalApproach(APPROACH_KINDS[legs[0][1]], ident, m.group(1), m.group(2) or "")
        legs.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 0)
        i_faf = next((i for i, r in enumerate(legs) if r[8][3:4] == "F"), None)
        if i_faf is not None:
            r = legs[i_faf]
            fa.faf = r[4]
            alt = _num(r[23])
            fa.faf_alt = int(alt) if alt else None
            dist = 0.0
            for nxt in legs[i_faf + 1:]:
                d = _num(nxt[21], 10)
                if d:
                    dist += d
                if nxt[4].startswith("RW") or nxt[8][3:4] == "M":
                    break
            fa.faf_dist_nm = round(dist, 1) or None
        for r in legs[i_faf or 0:]:
            c = _num(r[20], 10)
            if c is not None and r[11] in ("CF", "TF"):
                fa.course_mag = c
                break
        out.append(fa)
    return out


def final_for_runway(finals: list[FinalApproach], runway_end: str) -> FinalApproach | None:
    """ILS first, then localizer, then RNAV; among variants the alphabetically last (Z) first."""
    cand = [f for f in finals if f.runway == runway_end.upper()]
    order = list(APPROACH_KINDS.values())
    cand.sort(key=lambda f: (order.index(f.kind), -ord(f.variant or "Z")))
    return cand[0] if cand else None
