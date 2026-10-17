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
