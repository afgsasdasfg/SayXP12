"""apt.dat parser (X-Plane 12, format 1100/1200).

Row codes used (see developer.x-plane.com apt.dat spec):
  1/16/17  airport / seaplane base / heliport header
  100      land runway
  1201     taxi network node     1201 lat lon usage id [name]
  1202     taxi network edge     1202 a b oneway|twoway runway|taxiway[_A-F] [name]
  1204     active zone on the preceding 1202 edge: departure|arrival|ils rwy[,rwy]
  1300     start location (gate/tie-down)   1301 its metadata
  1302     airport metadata (datum_lat, datum_lon, ...)
  1050-1056 frequencies in kHz (ATIS, unicom/CTAF, delivery, ground, tower, approach, departure)
  50-56    legacy frequencies in 10 kHz units

Lookup order: Custom Scenery packs in scenery_packs.ini order, then Global Airports.
An index (ICAO -> file, byte offset, lat/lon) is cached in cache/apt_index.json.

    python -m atc.world.aptdat KBTV
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("aptdat")

FREQ_KINDS = {0: "atis", 1: "unicom", 2: "delivery", 3: "ground", 4: "tower", 5: "approach", 6: "departure"}
HEADER_CODES = (b"1 ", b"16 ", b"17 ")
EARTH_R = 6371008.8


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


@dataclass
class RunwayEnd:
    name: str
    lat: float
    lon: float
    displaced_m: float = 0.0
    heading_true: float = 0.0   # filled from geometry
    length_m: float = 0.0


@dataclass
class Runway:
    width_m: float
    surface: int
    ends: tuple[RunwayEnd, RunwayEnd]

    @property
    def name(self) -> str:
        return f"{self.ends[0].name}/{self.ends[1].name}"

    @property
    def length_m(self) -> float:
        a, b = self.ends
        return distance_m(a.lat, a.lon, b.lat, b.lon)


@dataclass
class TaxiNode:
    id: int
    lat: float
    lon: float
    usage: str = "both"
    name: str = ""


@dataclass
class TaxiEdge:
    a: int
    b: int
    oneway: bool
    kind: str            # "runway" or "taxiway"
    width_class: str     # A..F or ""
    name: str
    active_zones: list[tuple[str, list[str]]] = field(default_factory=list)

    def zone_runways(self, kinds=("departure", "arrival")) -> set[str]:
        out: set[str] = set()
        for k, rwys in self.active_zones:
            if k in kinds:
                out.update(rwys)
        return out


@dataclass
class Gate:
    name: str
    lat: float
    lon: float
    heading: float
    kind: str
    types: str


@dataclass
class Frequency:
    khz: int
    name: str


@dataclass
class Airport:
    icao: str
    name: str
    elevation_ft: float
    runways: list[Runway] = field(default_factory=list)
    freqs: dict[str, list[Frequency]] = field(default_factory=dict)
    gates: list[Gate] = field(default_factory=list)
    nodes: dict[int, TaxiNode] = field(default_factory=dict)
    edges: list[TaxiEdge] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def lat(self) -> float:
        if "datum_lat" in self.meta:
            return float(self.meta["datum_lat"])
        e = self.runways[0].ends[0] if self.runways else None
        return e.lat if e else 0.0

    @property
    def lon(self) -> float:
        if "datum_lon" in self.meta:
            return float(self.meta["datum_lon"])
        e = self.runways[0].ends[0] if self.runways else None
        return e.lon if e else 0.0

    def runway_end(self, name: str) -> RunwayEnd | None:
        for r in self.runways:
            for e in r.ends:
                if e.name == name:
                    return e
        return None

    def runway_of(self, end_name: str) -> Runway | None:
        for r in self.runways:
            if end_name in (r.ends[0].name, r.ends[1].name):
                return r
        return None

    def freq(self, kind: str) -> Frequency | None:
        lst = self.freqs.get(kind)
        return lst[0] if lst else None

    def nearest_gate(self, lat: float, lon: float) -> tuple[Gate | None, float]:
        best, bd = None, float("inf")
        for g in self.gates:
            d = distance_m(lat, lon, g.lat, g.lon)
            if d < bd:
                best, bd = g, d
        return best, bd


def parse_airport_lines(lines) -> Airport:
    """Parse the rows of a single airport (header first)."""
    ap: Airport | None = None
    last_edge: TaxiEdge | None = None
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        parts = raw.split()
        if not parts:
            continue
        code = parts[0]
        if code in ("1", "16", "17"):
            if ap is not None:
                break
            ap = Airport(icao=parts[4], name=" ".join(parts[5:]), elevation_ft=float(parts[1]))
            continue
        if ap is None:
            continue
        if code == "99":
            break
        if code == "100":
            ends = []
            for off in (8, 17):
                ends.append(RunwayEnd(name=parts[off], lat=float(parts[off + 1]),
                                      lon=float(parts[off + 2]), displaced_m=float(parts[off + 3])))
            a, b = ends
            a.heading_true = bearing_deg(a.lat, a.lon, b.lat, b.lon)
            b.heading_true = bearing_deg(b.lat, b.lon, a.lat, a.lon)
            a.length_m = b.length_m = distance_m(a.lat, a.lon, b.lat, b.lon)
            ap.runways.append(Runway(width_m=float(parts[1]), surface=int(parts[2]), ends=(a, b)))
        elif code == "1201":
            nid = int(parts[4])
            ap.nodes[nid] = TaxiNode(nid, float(parts[1]), float(parts[2]), parts[3], " ".join(parts[5:]))
        elif code == "1202":
            kind_raw = parts[4] if len(parts) > 4 else "taxiway"
            kind, _, wc = kind_raw.partition("_")
            last_edge = TaxiEdge(int(parts[1]), int(parts[2]), parts[3] == "oneway",
                                 "runway" if kind == "runway" else "taxiway", wc,
                                 " ".join(parts[5:]))
            ap.edges.append(last_edge)
        elif code == "1204" and last_edge is not None and len(parts) >= 3:
            last_edge.active_zones.append((parts[1], parts[2].split(",")))
        elif code == "1206":
            last_edge = None  # ground-truck edge, ignore its zones
        elif code == "1300":
            ap.gates.append(Gate(name=" ".join(parts[6:]), lat=float(parts[1]), lon=float(parts[2]),
                                 heading=float(parts[3]), kind=parts[4], types=parts[5]))
        elif code == "1302" and len(parts) >= 3:
            ap.meta[parts[1]] = " ".join(parts[2:])
        elif code.isdigit() and (1050 <= int(code) <= 1056 or 50 <= int(code) <= 56):
            c = int(code)
            khz = int(parts[1]) if c >= 1050 else int(parts[1]) * 10
            kind = FREQ_KINDS[c - (1050 if c >= 1050 else 50)]
            ap.freqs.setdefault(kind, []).append(Frequency(khz, " ".join(parts[2:])))
    if ap is None:
        raise ValueError("no airport header found")
    return ap


# ---------------------------------------------------------------- locating files

def scenery_apt_files(xp_root: Path) -> list[Path]:
    """apt.dat files in priority order: custom scenery (scenery_packs.ini order) then global."""
    files: list[Path] = []
    cs = xp_root / "Custom Scenery"
    ini = cs / "scenery_packs.ini"
    packs: list[Path] = []
    if ini.exists():
        for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("SCENERY_PACK ") and "*GLOBAL_AIRPORTS*" not in line:
                p = Path(line[len("SCENERY_PACK "):].strip())
                packs.append(p if p.is_absolute() else xp_root / p)
    elif cs.exists():
        packs = sorted(p for p in cs.iterdir() if p.is_dir())
    for p in packs:
        f = p / "Earth nav data" / "apt.dat"
        if f.exists():
            files.append(f)
    g = xp_root / "Global Scenery" / "Global Airports" / "Earth nav data" / "apt.dat"
    if g.exists():
        files.append(g)
    return files


def _index_file(path: Path) -> dict[str, list]:
    """ICAO -> [offset, lat, lon] for one apt.dat."""
    out: dict[str, list] = {}
    cur: str | None = None
    off = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(HEADER_CODES):
                parts = line.split()
                if len(parts) >= 5:
                    cur = parts[4].decode("ascii", "replace")
                    if cur not in out:
                        out[cur] = [off, None, None]
                    else:
                        cur = None  # duplicate in same file: keep first
            elif cur is not None:
                rec = out[cur]
                if line.startswith(b"1302 datum_lat "):
                    rec[1] = float(line.split()[2])
                elif line.startswith(b"1302 datum_lon "):
                    rec[2] = float(line.split()[2])
                elif rec[1] is None and line.startswith((b"100 ", b"101 ", b"102 ")):
                    p = line.split()
                    i = {b"100": 9, b"101": 4, b"102": 2}[p[0]]
                    rec[1], rec[2] = float(p[i]), float(p[i + 1])
            off += len(line)
    return out


class AptIndex:
    def __init__(self, xp_root: Path | str, cache_dir: Path | str = "cache"):
        self.xp_root = Path(xp_root)
        self.cache_file = Path(cache_dir) / "apt_index.json"
        self.files = scenery_apt_files(self.xp_root)
        self.entries: dict[str, tuple[str, int, float | None, float | None]] = {}
        self._load()

    def _signature(self) -> list:
        return [[str(f), f.stat().st_size, int(f.stat().st_mtime)] for f in self.files]

    def _load(self) -> None:
        sig = self._signature()
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                if data.get("signature") == sig:
                    self.entries = {k: tuple(v) for k, v in data["entries"].items()}
                    return
            except (ValueError, KeyError):
                pass
        log.info("building apt.dat index over %d files (one-time, may take a minute)...", len(self.files))
        entries: dict[str, tuple] = {}
        for f in self.files:  # priority order: first file wins
            for icao, (off, lat, lon) in _index_file(f).items():
                entries.setdefault(icao, (str(f), off, lat, lon))
        self.entries = entries
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps({"signature": sig, "entries": entries}))

    def load(self, icao: str) -> Airport:
        icao = icao.upper()
        if icao not in self.entries:
            raise KeyError(f"airport {icao} not found in apt.dat files")
        path, off, _, _ = self.entries[icao]

        def rows():
            with open(path, "rb") as fh:
                fh.seek(off)
                yield from fh

        ap = parse_airport_lines(rows())
        ap.source = path
        return ap

    def nearest(self, lat: float, lon: float, max_km: float = 20.0, need_runway=True) -> str | None:
        best, bd = None, max_km * 1000
        for icao, (_, _, alat, alon) in self.entries.items():
            if alat is None or abs(alat - lat) > 0.5 or abs(alon - lon) > 0.7:
                continue
            d = distance_m(lat, lon, alat, alon)
            if d < bd:
                best, bd = icao, d
        return best


def default_xp_root() -> Path | None:
    import os
    for f in ("x-plane_install_12.txt",):
        p = Path(os.environ.get("LOCALAPPDATA", "")) / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip() and Path(line.strip()).exists():
                    return Path(line.strip())
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = default_xp_root()
    idx = AptIndex(root)
    a = idx.load(sys.argv[1] if len(sys.argv) > 1 else "KBTV")
    print(f"{a.icao} {a.name} elev {a.elevation_ft:.0f} ft  ({a.source})")
    for r in a.runways:
        print(f"  RWY {r.name}  {r.length_m:.0f} m x {r.width_m:.0f} m  hdg {r.ends[0].heading_true:.0f}T")
    for k, lst in a.freqs.items():
        for f in lst:
            print(f"  {k:9s} {f.khz / 1000:.3f}  {f.name}")
    print(f"  gates: {len(a.gates)}  taxi nodes: {len(a.nodes)}  edges: {len(a.edges)}")
