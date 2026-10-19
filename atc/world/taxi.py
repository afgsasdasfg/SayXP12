"""Taxi routing over the apt.dat taxi network (1201/1202/1204).

Graph -> A* -> list of edges -> folded into an ATC instruction:
    "taxi via B, G, hold short runway 15"

    python -m atc.world.taxi KBTV 15 [lat lon]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from .aptdat import Airport, TaxiEdge, distance_m

log = logging.getLogger("taxi")

# Penalties (multipliers on edge length)
P_RUNWAY_OTHER = 5.0      # taxiing along / across another runway
P_RUNWAY_ACTIVE = 20.0    # backtracking on the departure runway: last resort
P_UNNAMED = 1.3           # apron taxilanes without a name
P_TOO_NARROW = 50.0
SHORT_CONNECTOR_M = 30.0

WIDTH_CLASSES = "ABCDEF"
# ICAO type -> ICAO aerodrome reference code letter (wingspan class)
TYPE_CLASS = {
    "C172": "A", "C152": "A", "SR22": "A", "PA28": "A", "C182": "A", "BE58": "A", "TBM9": "A",
    "PC12": "B", "C208": "B", "BE20": "B", "BE9L": "B", "CRJ2": "B", "E135": "B",
    "AT76": "C", "AT75": "C", "DH8D": "C", "CRJ9": "C", "E170": "C", "E175": "C", "E190": "C", "E195": "C",
    "A319": "C", "A320": "C", "A20N": "C", "A321": "C", "A21N": "C", "B737": "C", "B738": "C",
    "B739": "C", "B38M": "C", "B39M": "C", "MD82": "C", "MD88": "C",
    "B752": "D", "B753": "D", "B762": "D", "B763": "D", "MD11": "D",
    "A332": "E", "A333": "E", "A339": "E", "A359": "E", "A35K": "E", "B772": "E", "B77W": "E",
    "B77L": "E", "B788": "E", "B789": "E", "B78X": "E", "B744": "E",
    "A388": "F", "B748": "F",
}


def aircraft_class(icao_type: str) -> str:
    return TYPE_CLASS.get(icao_type.upper(), "C")


@dataclass
class TaxiLeg:
    """One clearance limit. Leg 0 is the taxi clearance; each later leg starts with a
    runway crossing that ATC clears when the aircraft reaches the previous hold point."""
    taxiways: list[str]
    hold_short: str | None     # runway (end "19" or name "15/33") to hold short of; None = destination
    cross: str | None          # runway crossed at the start of this leg
    end: int                   # index into TaxiRoute.nodes of this leg's hold point


@dataclass
class TaxiRoute:
    ok: bool
    taxiways: list[str] = field(default_factory=list)       # folded names, in order
    hold_short: str | None = None                            # runway end to hold short of
    crossings: list[str] = field(default_factory=list)      # runways crossed (by runway name)
    steps: list[tuple[str, str]] = field(default_factory=list)  # ("via","B") / ("cross","01/19")
    nodes: list[int] = field(default_factory=list)
    edges: list[TaxiEdge] = field(default_factory=list)
    length_m: float = 0.0
    destination: str = ""
    legs: list[TaxiLeg] = field(default_factory=list)
    intersection: str | None = None   # taxiway of an intersection departure


def _runway_matches(edge_name: str, end: str) -> bool:
    return end in edge_name.replace(" ", "").split("/")


def build_graph(ap: Airport, acf_class: str = "C", active_runway: str | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    ci = WIDTH_CLASSES.index(acf_class) if acf_class in WIDTH_CLASSES else 2
    for e in ap.edges:
        if e.a not in ap.nodes or e.b not in ap.nodes:
            continue
        na, nb = ap.nodes[e.a], ap.nodes[e.b]
        length = distance_m(na.lat, na.lon, nb.lat, nb.lon)
        cost = length
        if e.kind == "runway":
            cost *= P_RUNWAY_ACTIVE if active_runway and _runway_matches(e.name, active_runway) else P_RUNWAY_OTHER
        elif not e.name:
            cost *= P_UNNAMED
        if e.width_class and e.width_class in WIDTH_CLASSES and WIDTH_CLASSES.index(e.width_class) < ci:
            cost *= P_TOO_NARROW
        attrs = dict(length=length, cost=cost, edge=e)
        g.add_edge(e.a, e.b, **attrs)
        if not e.oneway:
            g.add_edge(e.b, e.a, **attrs)
    for nid in g.nodes:
        n = ap.nodes[nid]
        g.nodes[nid]["lat"], g.nodes[nid]["lon"] = n.lat, n.lon
    return g


def nearest_node(g: nx.DiGraph, lat: float, lon: float, taxiway_only: bool = True) -> tuple[int | None, float]:
    best, bd = None, float("inf")
    for nid, d in g.nodes(data=True):
        if taxiway_only and all(g.edges[nid, m]["edge"].kind == "runway" for m in g.successors(nid)):
            continue
        dist = distance_m(lat, lon, d["lat"], d["lon"])
        if dist < bd:
            best, bd = nid, dist
    return best, bd


def _heuristic(g):
    def h(u, v):
        if v == "SINK" or u == "SINK":
            return 0.0
        return distance_m(g.nodes[u]["lat"], g.nodes[u]["lon"], g.nodes[v]["lat"], g.nodes[v]["lon"])
    return h


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def _cut_runway(ap: Airport, e: TaxiEdge) -> str | None:
    na, nb = ap.nodes.get(e.a), ap.nodes.get(e.b)
    if na is None or nb is None:
        return None
    for r in ap.runways:
        a, b = r.ends
        if _segments_intersect((na.lon, na.lat), (nb.lon, nb.lat), (a.lon, a.lat), (b.lon, b.lat)):
            return r.name
    return None


def _protected(e: TaxiEdge) -> bool:
    """Runway pavement or a runway's departure/arrival protected area (needs a clearance)."""
    return e.kind == "runway" or bool(e.zone_runways())


def _fold_names(segs: list[list]) -> list[str]:
    """[(name, length)] -> taxiway names: merge repeats, drop unnamed lanes and short
    connectors (keeping the last named taxiway before the hold)."""
    named = [s for s in segs if s[0]]
    out: list[list] = []
    for i, s in enumerate(named):
        is_last = i == len(named) - 1
        if s[1] < SHORT_CONNECTOR_M and not is_last and out:
            continue
        if out and out[-1][0] == s[0]:
            out[-1][1] += s[1]
        else:
            out.append(list(s))
    return [s[0] for s in out]


def _legs(edges: list[TaxiEdge], lengths: list[float], ap: Airport | None,
          dep_runway: str | None) -> tuple[list[TaxiLeg], int]:
    """Split the path at runway protected areas.

    A run of protected edges that reaches the departure runway is the final hold short
    point (placed at the *start* of the run, so the aircraft never enters another
    runway's protected area unannounced). A run that crosses another runway becomes a
    clearance limit "hold short runway X"; the next leg starts with "cross runway X".
    Returns (legs, index of the final hold point in the edge list).
    """
    legs: list[TaxiLeg] = []
    segs: list[list] = []
    cross: str | None = None
    i = 0
    while i < len(edges):
        e = edges[i]
        if not _protected(e):
            segs.append([e.name.strip(), lengths[i]])
            i += 1
            continue
        k = i
        while k < len(edges) and _protected(edges[k]):
            k += 1
        run = edges[i:k]
        crossed: list[str] = []
        for r in run:
            if r.kind == "runway" and not (dep_runway and _runway_matches(r.name, dep_runway)):
                name = r.name.strip()
            elif r.kind != "runway" and ap is not None:
                # a zone alone also covers parallel taxiways: require cutting the centreline
                name = _cut_runway(ap, r)
                if name and dep_runway and _runway_matches(name, dep_runway):
                    name = None
            else:
                name = None
            if name and name not in crossed:
                crossed.append(name)
        reaches_dep = dep_runway is not None and (k == len(edges) or any(
            (r.kind == "runway" and _runway_matches(r.name, dep_runway)) or dep_runway in r.zone_runways()
            for r in run))
        if crossed:
            legs.append(TaxiLeg(_fold_names(segs), crossed[0], cross, i))
            cross, segs = crossed[0], []
        elif reaches_dep:
            legs.append(TaxiLeg(_fold_names(segs), dep_runway, cross, i))
            return legs, i
        for j in range(i, k):  # taxiway names inside the protected run still describe the way
            if edges[j].kind != "runway":
                segs.append([edges[j].name.strip(), lengths[j]])
        i = k
    legs.append(TaxiLeg(_fold_names(segs), dep_runway, cross, len(edges)))
    return legs, len(edges)


def _route_summary(legs: list[TaxiLeg]) -> tuple[list[tuple[str, str]], list[str]]:
    steps: list[tuple[str, str]] = []
    for leg in legs:
        if leg.cross:
            steps.append(("cross", leg.cross))
        for t in leg.taxiways:
            if not (steps and steps[-1] == ("via", t)) and not (
                    len(steps) >= 2 and steps[-1][0] == "cross" and steps[-2] == ("via", t)):
                steps.append(("via", t))
    return steps, [leg.cross for leg in legs if leg.cross]


def route_to_runway(ap: Airport, lat: float, lon: float, runway_end: str,
                    acf_class: str = "C") -> TaxiRoute:
    """Route from the aircraft position to the departure end of `runway_end`."""
    g = build_graph(ap, acf_class, runway_end)
    end = ap.runway_end(runway_end)
    if end is None or g.number_of_nodes() == 0:
        return TaxiRoute(ok=False, hold_short=runway_end)
    start, d0 = nearest_node(g, lat, lon)
    if start is None or d0 > 400:
        log.warning("no taxi node near aircraft (%.0f m)", d0)
        return TaxiRoute(ok=False, hold_short=runway_end)
    # entry candidates: nodes on runway edges of this runway, weighted by distance from threshold
    cands: dict[int, float] = {}
    for u, v, d in g.edges(data=True):
        e: TaxiEdge = d["edge"]
        if e.kind == "runway" and _runway_matches(e.name, runway_end):
            for n in (u, v):
                nd = g.nodes[n]
                cands[n] = distance_m(end.lat, end.lon, nd["lat"], nd["lon"])
    if not cands:
        # no runway edges in the network: use the nodes of edges carrying the runway's active zone
        for u, v, d in g.edges(data=True):
            if any(runway_end in rw for _, rw in d["edge"].active_zones):
                for n in (u, v):
                    nd = g.nodes[n]
                    cands[n] = distance_m(end.lat, end.lon, nd["lat"], nd["lon"])
    if not cands:
        return TaxiRoute(ok=False, hold_short=runway_end)
    # prefer full-length departures: penalise entries far from the threshold
    for n, dist in cands.items():
        g.add_edge(n, "SINK", cost=dist * 0.5 + (0 if dist < end.length_m * 0.25 else 5000), length=0, edge=None)
    try:
        path = nx.astar_path(g, start, "SINK", heuristic=_heuristic(g), weight="cost")
    except nx.NetworkXNoPath:
        return TaxiRoute(ok=False, hold_short=runway_end)
    path = path[:-1]
    entry = g.nodes[path[-1]]
    edges = [g.edges[a, b]["edge"] for a, b in zip(path, path[1:])]
    lengths = [g.edges[a, b]["length"] for a, b in zip(path, path[1:])]
    legs, hold_idx = _legs(edges, lengths, ap, runway_end)
    nodes = path[: hold_idx + 1]
    steps, crossings = _route_summary(legs)
    # intersection departure: the runway is entered well away from the threshold
    intersection = None
    if distance_m(end.lat, end.lon, entry["lat"], entry["lon"]) > max(300.0, 0.15 * end.length_m):
        intersection = legs[-1].taxiways[-1] if legs[-1].taxiways else None
    return TaxiRoute(ok=True, taxiways=[n for k, n in steps if k == "via"], hold_short=runway_end,
                     crossings=crossings, steps=steps, nodes=nodes, edges=edges[:hold_idx],
                     length_m=sum(lengths[:hold_idx]), destination=f"runway {runway_end}",
                     legs=legs, intersection=intersection)


def route_to_point(ap: Airport, lat: float, lon: float, dest_lat: float, dest_lon: float,
                   acf_class: str = "C", dest_name: str = "") -> TaxiRoute:
    """Route from the aircraft position to the node nearest a point (e.g. a gate)."""
    g = build_graph(ap, acf_class)
    start, d0 = nearest_node(g, lat, lon, taxiway_only=False)
    goal, d1 = nearest_node(g, dest_lat, dest_lon)
    if start is None or goal is None or d0 > 400:
        return TaxiRoute(ok=False, destination=dest_name)
    try:
        path = nx.astar_path(g, start, goal, heuristic=_heuristic(g), weight="cost")
    except nx.NetworkXNoPath:
        return TaxiRoute(ok=False, destination=dest_name)
    edges = [g.edges[a, b]["edge"] for a, b in zip(path, path[1:])]
    lengths = [g.edges[a, b]["length"] for a, b in zip(path, path[1:])]
    legs, _ = _legs(edges, lengths, ap, None)
    steps, crossings = _route_summary(legs)
    return TaxiRoute(ok=True, taxiways=[n for k, n in steps if k == "via"], crossings=crossings,
                     steps=steps, nodes=path, edges=edges, length_m=sum(lengths),
                     destination=dest_name, legs=legs)


def distance_to_route(ap: Airport, route: TaxiRoute, lat: float, lon: float) -> float:
    """Approximate cross-track distance (m) from the aircraft to the route polyline."""
    import math
    best = float("inf")
    pts = [(ap.nodes[n].lat, ap.nodes[n].lon) for n in route.nodes if n in ap.nodes]
    if len(pts) == 1:
        return distance_m(lat, lon, *pts[0])
    k = math.cos(math.radians(lat)) * 111320.0
    px, py = lon * k, lat * 110540.0
    for (alat, alon), (blat, blon) in zip(pts, pts[1:]):
        ax, ay, bx, by = alon * k, alat * 110540.0, blon * k, blat * 110540.0
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


if __name__ == "__main__":
    import sys

    from .aptdat import AptIndex, default_xp_root

    logging.basicConfig(level=logging.INFO)
    icao = sys.argv[1] if len(sys.argv) > 1 else "KBTV"
    rwy = sys.argv[2] if len(sys.argv) > 2 else None
    ap = AptIndex(default_xp_root()).load(icao)
    if len(sys.argv) > 4:
        lat, lon = float(sys.argv[3]), float(sys.argv[4])
        starts = [("given position", lat, lon)]
    else:
        starts = [(f"gate {g.name}", g.lat, g.lon) for g in ap.gates[:: max(1, len(ap.gates) // 5)]]
    rwys = [rwy] if rwy else [e.name for r in ap.runways for e in r.ends]
    for label, lat, lon in starts:
        for r in rwys:
            rt = route_to_runway(ap, lat, lon, r)
            if rt.ok:
                txt = ", ".join(f"cross runway {n}" if k == "cross" else n for k, n in rt.steps)
                print(f"{label:>18} -> {r}: via {txt}, hold short {r}  ({rt.length_m:.0f} m)")
            else:
                print(f"{label:>18} -> {r}: NO ROUTE")
