"""Flight plan from SimBrief: the latest OFP of a pilot, so the clearance can be "flight planned route".

SimBrief's public fetcher needs no key, only the Pilot ID (or username) from the SimBrief account:
    https://www.simbrief.com/api/xml.fetcher.php?userid=<id>&json=1

    python -m atc.world.simbrief <pilot_id>
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field

log = logging.getLogger("simbrief")

FETCH_URL = "https://www.simbrief.com/api/xml.fetcher.php"
_SID = re.compile(r"^[A-Z]{2,5}\d[A-Z]?$")


@dataclass
class RoutePoint:
    ident: str
    lat: float
    lon: float


@dataclass
class FlightPlan:
    origin: str
    destination: str
    callsign: str | None = None
    acf_type: str | None = None
    cruise_ft: int | None = None
    route: str = ""
    sid: str | None = None          # as planned; the brain still picks one for the runway in use
    sid_exit: str | None = None     # last fix of the SID = first enroute fix, chooses the SID from CIFP
    origin_rwy: str | None = None
    # enroute points with coordinates, from the SID exit to the STAR entry: what ATC sends the
    # pilot "direct" to when radar vectors end
    waypoints: list[RoutePoint] = field(default_factory=list)
    star: str | None = None         # as planned; the brain picks one for the arrival runway

    def summary(self) -> str:
        fl = f"FL{self.cruise_ft // 100}" if self.cruise_ft else "FL?"
        return f"{self.callsign or '?'} {self.acf_type or ''} {self.origin}-{self.destination} {fl} {self.route}".strip()


def _int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _list(v) -> list:
    return v if isinstance(v, list) else [v] if isinstance(v, dict) else []


def parse_ofp(d: dict) -> FlightPlan:
    """SimBrief OFP JSON -> FlightPlan. Raises ValueError on a fetch error ('Error: Unknown UserID')."""
    status = str((d.get("fetch") or {}).get("status", ""))
    if status and status.lower() != "success":
        raise ValueError(f"SimBrief: {status}")
    gen, atc = d.get("general") or {}, d.get("atc") or {}
    orig, dest = d.get("origin") or {}, d.get("destination") or {}
    if not orig.get("icao_code") or not dest.get("icao_code"):
        raise ValueError("SimBrief: no flight plan in the OFP")
    cs = atc.get("callsign") or (f"{gen.get('icao_airline', '')}{gen.get('flight_number', '')}" or None)
    fp = FlightPlan(
        origin=orig["icao_code"].upper(), destination=dest["icao_code"].upper(),
        callsign=cs.upper().replace(" ", "") if cs else None,
        acf_type=((d.get("aircraft") or {}).get("icaocode") or None),
        cruise_ft=_int(gen.get("initial_altitude")),
        route=str(gen.get("route") or "").strip(),
        origin_rwy=orig.get("plan_rwy") or None,
    )
    # SID and its exit fix from the navlog; the route string as fallback ("LUGA1A LUGA R805 ...")
    navlog = [f for f in _list((d.get("navlog") or {}).get("fix")) if f.get("type") != "ltlg"]
    proc = [str(f.get("is_sid_star")) == "1" and bool(_SID.match(str(f.get("via_airway", ""))))
            for f in navlog]
    first_enroute = next((i for i, p in enumerate(proc) if not p), len(navlog))
    sid_fixes = [f for f, p in zip(navlog[:first_enroute], proc) if p]
    sid_fixes = [f for f in sid_fixes if f["via_airway"] == sid_fixes[0]["via_airway"]] if sid_fixes else []
    star_fixes = [f for f, p in zip(navlog[first_enroute:], proc[first_enroute:]) if p]
    if star_fixes:
        fp.star = star_fixes[0]["via_airway"]
    # SID exit, enroute points, STAR entry (the STAR itself is ATC's to give)
    pts = sid_fixes[-1:] + [f for f, p in zip(navlog[first_enroute:], proc[first_enroute:])
                            if not p and f.get("type") != "apt"] + star_fixes[:1]
    for f in pts:
        lat, lon = _float(f.get("pos_lat")), _float(f.get("pos_long"))
        if f.get("ident") and lat is not None and lon is not None:
            fp.waypoints.append(RoutePoint(str(f["ident"]).upper(), lat, lon))
    if sid_fixes:
        fp.sid, fp.sid_exit = sid_fixes[0]["via_airway"], sid_fixes[-1].get("ident")
    else:
        words = fp.route.split()
        if len(words) >= 2 and _SID.match(words[0]):
            fp.sid, fp.sid_exit = words[0], words[1]
        elif words and words[0] != "DCT":
            fp.sid_exit = words[0]
    return fp


def apply_plan(fc, fp: FlightPlan) -> None:
    """Fill a FlightConfig from the plan. The plan replaces the config fallbacks; the pilot's
    first call still wins for the callsign and the clearance request for the destination."""
    if fp.callsign and fp.callsign != fc.callsign:
        fc.callsign, fc.telephony = fp.callsign, None
    fc.destination = fc.filed_destination = fp.destination
    fc.cruise_ft = fp.cruise_ft or fc.cruise_ft
    fc.acf_type = fp.acf_type or fc.acf_type
    if fp.sid_exit and not fc.sid:
        fc.first_fix = fp.sid_exit
    fc.route_points = list(fp.waypoints)
    fc.star = fp.star


def fetch(pilot_id: str = "", username: str = "", timeout_s: float = 10) -> FlightPlan:
    import httpx
    params = {"json": "1"}
    if pilot_id:
        params["userid"] = pilot_id
    elif username:
        params["username"] = username
    else:
        raise ValueError("SimBrief: set pilot_id or username in config.toml [simbrief]")
    r = httpx.get(FETCH_URL, params=params, timeout=timeout_s)
    try:
        d = r.json()
    except ValueError:
        raise ValueError(f"SimBrief: HTTP {r.status_code}, not an OFP") from None
    return parse_ofp(d)


if __name__ == "__main__":
    print(fetch(*sys.argv[1:2]).summary())
