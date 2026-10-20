"""Planned route and STAR: radar vectors end "direct <next route point>", the descent comes with
a STAR from the CIFP, and vectors to the ILS start at its last fix.

ULLI -> UUEE with the SimBrief OFP fixture (route and coordinates made up) and the real UUEE CIFP.
KEZVU, the last fix of every UUEE STAR, gets a made-up position in a small earth_fix.dat.
"""
import json

from atc.brain.fsm import Brain, FlightConfig
from atc.brain.monitor import Monitor
from atc.llm.intent_parser import IntentParser
from atc.state import Phase, Position
from atc.world.aptdat import bearing_deg
from atc.world.cifp import parse_cifp, star_entries, star_path, stars_for_runway
from atc.world.simbrief import apply_plan, parse_ofp
from conftest import FIX
from test_approach import CTR, UUEE_MAGVAR, fly_vectors, offset
from test_scenario import UUEE_APP, ULLI_DEP, ULLI_TWR, RouteIndex, air, say, ticks, wx

OFP = json.loads((FIX / "simbrief_ofp_ULLI_UUEE.json").read_text())
KEZVU = (56.15, 37.20)   # made up, ~13 NM north-west of Sheremetyevo
EARTH_FIX = f"""I
1200 Version - data cycle 2410, made up for the tests
  {KEZVU[0]:.9f}   {KEZVU[1]:.9f} KEZVU UUEE UU 2048
  60.000000000   10.000000000 KEZVU ENRT EN 2048
99
"""


def planned_brain(ulli, uuee, xp_root, nav=True, **fc_kw):
    if nav:
        (xp_root / "Resources" / "default data" / "earth_fix.dat").write_text(EARTH_FIX)
    fc = FlightConfig(callsign="DLH4AB", **fc_kw)
    apply_plan(fc, parse_ofp(OFP))
    return Brain(fc, ulli, index=RouteIndex(uuee), xp_root=xp_root)


def test_ofp_route_points_and_star():
    fp = parse_ofp(OFP)
    assert [p.ident for p in fp.waypoints] == ["SUMAN", "BITSA", "BUKUB"]   # SID exit .. STAR entry
    assert fp.star == "BUKU3E" and (fp.sid, fp.sid_exit) == ("SUMAN1A", "SUMAN")
    assert parse_ofp({**OFP, "navlog": {}}).waypoints == []


def test_uuee_stars_from_cifp():
    procs = parse_cifp(FIX / "CIFP" / "UUEE.dat")
    stars = {p.ident for p in stars_for_runway(procs, "24L")}
    assert {"BUKU3E", "BUKU3F", "DIMG3E", "ROMT3F"} <= stars   # no runway transitions: all runways
    p = procs["STAR"]["BUKU3E"]
    assert star_entries(p) == ["BUKUB"]
    path = star_path(p, "24L")
    assert path[0] == "BUKUB" and path[-1] == "KEZVU" and len(path) == 13


def test_next_route_point(ulli, uuee, xp_root):
    b = planned_brain(ulli, uuee, xp_root)
    b.sim = air(59.79, 30.18, ULLI_DEP, 3000)
    assert b.next_route_point().ident == "SUMAN"
    b.sim = air(59.45, 30.90, CTR, 12000)          # past SUMAN, on the way to BITSA
    assert b.next_route_point().ident == "BITSA"
    b.sim = air(57.81, 34.01, CTR, 35000)          # at BITSA
    assert b.next_route_point().ident == "BUKUB"
    b.sim = air(56.76, 36.39, CTR, 12000)          # at the STAR entry: end of the planned route
    assert b.next_route_point() is None


def test_planned_route_star_and_vectors_at_its_end(ulli, uuee, xp_root):
    b = planned_brain(ulli, uuee, xp_root)
    m = Monitor(b)
    p = IntentParser()
    cs = b.fc.callsign   # AFL51 from the plan
    b.runway, b.squawk, b.assigned_alt = "28L", "4521", 5000
    b._set_phase(Phase.TAKEOFF)
    out = ticks(m, air(59.80, 30.20, ULLI_TWR, 1800, vs_fpm=2000, **wx()))
    assert any("contact departure" in t.text for t in out)

    # --- Departure (no SID for the runway): heading towards SUMAN, not towards Moscow
    s = air(59.79, 30.18, ULLI_DEP, 2300, heading_true=280, heading_mag=269, **wx())
    out = say(b, p, f"Departure, {cs} passing 2300 climbing 5000", s)
    assert "radar contact" in out[0].text and b.assigned_hdg == 130   # SUMAN ~138 true, variation ~11 E
    say(b, p, f"heading 130, climb flight level 100, {cs}", s)

    # --- Center: end of vectors, direct the next point of the route
    ticks(m, air(59.45, 30.90, ULLI_DEP, 8600, **wx()), 1)
    s = air(59.45, 30.90, CTR, 9000, **wx())
    out = say(b, p, f"Center, {cs} passing 9000 climbing flight level 100", s)
    assert "climb FL350, proceed direct BITSA" in out[0].text and b.assigned_hdg is None
    assert say(b, p, f"climb flight level 350, direct BITSA, {cs}", s) == []

    # --- descent: the planned STAR, the approach to expect
    s = air(57.3, 35.0, CTR, 35000, **wx())
    out = say(b, p, f"Center, {cs} request descent", s)
    assert "descend FL100, cleared BUKU3E arrival, expect ILS approach runway 24L" in out[0].text
    assert "Buku three Echo arrival" in out[0].spoken
    assert b.star_fixes[0] == "BUKUB" and b.star_fixes[-1] == "KEZVU" and b.phase == Phase.ARRIVAL
    assert say(b, p, f"descend flight level 100, cleared BUKU 3 echo arrival, {cs}", s) == []

    # --- 40 NM out: Approach; on the STAR the check-in gets "continue", not vectors
    s = air(56.5, 36.9, CTR, 10000, **wx())
    out = ticks(m, s, 1)
    assert any("contact approach 126.6" in t.text for t in out)
    say(b, p, f"126.6, {cs}", s)
    s = air(56.5, 36.9, UUEE_APP, 10000, **wx())
    out = say(b, p, f"Approach, {cs} descending flight level 100", s)
    assert out[0].position == Position.APPROACH and b.phase == Phase.ARRIVAL
    assert "continue BUKU3E arrival, descend 4000 ft, expect ILS approach runway 24L" in out[0].text
    assert b.assigned_hdg is None
    assert say(b, p, f"continue BUKU 3 echo arrival, descend 4000 feet, {cs}", s) == []

    # --- fly the STAR to KEZVU: vectors start 5 NM before it, without a call from the pilot
    said = []
    for _ in range(60):
        brg = bearing_deg(s.lat, s.lon, *KEZVU)
        lat, lon = offset(s.lat, s.lon, brg, 1.0)
        s = air(lat, lon, UUEE_APP, 4000, heading_true=brg, heading_mag=(brg - UUEE_MAGVAR) % 360, **wx())
        out = m.tick(s)
        said += [t.text for t in out]
        if out:
            break
    assert b.phase == Phase.APPROACH and len(said) == 1, said
    assert "expect vectors ILS approach runway 24L" in said[0] and out[0].proactive
    assert b.final.faf_alt == 2000
    assert say(b, p, f"heading {b.assigned_hdg}, descend 3000 feet, {cs}", s) == []
    s, said, _ = fly_vectors(b, m, s, UUEE_MAGVAR)
    assert any("cleared ILS approach runway 24L" in t for t in said), said
    assert said[-1] == f"{cs}, contact tower 118.7"


def test_star_on_approach_check_in_without_descent_request(ulli, uuee, xp_root):
    b = planned_brain(ulli, uuee, xp_root)
    b._set_phase(Phase.ARRIVAL)   # Center sent us to Approach without a descent request
    b._load_dest()
    b.approach_handoff = True
    s = air(56.5, 36.9, UUEE_APP, 12000, **wx())
    out = say(b, IntentParser(), "Approach, AFL51 flight level 120", s)
    assert "cleared BUKU3E arrival, descend 4000 ft, expect ILS approach runway 24L" in out[0].text


def test_star_chosen_by_entry_when_the_planned_one_is_not_in_the_cifp(ulli, uuee, xp_root):
    b = planned_brain(ulli, uuee, xp_root)
    b.fc.star = "BUKU3Z"   # another AIRAC cycle: same entry, other letter
    b._set_phase(Phase.ENROUTE)
    out = say(b, IntentParser(), "Center, DLH4AB request descent", air(57.3, 35.0, CTR, 35000, **wx()))
    assert "cleared BUKU3E arrival" in out[0].text


def test_star_end_unknown_goes_to_vectors(ulli, uuee, xp_root):
    """No nav data for KEZVU: the STAR can't be followed, Approach vectors as before."""
    b = planned_brain(ulli, uuee, xp_root, nav=False)
    b._set_phase(Phase.ENROUTE)
    say(b, IntentParser(), "Center, DLH4AB request descent", air(57.3, 35.0, CTR, 35000, **wx()))
    b.pending = None
    b.approach_handoff = True
    out = say(b, IntentParser(), "Approach, DLH4AB flight level 100", air(56.5, 36.9, UUEE_APP, 10000, **wx()))
    assert "expect vectors ILS approach runway 24L" in out[0].text and b.phase == Phase.APPROACH


def test_without_a_plan_no_star(ulli, uuee, xp_root):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="UUEE"), ulli, index=RouteIndex(uuee), xp_root=xp_root)
    b._set_phase(Phase.ENROUTE)
    out = say(b, IntentParser(), "Center, DLH4AB request descent", air(57.3, 35.0, CTR, 35000, **wx()))
    assert out[0].text == "DLH4AB, descend FL100" and b.star is None


def test_top_of_descent_with_the_star(ulli, uuee, xp_root):
    """FL350, ~122 NM from UUEE but 65 NM from the STAR entry BUKUB: FL100 by BUKUB needs 75 NM."""
    b = planned_brain(ulli, uuee, xp_root)
    m = Monitor(b)
    b._set_phase(Phase.ENROUTE)
    b.assigned_alt = 35000
    assert ticks(m, air(57.6, 35.2, CTR, 34000, vs_fpm=1500, **wx())) == []   # still climbing
    assert ticks(m, air(57.6, 35.2, ULLI_DEP, 35000, **wx())) == []           # not on the Center frequency
    out = ticks(m, air(57.6, 35.2, CTR, 35000, **wx()))
    assert len(out) == 1 and out[0].proactive
    assert out[0].text == "AFL51, descend FL100, cleared BUKU3E arrival, expect ILS approach runway 24L"
    assert b.phase == Phase.ARRIVAL and b.assigned_alt == 10000
