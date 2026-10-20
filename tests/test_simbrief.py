"""SimBrief OFP -> flight plan -> "flight planned route" clearance.

The OFP fixture follows the shape of SimBrief's xml.fetcher JSON; the route, SID, STAR fixes and their coordinates are made up (STAR name from the UUEE CIFP).
"""
import json

import pytest

from atc.brain.fsm import Brain, FlightConfig
from atc.llm.intent_parser import IntentParser
from atc.world.simbrief import apply_plan, parse_ofp
from conftest import FIX
from test_scenario import ULLI_DL, say, sim_at, wx

OFP = json.loads((FIX / "simbrief_ofp_ULLI_UUEE.json").read_text())


def test_parse_ofp():
    fp = parse_ofp(OFP)
    assert (fp.origin, fp.destination, fp.callsign, fp.acf_type, fp.cruise_ft) == \
        ("ULLI", "UUEE", "AFL51", "A20N", 35000)
    assert (fp.sid, fp.sid_exit) == ("SUMAN1A", "SUMAN")
    assert fp.summary() == "AFL51 A20N ULLI-UUEE FL350 SUMAN1A SUMAN R805 BITSA DCT BUKUB BUKU3E"
    # no navlog: the route string gives the SID and its exit fix
    fp = parse_ofp({**OFP, "navlog": {}})
    assert (fp.sid, fp.sid_exit) == ("SUMAN1A", "SUMAN")
    with pytest.raises(ValueError, match="Unknown UserID"):
        parse_ofp({"fetch": {"status": "Error: Unknown UserID"}})


def planned(ulli, index=None, region="ICAO"):
    fc = FlightConfig(callsign="DLH4AB", region=region)
    apply_plan(fc, parse_ofp(OFP))
    assert fc.callsign == "AFL51" and fc.destination == "UUEE" and fc.first_fix == "SUMAN"
    return Brain(fc, ulli, index=index)


def at_stand(ulli):
    g = next(g for g in ulli.gates if g.name == "101A")
    return sim_at(g.lat, g.lon, ULLI_DL, **wx())


def test_clearance_as_filed(ulli):
    b = planned(ulli)
    out = say(b, IntentParser(), "Pulkovo Delivery, good day, A20N Aeroflot 51 on stand 101A, "
                                 "request IFR clearance", at_stand(ulli))
    assert out[0].text.startswith("AFL51, cleared to UUEE via radar vectors, flight planned route, runway")
    assert b.fc.cruise_ft == 35000
    b = planned(ulli, region="FAA")
    out = say(b, IntentParser(), "Pulkovo Delivery, Aeroflot 51, request IFR clearance to UUEE", at_stand(ulli))
    assert "radar vectors, then as filed, runway" in out[0].text


def test_moscow_with_a_filed_plan_needs_no_icao(ulli, city_index):
    b = planned(ulli, city_index)
    out = say(b, IntentParser(), "Pulkovo Delivery, Aeroflot 51, request IFR clearance to Moscow", at_stand(ulli))
    assert out[0].text.startswith("AFL51, cleared to UUEE via radar vectors, flight planned route")


def test_other_destination_is_not_as_filed(ulli, city_index):
    b = planned(ulli, city_index)
    out = say(b, IntentParser(), "Pulkovo Delivery, Aeroflot 51, request IFR clearance to UUDD", at_stand(ulli))
    assert out[0].text.startswith("AFL51, cleared to UUDD via radar vectors, runway")


def test_first_call_still_wins_over_the_plan(ulli):
    b = planned(ulli)
    out = say(b, IntentParser(), "Pulkovo Delivery, good day, A20N Lufthansa 4AB on stand 101A, "
                                 "request IFR clearance", at_stand(ulli))
    assert out[0].text.startswith("DLH4AB, cleared to UUEE")
