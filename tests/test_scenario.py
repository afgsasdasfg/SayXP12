"""Full departure at KBTV driven by synthetic SimStates and the rule-based parser."""
import time

from atc.brain.fsm import Brain, FlightConfig
from atc.brain.monitor import Monitor
from atc.llm.intent_parser import IntentParser
from atc.state import Phase, Position, SimState


def sim_at(lat, lon, khz, **kw):
    base = dict(lat=lat, lon=lon, com1_khz=khz, on_ground=True, acf_icao="A20N",
                wind_dir_true=150, wind_speed_kt=12, qnh_hpa=1013, engines_running=[1, 1])
    base.update(kw)
    return SimState(**base)


def say(brain, parser, text, s):
    return brain.handle(parser.parse_rules(text), s)


def test_departure_flow(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    m = Monitor(b)
    p = IntentParser()
    gate = next(g for g in kbtv.gates if g.name == "1")
    dl, gnd, twr = 119150, 126300, 118300

    # wrong frequency: ground asked for clearance -> sends us to delivery
    out = say(b, p, "Burlington ground DLH4AB request IFR clearance to KBOS", sim_at(gate.lat, gate.lon, gnd))
    assert "contact delivery 119.15" in out[0].text

    # clearance
    s = sim_at(gate.lat, gate.lon, dl)
    out = say(b, p, "Burlington delivery DLH4AB request IFR clearance to KBOS", s)
    assert out[0].position == Position.DELIVERY and "runway 15" in out[0].text
    sq = b.squawk
    # wrong squawk -> correction
    out = say(b, p, "cleared KBOS runway 15 climb 5000 squawk 1234 DLH4AB", s)
    assert out[0].text.startswith("DLH4AB, negative") and sq in out[0].text
    out = say(b, p, f"cleared KBOS runway 15 climb 5000 squawk {sq} DLH4AB", s)
    assert "readback correct" in out[0].text and "126.3" in out[0].text
    assert b.phase == Phase.CLEARANCE

    # taxi
    s = sim_at(gate.lat, gate.lon, gnd)
    out = say(b, p, "Burlington ground DLH4AB request taxi", s)
    assert "hold short runway 15" in out[0].text and b.phase == Phase.TAXI_OUT
    out = say(b, p, "taxi via G M hold short runway 15 DLH4AB", s)
    assert out == [] and b.pending is None

    # reach the hold short point -> ground hands off to tower
    hold = kbtv.nodes[b.route.nodes[-1]]
    out = m.tick(sim_at(hold.lat, hold.lon, gnd, groundspeed_kt=2))
    assert any("contact tower 118.3" in t.text for t in out) and b.phase == Phase.HOLDING
    b.pending = None

    # takeoff
    s = sim_at(hold.lat, hold.lon, twr)
    out = say(b, p, "Burlington tower DLH4AB holding short 15 ready for departure", s)
    assert "cleared for takeoff" in out[0].text and b.phase == Phase.TAKEOFF
    say(b, p, "cleared for takeoff runway 15 DLH4AB", s)

    # airborne -> handoff to departure above 1000 ft AGL
    for _ in range(4):
        out = m.tick(sim_at(hold.lat, hold.lon, twr, on_ground=False, agl_ft=1500, altitude_ft=1800,
                            vs_fpm=2000))
    assert b.phase == Phase.DEPARTURE

    # check in with departure
    out = say(b, p, "departure DLH4AB with you passing 2000", sim_at(hold.lat, hold.lon, 121100, on_ground=False,
                                                                     agl_ft=1700, altitude_ft=2000))
    assert "radar contact" in out[0].text and b.assigned_alt == 10000


def test_no_readback_repeat(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    m = Monitor(b)
    gate = kbtv.gates[0]
    s = sim_at(gate.lat, gate.lon, 119150)
    say(b, IntentParser(), "delivery DLH4AB request clearance to KBOS", s)
    b.pending.t = time.time() - 11
    out = m.tick(s)
    assert out and "how do you read" in out[0].text


def test_runway_incursion(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    m = Monitor(b)
    b.runway, b.phase = "15", Phase.TAXI_OUT
    mid = kbtv.runway_of("15")
    lat = (mid.ends[0].lat + mid.ends[1].lat) / 2
    lon = (mid.ends[0].lon + mid.ends[1].lon) / 2
    out = m.tick(sim_at(lat, lon, 126300, groundspeed_kt=10))
    assert out and "STOP" in out[0].text


def test_radio_check_and_initial_call(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    p = IntentParser()
    s = sim_at(44.47, -73.15, 119150)
    out = say(b, p, "Living good day, Lufthansa four Alpha Bravo, ready check.", s)
    assert out[0].text == "DLH4AB, Burlington Delivery, read you five"
    out = say(b, p, "Burlington Delivery, Lufthansa four alpha bravo, good day", s)
    assert out[0].text == "DLH4AB, Burlington Delivery, go ahead"
    assert "Burlington Ground" in b.stt_prompt()


def test_unknown_destination_is_rejected(kbtv):
    class FakeIndex:
        entries = {"KBOS": (), "KBTV": ()}

    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv, index=FakeIndex())
    p = IntentParser()
    s = sim_at(44.47, -73.15, 119150)
    out = say(b, p, "Delivery DLH4AB request IFR clearance to KILA", s)
    assert "unable to find KILA" in out[0].text and b.squawk is None
    out = say(b, p, "Delivery DLH4AB request IFR clearance to bos", s)
    assert "cleared to KBOS" in out[0].text


def test_frequency_nobody_listening(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    out = say(b, IntentParser(), "request taxi", sim_at(44.47, -73.15, 130000))
    assert out == []
