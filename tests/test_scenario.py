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


def test_handoff_thanks_and_push_facing(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    p = IntentParser()
    gate = next(g for g in kbtv.gates if g.name == "1")
    # live test: frequency readback on the old frequency must not trigger "wrong frequency"
    out = say(b, p, "26 decimal three for push and start, thank you.", sim_at(gate.lat, gate.lon, 119150))
    assert out == []
    b.phase = Phase.CLEARANCE
    out = say(b, p, "Lufthansa four Bravo, request push and start.", sim_at(gate.lat, gate.lon, 126300))
    assert "push and start approved, facing" in out[0].text


def test_taxi_with_runway_crossing_in_legs(kbtv):
    """Live test: from this spot the route to 19 crosses 15/33, but ATC said only
    'via A, hold short runway 19'. Now: hold short 15 first, then cross on reaching it."""
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    m = Monitor(b)
    p = IntentParser()
    b.phase, b.runway = Phase.CLEARANCE, "19"
    here = sim_at(44.471319, -73.153743, 126300)  # wind 150 -> runway 15 is the end in use
    out = say(b, p, "Lufthansa four alpha bravo request taxi", here)
    assert "taxi to holding point runway 19 via A, hold short runway 15" in out[0].text
    assert len(b.route.legs) == 2 and b.route.legs[1].cross == "15/33"
    say(b, p, "via alpha hold short runway one five", here)
    assert b.pending is None

    # approaching the 15/33 hold point -> crossing clearance and the next limit
    lat0, lon0 = b.hold_point()
    out = m.tick(sim_at(lat0, lon0, 126300, groundspeed_kt=10))
    assert out and out[0].text == "DLH4AB, cross runway 15, continue via A, hold short runway 19"
    assert "15/33" in b.cleared_runways
    assert b.phase == Phase.TAXI_OUT


def test_incursion_on_uncleared_crossing_runway(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    m = Monitor(b)
    b.runway, b.phase = "19", Phase.TAXI_OUT
    r = kbtv.runway_of("15")
    lat = (r.ends[0].lat + r.ends[1].lat) / 2
    lon = (r.ends[0].lon + r.ends[1].lon) / 2
    out = m._runway_incursion(sim_at(lat, lon, 126300, groundspeed_kt=10))
    assert out and "not cleared onto runway 15" in out[0].text
    b.cleared_runways.add("15/33")
    m._last.clear()
    assert m._runway_incursion(sim_at(lat, lon, 126300, groundspeed_kt=10)) == []


def test_snapshot_restore_and_skip_clearance(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    b.sim = sim_at(44.47, -73.15, 126300)
    tx = b.skip_clearance()
    assert b.phase == Phase.CLEARANCE and b.squawk and "cleared to KBOS" in tx.text
    snap = b.snapshot()
    b2 = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    b2.restore(snap)
    assert (b2.phase, b2.squawk, b2.runway, b2.fc.destination) == (Phase.CLEARANCE, b.squawk, b.runway, "KBOS")


def test_taxi_overspeed_and_confirm_readback(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", confirm_readback=True), kbtv)
    m = Monitor(b)
    p = IntentParser()
    gate = next(g for g in kbtv.gates if g.name == "1")
    b.phase = Phase.CLEARANCE
    say(b, p, "request taxi", sim_at(gate.lat, gate.lon, 126300))
    out = say(b, p, "via golf mike hold short runway 15", sim_at(gate.lat, gate.lon, 126300))
    assert out[0].text == "DLH4AB, readback correct"
    fast = sim_at(gate.lat, gate.lon, 126300, groundspeed_kt=40)
    said = [t.text for _ in range(3) for t in m._taxi_speed(fast)]
    assert said == ["DLH4AB, reduce taxi speed"]


def test_flight_level_readbacks_from_live_test(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB"), kbtv)
    p = IntentParser()
    s = sim_at(44.47, -73.15, 121100, on_ground=False, agl_ft=3000, altitude_ft=3300)
    for said in ("Climbing flight level, 1, 0, 0, Lufthansa Alpha four Alpha Bravo.",
                 "Radar contact, flight level, climbing flight level, 100, Lufthansa four Alpha Bravo.",
                 "Roger contact, climb flight level, one, zero, zero, Lufthansa four Alpha Bravo.",
                 "Radar contact, climb, flight level, 100, Lufthansa four Alpha Bravo."):
        b.phase, b.assigned_alt = Phase.DEPARTURE, 10000
        tx = b._tx(Position.DEPARTURE, b.phr.radar_contact(10000), True)
        b._expect(tx, {"altitude": 10000}, set())
        assert say(b, p, said, s) == [], said
        assert b.pending is None


def test_radar_vectors_after_departure(kbtv):
    from atc.world.aptdat import bearing_deg

    class FakeIndex:
        entries = {"KBOS": ("x", 0, 42.3656, -71.0096), "KBTV": ("x", 0, 44.47, -73.15)}

    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv, index=FakeIndex())
    m = Monitor(b)
    p = IntentParser()
    b.phase, b.runway, b.assigned_alt = Phase.DEPARTURE, "15", 5000
    # Vermont: magnetic variation ~14 W -> magnetic = true + 14
    s = sim_at(44.46, -73.14, 121100, on_ground=False, agl_ft=2000, altitude_ft=2300,
               heading_true=131, heading_mag=145)
    out = say(b, p, "Departure, Lufthansa four alpha bravo, passing 2300 climbing 5000", s)
    want = int(round(((bearing_deg(44.46, -73.14, 42.3656, -71.0096) + 14) % 360) / 10) * 10)
    assert b.assigned_hdg == want
    assert f"heading {want:03d}, climb FL100" in out[0].text  # "fly heading" or "turn ... heading"
    # readback with the heading
    out = say(b, p, f"heading {want}, climb flight level 100, Lufthansa four alpha bravo", s)
    assert out == [] and b.pending is None
    # on the heading, then 30 degrees off -> reminder
    on = sim_at(44.46, -73.14, 121100, on_ground=False, agl_ft=3000, heading_mag=want, heading_true=want - 14)
    for _ in range(3):
        m.tick(on)
    off = sim_at(44.46, -73.14, 121100, on_ground=False, agl_ft=3000, heading_mag=(want + 30) % 360)
    assert any(f"check heading, fly heading {want:03d}" in t.text for t in m._heading_deviation(off))
    # Center ends the vectors
    b.phase = Phase.ENROUTE
    out = say(b, p, "Center, Lufthansa four alpha bravo, passing flight level 110",
              sim_at(44.4, -73.0, 127900, on_ground=False, agl_ft=11000, altitude_ft=11000))
    assert "proceed direct KBOS" in out[0].text and b.assigned_hdg is None


def test_unknown_destination_is_rejected(kbtv):
    class FakeIndex:
        entries = {"KBOS": (), "KBTV": ()}

        def in_city(self, name):
            return []

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


def test_vatsim_first_call_sets_callsign_type_stand(kbtv, city_index):
    """'good day, [type] [callsign] on stand [stand]': the call wins over the config callsign."""
    b = Brain(FlightConfig(callsign="N123AB"), kbtv, index=city_index)
    p = IntentParser()
    s = sim_at(44.47, -73.15, 119150, acf_icao="")
    out = say(b, p, "Burlington Delivery, good day, A320 Lufthansa 4AB on stand 1, information alpha, "
                    "request IFR clearance to Boston", s)
    assert out[0].text.startswith("DLH4AB, cleared to KBOS")
    assert out[0].spoken.startswith("Lufthansa four Alpha Bravo")
    assert b.fc.callsign == "DLH4AB" and b.called_type == "A320" and b.acf_type == "A320"
    assert b.stand == "1"
    # later calls do not change the callsign
    say(b, p, "say again, Speedbird 12", s)
    assert b.fc.callsign == "DLH4AB"
    # survives a restart
    b2 = Brain(FlightConfig(callsign="N123AB"), kbtv, index=city_index)
    b2.restore(b.snapshot())
    assert b2.phr.cs == "DLH4AB" and b2.stand == "1"


def test_config_callsign_is_the_fallback(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    out = say(b, IntentParser(), "Burlington delivery, good day, request IFR clearance", sim_at(44.47, -73.15, 119150))
    assert out[0].text.startswith("DLH4AB, cleared to KBOS")


def test_city_with_several_airports_asks_for_icao(kbtv, city_index):
    b = Brain(FlightConfig(callsign="N123AB"), kbtv, index=city_index)
    p = IntentParser()
    s = sim_at(44.47, -73.15, 119150)
    out = say(b, p, "Burlington Delivery, good day, A320 Lufthansa 4AB on stand 1, "
                    "request IFR clearance to New York", s)
    assert out[0].text == "DLH4AB, several airports in New York, say destination ICAO"
    assert b.squawk is None
    out = say(b, p, "kilo juliett foxtrot kilo, Lufthansa 4AB", s)
    assert out[0].text.startswith("DLH4AB, cleared to KJFK") and b.fc.destination == "KJFK"


def test_unknown_city_is_rejected(kbtv, city_index):
    b = Brain(FlightConfig(callsign="N123AB"), kbtv, index=city_index)
    out = say(b, IntentParser(), "Delivery, good day, A320 Lufthansa 4AB on stand 1, request clearance to Atlantis",
              sim_at(44.47, -73.15, 119150))
    assert out[0].text == "DLH4AB, unable to find ATLANTIS, say again destination"
    assert "Atlantis" in out[0].spoken


def test_ulli_first_call_to_moscow_asks_icao(ulli, city_index):
    """Test flight ULLI -> UUEE: Moscow has several airports, so Delivery asks for the ICAO."""
    b = Brain(FlightConfig(callsign="N123AB"), ulli, index=city_index)
    p = IntentParser()
    s = sim_at(59.8005, 30.2700, 132200, acf_icao="A20N")
    out = say(b, p, "Pulkovo Delivery, good day, A20N Aeroflot 1 2 3 4 on stand 12, information alpha, "
                    "request IFR clearance to Moscow", s)
    assert out[0].text == "AFL1234, several airports in Moscow, say destination ICAO"
    assert out[0].spoken.startswith("Aeroflot one two three four")
    assert b.fc.callsign == "AFL1234" and b.called_type == "A20N" and b.stand == "12"
    out = say(b, p, "uniform uniform echo echo, Aeroflot 1234", s)
    assert out[0].text.startswith("AFL1234, cleared to UUEE") and b.fc.destination == "UUEE"


def test_ulli_icao_and_unique_city(ulli, city_index):
    p = IntentParser()
    s = sim_at(59.8005, 30.2700, 132200)
    b = Brain(FlightConfig(callsign="N123AB"), ulli, index=city_index)
    out = say(b, p, "Pulkovo Delivery, good day, A20N DLH4AB on stand 14, request IFR clearance to UUEE", s)
    assert out[0].text.startswith("DLH4AB, cleared to UUEE")
    # the way back: St. Petersburg has one airport, whichever way it is spelled
    for said in ("Saint Petersburg", "St. Petersburg"):
        b = Brain(FlightConfig(callsign="DLH4AB"), ulli, index=city_index)
        out = say(b, p, f"Delivery, good day, A20N DLH4AB on stand 12, request IFR clearance to {said}", s)
        assert out[0].text.startswith("DLH4AB, cleared to ULLI"), said
