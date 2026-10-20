"""Full departure at KBTV driven by synthetic SimStates and the rule-based parser."""
import time

import pytest

from atc.brain.fsm import Brain, FlightConfig
from atc.brain.monitor import Monitor
from atc.llm.intent_parser import IntentParser
from atc.state import Phase, Position, SimState
from test_approach import KBTV_MAGVAR, UUEE_MAGVAR, fly_vectors


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
    assert out[0].text == "DLH4AB, contact ground 126.3 for push and start"  # no "readback correct"
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

        def in_city(self, name):
            return []

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


# ---------------------------------------------------------------- full flight
# One flight from the first radio call to the stand, as far as the rules go today:
# KBTV -> KBTV (the only airport in the fixtures), so arrival frequencies are known.
# Steps that are not implemented yet are separate xfail tests below.

DL, GND, TWR, APP, CTR = 119150, 126300, 118300, 121100, 127900
VACATE = (44.46626277, -73.14408472)  # taxiway node beside the 33 end of runway 15/33


class KbtvIndex:
    """AptIndex stand-in: destination KBTV loads the fixture airport."""

    def __init__(self, ap):
        self.ap = ap
        self.entries = {"KBTV": ("x", 0, 44.471955, -73.153277)}

    def in_city(self, name):
        return []

    def load(self, icao):
        if icao != "KBTV":
            raise KeyError(icao)
        return self.ap


def air(lat, lon, khz, alt, **kw):
    return sim_at(lat, lon, khz, on_ground=False, agl_ft=alt - 333, altitude_ft=alt, **kw)


def ticks(m, s, n=4):
    out = []
    for _ in range(n):
        out += m.tick(s)
    return out


def test_full_flight_radio_check_to_stand(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBTV"), kbtv, index=KbtvIndex(kbtv))
    m = Monitor(b)
    p = IntentParser()
    gate = next(g for g in kbtv.gates if g.name == "1")
    at_gate = lambda khz, **kw: sim_at(gate.lat, gate.lon, khz, **kw)  # noqa: E731

    # --- radio check and initial call (Delivery)
    out = say(b, p, "Burlington Delivery, Lufthansa four alpha bravo, radio check", at_gate(DL))
    assert "read you five" in out[0].text
    out = say(b, p, "Burlington Delivery, Lufthansa four alpha bravo, good day", at_gate(DL))
    assert "go ahead" in out[0].text and b.phase == Phase.PARKED

    # --- IFR clearance
    out = say(b, p, "DLH4AB request IFR clearance to KBTV", at_gate(DL))
    assert out[0].position == Position.DELIVERY and "runway 15" in out[0].text
    out = say(b, p, f"cleared KBTV runway 15 climb 5000 squawk {b.squawk} DLH4AB", at_gate(DL))
    assert "contact ground 126.3" in out[0].text and b.phase == Phase.CLEARANCE
    assert say(b, p, "126.3 for push and start, DLH4AB", at_gate(DL)) == []

    # --- push and start, taxi (Ground)
    out = say(b, p, "Burlington Ground, DLH4AB, stand 1, request push and start", at_gate(GND))
    assert "push and start approved" in out[0].text and b.phase == Phase.PUSH_START
    assert say(b, p, "push and start approved, DLH4AB", at_gate(GND)) == []
    out = say(b, p, "DLH4AB request taxi", at_gate(GND))
    assert "hold short runway 15" in out[0].text and b.phase == Phase.TAXI_OUT
    assert say(b, p, "taxi via G M hold short runway 15 DLH4AB", at_gate(GND)) == []
    hold = kbtv.nodes[b.route.nodes[-1]]
    out = m.tick(sim_at(hold.lat, hold.lon, GND, groundspeed_kt=2))
    assert any("contact tower 118.3" in t.text for t in out) and b.phase == Phase.HOLDING
    assert say(b, p, "tower 118.3, DLH4AB", sim_at(hold.lat, hold.lon, GND)) == []

    # --- takeoff (Tower)
    out = say(b, p, "Burlington Tower, DLH4AB holding short 15 ready for departure", sim_at(hold.lat, hold.lon, TWR))
    assert "cleared for takeoff" in out[0].text and b.phase == Phase.TAKEOFF
    assert say(b, p, "cleared for takeoff runway 15 DLH4AB", sim_at(hold.lat, hold.lon, TWR)) == []
    out = ticks(m, air(44.476, -73.158, TWR, 1800, vs_fpm=2000))
    assert any("contact departure 121.1" in t.text for t in out) and b.phase == Phase.DEPARTURE
    assert say(b, p, "departure 121.1, DLH4AB", air(44.476, -73.158, TWR, 1800)) == []

    # --- Departure: radar contact, vectors (no SID), climb FL100
    s = air(44.46, -73.14, APP, 2300, heading_true=131, heading_mag=145)
    out = say(b, p, "Burlington Departure, DLH4AB passing 2300 climbing 5000", s)
    assert "radar contact" in out[0].text and "climb FL100" in out[0].text and b.assigned_hdg is not None
    hdg = b.assigned_hdg
    assert say(b, p, f"heading {hdg}, climb flight level 100, DLH4AB", s) == [] and b.pending is None

    # --- Center: handoff above 10000, cruise level, end of vectors
    out = ticks(m, air(44.3, -72.9, APP, 10100, heading_mag=hdg), 1)
    assert any("contact center 127.9" in t.text for t in out) and b.phase == Phase.ENROUTE
    assert say(b, p, "center 127.9, DLH4AB", air(44.3, -72.9, APP, 10100)) == []
    out = say(b, p, "Center, DLH4AB passing flight level 105", air(44.3, -72.9, CTR, 10500))
    assert "climb FL350" in out[0].text and "proceed direct KBTV" in out[0].text and b.assigned_hdg is None
    assert say(b, p, "climb flight level 350, direct KBTV, DLH4AB", air(44.3, -72.9, CTR, 10500)) == []
    ticks(m, air(44.0, -72.5, CTR, 35000))
    assert b.alt_captured

    # --- descent (pilot request; ATC-initiated TOD is an xfail below)
    out = say(b, p, "Center, DLH4AB request descent", air(44.2, -72.8, CTR, 35000))
    assert "descend FL100" in out[0].text and b.phase == Phase.ARRIVAL and b.dest is kbtv
    assert say(b, p, "descend flight level 100, DLH4AB", air(44.2, -72.8, CTR, 35000)) == []

    # --- Approach: vectors to the ILS (no CIFP here: FAF 2500 = field 333 ft + 2000, rounded to 500)
    s = air(44.6, -73.3, APP, 10000)
    out = say(b, p, "Burlington Approach, DLH4AB, flight level 100, request ILS approach", s)
    assert "expect vectors ILS approach runway 15" in out[0].text and "descend 3500 ft" in out[0].text
    assert b.phase == Phase.APPROACH and b.assigned_alt == 3500
    assert say(b, p, f"heading {b.assigned_hdg}, descend 3500, DLH4AB", s) == []
    s, said, _ = fly_vectors(b, m, s, KBTV_MAGVAR)
    assert any("maintain 2500 ft until established on the localizer, cleared ILS approach runway 15" in t
               for t in said), said
    assert said[-1] == "DLH4AB, contact tower 118.3"

    # --- landing clearance (Tower)
    out = say(b, p, "Burlington Tower, DLH4AB established ILS runway 15", air(44.50, -73.20, TWR, 2000))
    assert "cleared to land" in out[0].text and "runway 15" in out[0].text and b.phase == Phase.LANDING
    assert say(b, p, "cleared to land runway 15, DLH4AB", air(44.50, -73.20, TWR, 2000)) == []

    # --- touchdown, rollout, vacate -> Ground
    ticks(m, sim_at(44.478, -73.162, TWR, groundspeed_kt=120))
    assert not b.tracker.airborne and b.phase == Phase.LANDING
    out = ticks(m, sim_at(*VACATE, TWR, groundspeed_kt=15), 1)
    assert any("contact ground 126.3" in t.text for t in out) and b.phase == Phase.TAXI_IN
    assert say(b, p, "ground 126.3, DLH4AB", sim_at(*VACATE, TWR)) == []

    # --- taxi to the stand
    out = say(b, p, "Burlington Ground, DLH4AB vacated runway 15, request taxi to stand", sim_at(*VACATE, GND))
    assert "taxi to stand" in out[0].text and b.gate
    if b.pending:  # the route crosses a runway: hold short readback
        assert say(b, p, f"hold short runway {b.pending.expected['hold_short']}, DLH4AB", sim_at(*VACATE, GND)) == []
    stand = next(g for g in kbtv.gates if g.name == b.gate)

    # --- engines off with the brake set at the stand -> ready for the next flight
    ticks(m, sim_at(stand.lat, stand.lon, GND, parking_brake=1.0, engines_running=[0, 0]), 1)
    assert b.phase == Phase.PARKED and b.dep is kbtv and b.squawk is None



# ULLI -> UUEE: the route of the live arrival test (flight_test_plan_ULLI-UUEE.md).
ULLI_DL, ULLI_GND, ULLI_TWR, ULLI_DEP = 121900, 121700, 118100, 120300
UUEE_APP, UUEE_TWR, UUEE_GND = 126600, 118700, 123600
UUEE_VACATE = (55.970157, 37.414127)  # taxiway node beside 24L, ~1700 m from its threshold


class RouteIndex:
    """AptIndex stand-in for ULLI -> UUEE: only the destination needs to load."""

    def __init__(self, uuee):
        self.uuee = uuee
        self.entries = {"ULLI": ("x", 0, 59.800277778, 30.2625), "UUEE": ("x", 0, 55.970167, 37.407833)}

    def in_city(self, name):
        return []

    def load(self, icao):
        if icao != "UUEE":
            raise KeyError(icao)
        return self.uuee


def wx(**kw):
    """Pulkovo and Sheremetyevo: same westerly wind, runways 28L and 24L in use."""
    return dict(wind_dir_true=280, wind_speed_kt=12, **kw)


def test_full_flight_ulli_to_uuee(ulli, uuee, xp_root):
    b = Brain(FlightConfig(callsign="DLH4AB"), ulli, index=RouteIndex(uuee), xp_root=xp_root)
    m = Monitor(b)
    p = IntentParser()
    stand = next(g for g in ulli.gates if g.name == "101A")
    at = lambda khz, **kw: sim_at(stand.lat, stand.lon, khz, **wx(**kw))  # noqa: E731

    # --- Pulkovo Delivery: radio check, clearance (destination only named in the request)
    out = say(b, p, "Delivery, Lufthansa four alpha bravo, radio check", at(ULLI_DL))
    assert "read you five" in out[0].text
    out = say(b, p, "Delivery, DLH4AB request IFR clearance to UUEE", at(ULLI_DL))
    assert "cleared to UUEE via radar vectors, runway 28L" in out[0].text and "departure frequency 120.3" in out[0].text
    out = say(b, p, f"cleared UUEE runway 28 left climb 5000 squawk {b.squawk} DLH4AB", at(ULLI_DL))
    assert "contact ground 121.7" in out[0].text and b.phase == Phase.CLEARANCE

    # --- Ground: push, taxi to 28L, handoff to the 10R/28L tower (first tower frequency: 118.1)
    out = say(b, p, "Ground, DLH4AB request push and start", at(ULLI_GND))
    assert "push and start approved" in out[0].text
    out = say(b, p, "DLH4AB request taxi", at(ULLI_GND))
    assert "taxi to holding point runway 28L via D, A, hold short runway 28L" in out[0].text
    assert say(b, p, "taxi via delta alpha hold short runway 28 left DLH4AB", at(ULLI_GND)) == []
    hold = ulli.nodes[b.route.nodes[-1]]
    out = m.tick(sim_at(hold.lat, hold.lon, ULLI_GND, groundspeed_kt=2, **wx()))
    assert any("contact tower 118.1" in t.text for t in out) and b.phase == Phase.HOLDING

    # --- Tower: takeoff, handoff to Departure
    hs = sim_at(hold.lat, hold.lon, ULLI_TWR, **wx())
    out = say(b, p, "Tower, DLH4AB holding short 28 left ready for departure", hs)
    assert "runway 28L, cleared for takeoff" in out[0].text
    assert say(b, p, "cleared for takeoff runway 28 left DLH4AB", hs) == []
    out = ticks(m, air(59.80, 30.20, ULLI_TWR, 1800, vs_fpm=2000, **wx()))
    assert any("contact departure 120.3" in t.text for t in out) and b.phase == Phase.DEPARTURE

    # --- Departure: vectors towards Moscow (south-east), FL100
    s = air(59.79, 30.18, ULLI_DEP, 2300, heading_true=280, heading_mag=269, **wx())  # variation ~11 E
    out = say(b, p, "Departure, DLH4AB passing 2300 climbing 5000", s)
    assert "radar contact" in out[0].text and "climb FL100" in out[0].text
    assert b.assigned_hdg == 120  # true bearing ULLI -> UUEE ~134, minus 11 E variation, rounded to 10
    assert say(b, p, f"heading {b.assigned_hdg}, climb flight level 100, DLH4AB", s) == []

    # --- Center: handoff at 10000, FL350, direct UUEE
    out = ticks(m, air(59.6, 30.5, ULLI_DEP, 10100, **wx()), 1)
    assert any("contact center 127.9" in t.text for t in out) and b.phase == Phase.ENROUTE
    out = say(b, p, "Center, DLH4AB passing flight level 105", air(59.6, 30.5, CTR, 10500, **wx()))
    assert "climb FL350, proceed direct UUEE" in out[0].text and b.assigned_hdg is None
    assert say(b, p, "climb flight level 350, direct UUEE, DLH4AB", air(59.6, 30.5, CTR, 10500, **wx())) == []

    # --- descent on request (~200 km out); from here the field is Sheremetyevo
    out = say(b, p, "Center, DLH4AB request descent", air(57.7, 36.0, CTR, 35000, **wx()))
    assert "descend FL100" in out[0].text and b.phase == Phase.ARRIVAL and b.dest is uuee
    assert say(b, p, "descend flight level 100, DLH4AB", air(57.7, 36.0, CTR, 35000, **wx())) == []

    # --- Sheremetyevo Approach (own frequency, unlike KBTV): vectors to ILS 24L, FAF EE122 2000 ft (CIFP)
    s = air(56.3, 37.6, UUEE_APP, 10000, **wx())
    out = say(b, p, "Approach, DLH4AB descending flight level 100", s)
    assert out[0].position == Position.APPROACH
    assert "expect vectors ILS approach runway 24L" in out[0].text and "descend 3000 ft" in out[0].text
    assert b.assigned_alt == 3000 and b.final.faf_alt == 2000
    assert say(b, p, f"heading {b.assigned_hdg}, descend 3000 feet, DLH4AB", s) == []
    s, said, _ = fly_vectors(b, m, s, UUEE_MAGVAR)
    assert any("maintain 2000 ft until established on the localizer, cleared ILS approach runway 24L" in t
               for t in said), said
    assert said[-1] == "DLH4AB, contact tower 118.7"

    # --- Tower: landing clearance on the same runway as the approach
    out = say(b, p, "Tower, DLH4AB established ILS runway 24 left", air(56.02, 37.55, UUEE_TWR, 2500, **wx()))
    assert "runway 24L, cleared to land" in out[0].text and b.phase == Phase.LANDING
    assert say(b, p, "cleared to land runway 24 left DLH4AB", air(56.02, 37.55, UUEE_TWR, 2500, **wx())) == []

    # --- touchdown on 24L, vacate -> Ground 123.6
    td = uuee.runway_end("24L")
    ticks(m, sim_at(td.lat - 0.001, td.lon - 0.003, UUEE_TWR, groundspeed_kt=130, **wx()))
    assert b.phase == Phase.LANDING
    out = ticks(m, sim_at(*UUEE_VACATE, UUEE_TWR, groundspeed_kt=15, **wx()), 1)
    assert any("contact ground 123.6" in t.text for t in out) and b.phase == Phase.TAXI_IN
    assert say(b, p, "ground 123.6, DLH4AB", sim_at(*UUEE_VACATE, UUEE_TWR, **wx())) == []

    # --- taxi to a stand at Sheremetyevo, shutdown -> next flight departs from UUEE
    out = say(b, p, "Ground, DLH4AB vacated runway 24 left, request taxi to stand", sim_at(*UUEE_VACATE, UUEE_GND, **wx()))
    assert "taxi to stand" in out[0].text and b.route.ok
    if b.pending:
        assert say(b, p, f"hold short runway {b.pending.expected['hold_short']} DLH4AB",
                   sim_at(*UUEE_VACATE, UUEE_GND, **wx())) == []
    gate = next(g for g in uuee.gates if g.name == b.gate)
    ticks(m, sim_at(gate.lat, gate.lon, UUEE_GND, parking_brake=1.0, engines_running=[0, 0], **wx()), 1)
    assert b.phase == Phase.PARKED and b.dep is uuee

def test_no_readback_reminder_while_pilot_talks(kbtv):
    """Live bug: 'how do you read' came in the middle of the pilot's readback (PTT + STT time)."""
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBOS"), kbtv)
    m = Monitor(b)
    gate = kbtv.gates[0]
    s = sim_at(gate.lat, gate.lon, DL)
    say(b, IntentParser(), "delivery DLH4AB request clearance to KBOS", s)
    b.pending.t = time.time() - 11
    b.last_pilot_t = time.time()  # the pilot has spoken since the clearance (garbled readback)
    assert m.tick(s) == []


# ------------------------------------------------- not implemented yet (xfail)
# Not strict: other threads are building some of these; an XPASS means the marker can go.
# Handoff to Approach, the shared 121.1 and vectors to the ILS are implemented: no marker on those.

def _arrival_brain(ulli, uuee, alt=10000):
    """ULLI -> UUEE flight already handed to Center and cleared down to `alt`."""
    b = Brain(FlightConfig(callsign="DLH4AB", destination="UUEE"), ulli, index=RouteIndex(uuee))
    b.phase, b.runway, b.squawk, b.assigned_alt = Phase.ARRIVAL, "28L", "5304", alt
    b._load_dest()
    return b, Monitor(b)


def test_center_hands_off_to_approach(ulli, uuee):
    b, m = _arrival_brain(ulli, uuee)
    # ~30 NM north-west of UUEE, level 10000, descending into the terminal area
    out = ticks(m, air(56.4, 37.0, CTR, 10000, vs_fpm=-1500, **wx()))
    assert any("contact approach 126.6" in t.text for t in out)


def test_shared_approach_frequency_answers_as_approach(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBTV"), kbtv, index=KbtvIndex(kbtv))
    b.phase, b.runway, b.assigned_alt = Phase.ARRIVAL, "15", 10000
    b._load_dest()
    out = say(b, IntentParser(), "Burlington Approach, DLH4AB, descending FL100", air(44.88, -73.15, APP, 10000))
    assert out[0].position == Position.APPROACH


@pytest.mark.xfail(reason="ATC-initiated descent at top of descent is not implemented (PLAN stage 2)")
def test_center_starts_descent_at_tod(ulli, uuee):
    b, m = _arrival_brain(ulli, uuee, alt=35000)
    b.phase = Phase.ENROUTE
    b.alt_captured = True
    # FL350, ~100 NM north-west of UUEE: past the 3:1 top of descent (~105 NM) -> Center calls first
    out = ticks(m, air(57.3, 36.0, CTR, 35000, **wx()))
    assert any("descend" in t.text for t in out) and b.assigned_alt < 35000


def test_approach_vectors_to_ils(ulli, uuee):
    b, m = _arrival_brain(ulli, uuee)
    out = say(b, IntentParser(), "Approach, DLH4AB, FL100, request ILS approach",
              air(56.3, 37.6, UUEE_APP, 10000, heading_mag=180, heading_true=191, **wx()))
    assert "heading" in out[0].text and "cleared ILS" not in out[0].text
    assert b.assigned_hdg is not None


@pytest.mark.xfail(reason="'runway vacated' without 'request taxi' is not understood ('say again')")
def test_ground_understands_runway_vacated(ulli, uuee):
    b, m = _arrival_brain(ulli, uuee)
    b.phase = Phase.TAXI_IN
    out = say(b, IntentParser(), "Sheremetyevo Ground, Lufthansa four alpha bravo, runway 24 left vacated",
              sim_at(*UUEE_VACATE, UUEE_GND, **wx()))
    assert "taxi to stand" in out[0].text


def test_station_names_ulli_uuee(ulli, uuee, kbtv):
    """Names from the apt.dat frequency names, not the city ('St. Petersburg Delivery')."""
    b = Brain(FlightConfig(callsign="DLH4AB"), ulli)
    assert b.station_name(Position.DELIVERY) == "Pulkovo Delivery"
    assert b.station_name(Position.APPROACH) == "Peterburg Approach"
    assert b.station_name(Position.DEPARTURE) == "Pulkovo Departure"
    # UUEE's frequencies are just 'CLD', 'TWR': the airport name
    assert Brain(FlightConfig(), uuee).station_name(Position.TOWER) == "Sheremetyevo Tower"
    assert Brain(FlightConfig(), kbtv).station_name(Position.DELIVERY) == "Burlington Delivery"


def test_negative_gets_roger(ulli):
    b = Brain(FlightConfig(callsign="AFL51"), ulli)
    b.phase = Phase.CLEARANCE
    stand = next(g for g in ulli.gates if g.name == "101A")
    out = say(b, IntentParser(), "Negative, Aeroflot 51.", sim_at(stand.lat, stand.lon, 121700, **wx()))
    assert out[0].text == "AFL51, roger" and b.pending is None


def test_vatsim_first_call_sets_callsign_and_stand(ulli):
    b = Brain(FlightConfig(), ulli)  # config callsign N123AB is only a fallback
    stand = next(g for g in ulli.gates if g.name == "101A")
    out = say(b, IntentParser(), "Pulkovo Delivery, good day, A320 Lufthansa four alpha bravo on stand 101A, "
                                 "request IFR clearance to UUEE", sim_at(stand.lat, stand.lon, ULLI_DL, **wx()))
    assert out[0].text.startswith("DLH4AB")
    assert b.stand == "101A" and b.called_type == "A320"  # departure stand; b.gate is the arrival stand


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
    b = Brain(FlightConfig(), ulli, index=city_index)
    p = IntentParser()
    stand = next(g for g in ulli.gates if g.name == "101A")
    s = sim_at(stand.lat, stand.lon, ULLI_DL, **wx(acf_icao="A20N"))
    out = say(b, p, "Pulkovo Delivery, good day, A20N Aeroflot 1 2 3 4 on stand 101A, information alpha, "
                    "request IFR clearance to Moscow", s)
    assert out[0].text == "AFL1234, several airports in Moscow, say destination ICAO"
    assert out[0].spoken.startswith("Aeroflot one two three four")
    assert b.fc.callsign == "AFL1234" and b.called_type == "A20N" and b.stand == "101A"
    out = say(b, p, "uniform uniform echo echo, Aeroflot 1234", s)
    assert out[0].text.startswith("AFL1234, cleared to UUEE") and b.fc.destination == "UUEE"


def test_ulli_icao_and_unique_city(ulli, uuee, city_index):
    p = IntentParser()
    stand = next(g for g in ulli.gates if g.name == "101A")
    s = sim_at(stand.lat, stand.lon, ULLI_DL, **wx())
    b = Brain(FlightConfig(), ulli, index=city_index)
    out = say(b, p, "Pulkovo Delivery, good day, A20N DLH4AB on stand 101A, request IFR clearance to UUEE", s)
    assert out[0].text.startswith("DLH4AB, cleared to UUEE")
    # the way back: St. Petersburg has one airport, whichever way it is spelled
    gate = next(g for g in uuee.gates if g.kind == "gate")
    s = sim_at(gate.lat, gate.lon, uuee.freq("delivery").khz, **wx())
    for said in ("Saint Petersburg", "St. Petersburg"):
        b = Brain(FlightConfig(), uuee, index=city_index)
        out = say(b, p, f"Delivery, good day, A20N DLH4AB on stand {gate.name}, request IFR clearance to {said}", s)
        assert out[0].text.startswith("DLH4AB, cleared to ULLI"), said


def test_callsign_from_radio_check_on_any_frequency(ulli):
    """Live test ULLI: the first call was a radio check on Center 127.9, misheard by Whisper."""
    b = Brain(FlightConfig(callsign="DLH4AB"), ulli)
    assert "Aeroflot" in b.stt_prompt() and "Lufthansa four" not in b.stt_prompt()
    stand = next(g for g in ulli.gates if g.name == "101A")
    out = say(b, IntentParser(), "Terrific, good day, Airflot 51, radio check.",
              sim_at(stand.lat, stand.lon, 127900, **wx()))
    assert out[0].text.startswith("AFL51, ") and b.fc.callsign == "AFL51"
    assert "Aeroflot five one" in b.stt_prompt()


def test_push_and_start_whisper_variants(ulli):
    """Live test ULLI: 'rest start' got a taxi clearance, the push readback was taken as a new request."""
    b = Brain(FlightConfig(callsign="AFL51"), ulli)
    b.phase = Phase.CLEARANCE
    p = IntentParser()
    stand = next(g for g in ulli.gates if g.name == "101A")
    s = sim_at(stand.lat, stand.lon, 121700, **wx())
    out = say(b, p, "Ground, good day, Aeroflot 51", s)
    assert "go ahead" in out[0].text and b.phase == Phase.CLEARANCE
    out = say(b, p, "Ground, good day, rest start.", s)
    assert "taxi" not in out[0].text and b.phase == Phase.CLEARANCE
    out = say(b, p, "Ground negative, Aeroflot 51, request push and sort.", s)
    assert "push and start approved" in out[0].text and b.phase == Phase.PUSH_START
    assert say(b, p, "Push and start the proof facing south, Aeroflot 51.", s) == []


def test_qnh_request_gets_only_qnh(ulli):
    """Live test ULLI: 'request QNH' on Ground got a taxi clearance."""
    b = Brain(FlightConfig(callsign="AFL51"), ulli)
    b.phase = Phase.CLEARANCE
    stand = next(g for g in ulli.gates if g.name == "101A")
    out = say(b, IntentParser(), "Ground, good day, request QNH, Aeroflot 51.",
              sim_at(stand.lat, stand.lon, 121700, **wx(qnh_hpa=1036)))
    assert out[0].text == "AFL51, QNH 1036" and b.phase == Phase.CLEARANCE and b.route is None
