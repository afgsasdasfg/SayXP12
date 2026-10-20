"""Vectors to the ILS: CIFP final approach data, geometry, and the radio flow.

Main case: the ULLI -> UUEE flight, arriving at Sheremetyevo from the north-west.
KBTV covers the fallback without CIFP and the shared Departure/Approach frequency.
"""
import math

from atc.brain import vectors
from atc.brain.fsm import Brain, FlightConfig
from atc.brain.monitor import Monitor
from atc.llm.intent_parser import IntentParser
from atc.state import Phase, Position, SimState
from atc.world.cifp import final_for_runway, parse_finals

from conftest import FIX

CTR = 127900
UUEE_MAGVAR, KBTV_MAGVAR = 12.0, -14.0   # true minus magnetic: Moscow ~12 E, Vermont ~14 W


def offset(lat, lon, brg_true, nm):
    d = nm * 1852 / 6371000.0
    p1, l1, b = math.radians(lat), math.radians(lon), math.radians(brg_true)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def air(lat, lon, khz, hdg_mag, alt, magvar, elev=0):
    return SimState(lat=lat, lon=lon, com1_khz=khz, on_ground=False, agl_ft=alt - elev, altitude_ft=alt,
                    heading_mag=hdg_mag % 360, heading_true=(hdg_mag + magvar) % 360, acf_icao="A20N",
                    wind_dir_true=250, wind_speed_kt=12, engines_running=[1, 1])


def fly(s, hdg_mag, nm, magvar):
    lat, lon = offset(s.lat, s.lon, hdg_mag + magvar, nm)
    return air(lat, lon, s.com1_khz, hdg_mag, s.altitude_ft, magvar)


def say(b, text, s):
    return b.handle(IntentParser().parse_rules(text), s)


def readback(b, tx, s):
    """Pilot repeats the instruction (the text after the callsign)."""
    return say(b, f"{tx.text.split(', ', 1)[1]}, DLH4AB", s)


def airborne(m, s):
    for _ in range(3):
        m.tick(s)


class OneAirportIndex:
    """AptIndex stand-in: the destination is loaded from the fixture."""
    def __init__(self, ap):
        self.ap = ap
        self.entries = {ap.icao: ("fixture", 0, ap.lat, ap.lon)}

    def load(self, icao):
        if icao != self.ap.icao:
            raise KeyError(icao)
        return self.ap


def fly_vectors(b, m, s, magvar, max_ticks=600):
    """Fly the assigned headings 0.3 NM per tick, read everything back; returns what ATC said."""
    said, legs = [], [b.vector.leg]
    for _ in range(max_ticks):
        hdg = b.assigned_hdg
        if b.appr_cleared:   # on the intercept heading until the localizer, then along it
            _, y = b.final.frame(s.lat, s.lon)
            hdg = b.final.course_mag if abs(y) < 0.4 else b.vector.heading
        alt = b.assigned_alt or s.altitude_ft
        s = fly(s, hdg, 0.3, magvar)
        s.altitude_ft = alt
        for t in m.tick(s):
            said.append(t.text)
            if t.expects_readback:
                assert readback(b, t, s) == [], t.text
        if b.vector.leg != legs[-1]:
            legs.append(b.vector.leg)
        if b.established:
            break
    return s, said, legs


# ------------------------------------------------------------------ CIFP
def test_cifp_finals_uuee():
    finals = parse_finals(FIX / "CIFP" / "UUEE.dat")
    i24l = final_for_runway(finals, "24L")
    assert (i24l.kind, i24l.faf, i24l.faf_alt, i24l.faf_dist_nm, i24l.course_mag) == \
        ("ILS", "EE122", 2000, 4.1, 243.0)
    i06l = final_for_runway(finals, "06L")
    assert (i06l.faf, i06l.faf_alt, i06l.faf_dist_nm) == ("EE612", 3000, 7.2)
    assert {f.kind for f in finals if f.runway == "24C"} == {"ILS", "localizer", "RNAV"}


def test_cifp_finals_kbtv():
    finals = parse_finals(FIX / "CIFP" / "KBTV.dat")
    i15 = final_for_runway(finals, "15")
    assert (i15.kind, i15.faf, i15.faf_alt, i15.faf_dist_nm, i15.course_mag) == ("ILS", "FOVES", 2000, 5.2, 146.0)
    r01 = final_for_runway(finals, "01")   # no ILS to 01: the RNAV
    assert (r01.kind, r01.faf, r01.faf_alt) == ("RNAV", "CESAL", 2400)
    assert final_for_runway(finals, "19") is None


# -------------------------------------------------------------- geometry
def test_frame_and_first_vectors(uuee):
    e = uuee.runway_end("24L")
    f = vectors.Final(e.lat, e.lon, e.heading_true, 243.0, 4.1, 2000)
    out = (f.course_true + 180) % 360
    x, y = f.frame(*offset(f.lat, f.lon, out, 15))     # 15 NM out on the extended centreline
    assert abs(x - 15) < 0.1 and abs(y) < 0.1
    # 20 NM out, 4 NM right of the inbound course (north of it): straight in, cut left 30°
    x, y = f.frame(*offset(*offset(f.lat, f.lon, out, 20), f.course_true + 90, 4))
    assert y > 3.9
    v = vectors.first_vector(f, x, y)
    assert (v.leg, v.heading) == ("intercept", 210)
    # just north of the field: downwind, cut 30° away from the final while still close
    v = vectors.first_vector(f, *f.frame(*offset(f.lat, f.lon, f.course_true + 90, 2)))
    assert (v.leg, v.heading) == ("downwind", 30)        # 063 - 30
    # far north: downwind cut in towards the final
    v = vectors.first_vector(f, *f.frame(*offset(f.lat, f.lon, f.course_true + 90, 20)))
    assert (v.leg, v.heading) == ("downwind", 90)        # 063 + 30, rounded
    # 12 NM out, 12 NM south: base leg, turn towards the course
    v = vectors.first_vector(f, *f.frame(*offset(*offset(f.lat, f.lon, out, 12), f.course_true - 90, 12)))
    assert (v.leg, v.side, v.heading) == ("base", -1, 330)


# ------------------------------------------------------------- radio flow
def test_ulli_uuee_vectors_to_ils_24l(kbtv, uuee, xp_root):
    """Arrival from Pulkovo: Center hands over to Approach, vectors, ILS 24L, Tower."""
    b = Brain(FlightConfig(callsign="DLH4AB", destination="UUEE"), kbtv,
              index=OneAirportIndex(uuee), xp_root=xp_root)   # departure field is only a stand-in
    m = Monitor(b)
    b.phase = Phase.ENROUTE
    # 60 NM north-west of Sheremetyevo, FL100, on Center
    s = air(*offset(uuee.lat, uuee.lon, 315, 60), CTR, 135, 10000, UUEE_MAGVAR)
    airborne(m, s)
    out = say(b, "Moscow Center, DLH4AB, request descent", s)
    assert "descend" in out[0].text and b.phase == Phase.ARRIVAL and b.dest is uuee
    readback(b, out[0], s)
    m.tick(s)
    assert not b.approach_handoff   # still too far out

    # inside 40 NM Center hands over to Approach by itself
    s = air(*offset(uuee.lat, uuee.lon, 315, 35), CTR, 135, 10000, UUEE_MAGVAR)
    out = m.tick(s)
    assert [t.text for t in out] == ["DLH4AB, contact approach 126.6"] and out[0].position == Position.CENTER
    assert readback(b, out[0], s) == [] and b.pending is None

    # Approach: runway 24L by the wind, ILS 24L from the CIFP (FAF EE122 at 2000 ft, 4.1 NM)
    s = air(s.lat, s.lon, 126600, 135, 10000, UUEE_MAGVAR)
    tx = say(b, "Moscow Approach, DLH4AB, descending FL100", s)[0]
    assert tx.position == Position.APPROACH, tx
    assert "expect vectors ILS approach runway 24L" in tx.text and "descend 3000 ft" in tx.text, tx.text
    assert b.arr_runway == "24L" and b.final.faf_alt == 2000 and b.final.faf_dist_nm == 4.1
    assert b.vector.leg == "downwind"
    assert readback(b, tx, s) == [] and b.pending is None

    s, said, legs = fly_vectors(b, m, s, UUEE_MAGVAR)
    assert b.established, said
    assert legs[0] == "downwind" and "base" in legs and legs[-1] == "intercept", legs
    assert any("for base, descend 2000 ft" in t for t in said), said
    clr = next(t for t in said if "cleared ILS approach runway 24L" in t)
    assert "maintain 2000 ft until established on the localizer" in clr
    assert said[-1] == "DLH4AB, contact tower 118.7"
    assert b.assigned_alt is None and b.assigned_hdg is None   # the glideslope descent is not a bust

    # Tower: the same parallel runway as the approach
    s = air(s.lat, s.lon, 118700, 243, 1800, UUEE_MAGVAR)
    out = say(b, "Sheremetyevo Tower, DLH4AB, established ILS runway 24L", s)
    assert "runway 24L, cleared to land" in out[0].text and b.phase == Phase.LANDING


def test_straight_in_and_no_cifp_fallback(kbtv):
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBTV"), kbtv)   # no X-Plane folder: no CIFP
    b.arr_runway = "15"
    m = Monitor(b)
    b.phase = Phase.ARRIVAL
    thr = kbtv.runway_end("15")
    out = (thr.heading_true + 180) % 360
    lat, lon = offset(*offset(thr.lat, thr.lon, out, 25), thr.heading_true - 90, 3)  # 3 NM left of the final
    s = air(lat, lon, 121100, 150, 5000, KBTV_MAGVAR)
    airborne(m, s)
    tx = say(b, "Burlington Approach, DLH4AB, with you 5000", s)[0]
    # FAF from the field elevation (333 + 2000, rounded to 500) on a 3° glideslope
    assert b.final.faf_alt == 2500 and b.vector.leg == "intercept"
    assert "turn right heading 170" in tx.text, tx.text
    readback(b, tx, s)
    said = []
    for _ in range(100):
        s = fly(s, b.assigned_hdg or b.vector.heading, 0.3, KBTV_MAGVAR)
        said += [t.text for t in m.tick(s)]
        if b.appr_cleared:
            break
    assert any("descend and maintain 2500 ft until established on the localizer, cleared ILS approach runway 15"
               in t for t in said), said


def test_shared_frequency_answers_as_approach_when_inbound(kbtv):
    """KBTV: Departure and Approach are both 121.1; inbound it is Approach that answers."""
    b = Brain(FlightConfig(callsign="DLH4AB", destination="KBTV"), kbtv)
    b.phase = Phase.ARRIVAL
    assert b.position_on(121100) == Position.APPROACH
    b.phase = Phase.DEPARTURE
    assert b.position_on(121100) == Position.DEPARTURE
