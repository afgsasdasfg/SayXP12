from atc.phase_tracker import on_runway
from atc.world import taxi
from atc.world.aptdat import parse_airport_lines
from atc.world.runways import select_runway, wind_components

MINI = [
    "1 100 0 0 TEST Test Field",
    "100 30.00 1 0 0.25 0 0 0 09 10.0 20.0 0 0 1 0 0 0 27 10.0 20.02 0 0 1 0 0 0",
    "53 12190 GND",
    "99",
]


def test_header_runways_freqs(kbtv):
    assert kbtv.icao == "KBTV"
    assert kbtv.elevation_ft == 333
    assert sorted(r.name for r in kbtv.runways) == ["01/19", "15/33"]
    assert 2500 < kbtv.runway_of("15").length_m < 2560
    assert 125 < kbtv.runway_end("15").heading_true < 135
    assert kbtv.freq("ground").khz == 126300
    assert kbtv.freq("tower").khz == 118300
    assert kbtv.freq("delivery").khz == 119150
    assert len(kbtv.gates) == 94


def test_taxi_network(kbtv):
    assert len(kbtv.nodes) > 200
    assert {"A", "B", "C", "G"} <= {e.name for e in kbtv.edges}
    rwy_edges = [e for e in kbtv.edges if e.kind == "runway"]
    assert rwy_edges and all(e.name in ("15/33", "01/19") for e in rwy_edges)
    assert any("15" in e.zone_runways() for e in kbtv.edges)


def test_legacy_freq_and_minimal_airport():
    ap = parse_airport_lines(MINI)
    assert ap.freq("ground").khz == 121900
    assert 85 < ap.runway_end("09").heading_true < 95


def test_runway_by_wind(kbtv):
    assert select_runway(kbtv, 150, 15).name == "15"
    assert select_runway(kbtv, 330, 12).name == "33"
    assert select_runway(kbtv, 0, 0).name in ("15", "33")  # calm -> longest
    end = kbtv.runway_end("15")
    head, cross = wind_components(end, end.heading_true, 10)
    assert abs(head - 10) < 0.01 and abs(cross) < 0.01


def test_runway_too_short_for_type(kbtv):
    # live test: wind favoured 19 (01/19 is 1254 x 23 m) for an A20N
    assert select_runway(kbtv, 190, 10).name == "19"                      # no type known
    assert select_runway(kbtv, 190, 10, acf_type="A20N").name in ("15", "33")
    assert select_runway(kbtv, 190, 10, acf_type="C172").name == "19"     # fine for a Cessna
    # nothing long enough -> longest runway, never the short one
    assert select_runway(kbtv, 190, 10, acf_type="A388").name in ("15", "33")


def test_taxi_route_gate_to_runway(kbtv):
    g = next(g for g in kbtv.gates if g.name == "1")
    rt = taxi.route_to_runway(kbtv, g.lat, g.lon, "15")
    assert rt.ok and rt.hold_short == "15"
    assert rt.taxiways and rt.taxiways[-1] == "M"
    assert "" not in rt.taxiways
    last = kbtv.nodes[rt.nodes[-1]]
    assert on_runway(kbtv, last.lat, last.lon) is None  # hold short point is off the runway


def test_taxi_route_crossing(kbtv):
    g = next(g for g in kbtv.gates if g.name == "GA 1")
    rt = taxi.route_to_runway(kbtv, g.lat, g.lon, "15")
    assert rt.ok and "01/19" in rt.crossings


def test_taxi_route_to_gate(kbtv):
    g = next(g for g in kbtv.gates if g.name == "3")
    start = kbtv.runway_end("33")
    rt = taxi.route_to_point(kbtv, start.lat, start.lon, g.lat, g.lon, dest_name="3")
    assert rt.ok and rt.taxiways


def test_taxi_no_network_fallback():
    rt = taxi.route_to_runway(parse_airport_lines(MINI), 10.001, 20.01, "09")
    assert not rt.ok and rt.hold_short == "09"


def test_aircraft_class():
    assert taxi.aircraft_class("A20N") == "C"
    assert taxi.aircraft_class("B77W") == "E"
    assert taxi.aircraft_class("ZZZZ") == "C"


def test_airports_by_city(city_index, tmp_path):
    assert city_index.in_city("Boston") == ["KBOS"]          # heliport does not count
    assert city_index.in_city("new york") == ["KJFK", "KLGA"]  # untowered strip dropped
    assert city_index.in_city("Burlington") == ["KBTV"]
    assert city_index.in_city("Moscow") == ["UUDD", "UUEE", "UUWW"]
    assert city_index.in_city("Saint-Petersburg") == ["ULLI"]
    assert city_index.in_city("Atlantis") == []
    from atc.world.aptdat import AptIndex
    cached = AptIndex(tmp_path / "xp", cache_dir=tmp_path / "cache")  # from the json cache
    assert cached.in_city("New York") == ["KJFK", "KLGA"] and "KBOS" in cached.entries
