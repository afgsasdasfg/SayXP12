from pathlib import Path

import pytest

from atc.world.aptdat import parse_airport_lines

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def kbtv():
    """Burlington Intl taxi network excerpt from the X-Plane 12 apt.dat (Scenery Gateway)."""
    with open(FIX / "KBTV_apt.dat", "rb") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(b"1 "))
    return parse_airport_lines(lines[start:])


CITY_APT_DAT = """I
1200 Version
1 20 0 0 KBOS General Edward Lawrence Logan Intl
1302 city Boston
1302 datum_lat 42.36
1302 datum_lon -71.01
1054 128800 BOSTON TOWER
17 20 0 0 MA01 Boston Heliport
1302 city Boston
1 13 0 0 KJFK John F Kennedy Intl
1302 city New York
1302 datum_lat 40.64
1302 datum_lon -73.78
1054 119100 KENNEDY TOWER
1 21 0 0 KLGA La Guardia
1302 city New York
1302 datum_lat 40.78
1302 datum_lon -73.87
1054 118700 LA GUARDIA TOWER
1 10 0 0 NY99 Small Strip
1302 city New York
1302 datum_lat 40.70
1302 datum_lon -73.90
1 78 0 0 ULLI Pulkovo
1302 city St. Petersburg
1302 datum_lat 59.80
1302 datum_lon 30.26
1054 118100 PULKOVO TOWER
1 622 0 0 UUEE Sheremetyevo Intl
1302 city Moscow
1302 datum_lat 55.97
1302 datum_lon 37.41
1054 131500 SHEREMETYEVO TOWER
1 588 0 0 UUDD Domodedovo Intl
1302 city Moscow
1302 datum_lat 55.41
1302 datum_lon 37.90
1054 119700 DOMODEDOVO TOWER
1 686 0 0 UUWW Vnukovo Intl
1302 city Moscow
1302 datum_lat 55.60
1302 datum_lon 37.27
1054 119300 VNUKOVO TOWER
1 333 0 0 KBTV Burlington Intl
1302 city Burlington
1302 datum_lat 44.47
1302 datum_lon -73.15
1054 118300 BURLINGTON TOWER
99
"""


@pytest.fixture
def city_index(tmp_path):
    """AptIndex over a small Global Airports apt.dat: one airport in Boston, two towered in New York."""
    from atc.world.aptdat import AptIndex
    d = tmp_path / "xp" / "Global Scenery" / "Global Airports" / "Earth nav data"
    d.mkdir(parents=True)
    (d / "apt.dat").write_text(CITY_APT_DAT)
    return AptIndex(tmp_path / "xp", cache_dir=tmp_path / "cache")


def _fixture_airport(name):
    with open(FIX / f"{name}_apt.dat", "rb") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(b"1 "))
    return parse_airport_lines(lines[start:])


@pytest.fixture(scope="session")
def ulli():
    """Pulkovo: runways, frequencies, taxi network and stands from the X-Plane 12 Global Airports apt.dat."""
    return _fixture_airport("ULLI")


@pytest.fixture(scope="session")
def uuee():
    """Sheremetyevo, same excerpt as ulli."""
    return _fixture_airport("UUEE")


@pytest.fixture
def xp_root(tmp_path):
    """A fake X-Plane folder with the real CIFP files of KBTV and UUEE (X-Plane 12 default data)."""
    d = tmp_path / "Resources" / "default data" / "CIFP"
    d.mkdir(parents=True)
    for f in (FIX / "CIFP").glob("*.dat"):
        (d / f.name).write_bytes(f.read_bytes())
    return tmp_path
