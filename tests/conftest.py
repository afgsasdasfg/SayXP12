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


@pytest.fixture
def xp_root(tmp_path):
    """A fake X-Plane folder with the real CIFP files of KBTV and UUEE (X-Plane 12 default data)."""
    d = tmp_path / "Resources" / "default data" / "CIFP"
    d.mkdir(parents=True)
    for f in (FIX / "CIFP").glob("*.dat"):
        (d / f.name).write_bytes(f.read_bytes())
    return tmp_path


def load_fixture_airport(name: str):
    with open(FIX / name, "rb") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(b"1 "))
    return parse_airport_lines(lines[start:])


@pytest.fixture(scope="session")
def uuee():
    """Sheremetyevo: runways and frequencies from the X-Plane 12 apt.dat (no taxi network)."""
    return load_fixture_airport("UUEE_apt.dat")
