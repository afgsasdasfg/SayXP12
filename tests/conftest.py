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
