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
