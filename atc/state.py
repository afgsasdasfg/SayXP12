"""Shared dataclasses: sim state, flight phases, ATC positions, transmissions."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

M_TO_FT = 3.28084
MS_TO_KT = 1.943844


class Phase(str, enum.Enum):
    PARKED = "PARKED"
    CLEARANCE = "CLEARANCE"          # IFR clearance received, waiting push/taxi
    PUSH_START = "PUSH_START"
    TAXI_OUT = "TAXI_OUT"
    HOLDING = "HOLDING"              # at/near hold short, waiting takeoff clearance
    TAKEOFF = "TAKEOFF"              # cleared for takeoff / rolling
    DEPARTURE = "DEPARTURE"
    ENROUTE = "ENROUTE"
    ARRIVAL = "ARRIVAL"
    APPROACH = "APPROACH"
    LANDING = "LANDING"
    TAXI_IN = "TAXI_IN"


class Position(str, enum.Enum):
    DELIVERY = "Delivery"
    GROUND = "Ground"
    TOWER = "Tower"
    DEPARTURE = "Departure"
    CENTER = "Center"
    APPROACH = "Approach"


@dataclass
class SimState:
    t: float = field(default_factory=time.time)
    lat: float = 0.0
    lon: float = 0.0
    elevation_ft: float = 0.0      # MSL, geometric
    altitude_ft: float = 0.0       # indicated altitude (pilot altimeter)
    agl_ft: float = 0.0
    heading_true: float = 0.0
    heading_mag: float = 0.0
    track_true: float = 0.0
    groundspeed_kt: float = 0.0
    ias_kt: float = 0.0
    vs_fpm: float = 0.0
    on_ground: bool = True
    com1_khz: int = 0
    com2_khz: int = 0
    tx_com: int = 1                # which COM the mic is on
    xpdr_code: int = 0
    xpdr_mode: int = 0
    parking_brake: float = 0.0
    engines_running: list[int] = field(default_factory=list)
    wind_dir_true: float = 0.0     # direction wind blows FROM
    wind_speed_kt: float = 0.0
    sfc_wind_dir_true: float | None = None   # lowest layer of the regional weather (None: no dataref)
    sfc_wind_speed_kt: float | None = None
    qnh_hpa: float = 1013.25
    oat_c: float = 15.0
    acf_icao: str = ""
    tailnum: str = ""
    paused: bool = False
    zulu_sec: float = 0.0

    @property
    def tx_khz(self) -> int:
        return self.com2_khz if self.tx_com == 2 else self.com1_khz

    @property
    def surface_wind(self) -> tuple[float, float]:
        """(from true, kt) near the ground. The aircraft's own wind is the wind at its altitude:
        at FL350 it says nothing about the runway in use."""
        if self.on_ground or self.agl_ft < 1500 or self.sfc_wind_dir_true is None:
            return self.wind_dir_true, self.wind_speed_kt
        return self.sfc_wind_dir_true, self.sfc_wind_speed_kt or 0.0

    @property
    def any_engine_running(self) -> bool:
        return any(self.engines_running)


@dataclass
class Transmission:
    """One ATC radio call: what is shown on screen and what is spoken."""
    position: Position
    text: str
    spoken: str
    expects_readback: bool = False
    freq_khz: int = 0
    t: float = field(default_factory=time.time)
    proactive: bool = False   # ATC-initiated (monitor), not an answer to the pilot


@dataclass
class PilotIntent:
    intent: str
    callsign: str | None = None
    readback: dict = field(default_factory=dict)
    readback_ok: bool | None = None
    unclear: bool = False
    raw_text: str = ""
    data: dict = field(default_factory=dict)  # extra slots (destination, altitude, ...)
