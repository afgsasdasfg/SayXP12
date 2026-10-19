"""Link to X-Plane 12 through its built-in Web API (REST + WebSocket, 12.1.1+).

REST resolves dataref/command names to session ids, the WebSocket streams values
(server pushes changed values at ~10 Hz). See
https://developer.x-plane.com/article/x-plane-web-api/

Run standalone to print SimState once a second:
    python -m atc.sim_link
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import time
from typing import Callable

import httpx
import websockets

from .state import M_TO_FT, MS_TO_KT, SimState

log = logging.getLogger("sim_link")

# name -> (SimState field, index spec or None)
DATAREFS: dict[str, tuple[str, object]] = {
    "sim/flightmodel/position/latitude": ("lat", None),
    "sim/flightmodel/position/longitude": ("lon", None),
    "sim/flightmodel/position/elevation": ("elevation_ft", None),          # m
    "sim/cockpit2/gauges/indicators/altitude_ft_pilot": ("altitude_ft", None),
    "sim/flightmodel/position/y_agl": ("agl_ft", None),                     # m
    "sim/flightmodel/position/psi": ("heading_true", None),
    "sim/flightmodel/position/mag_psi": ("heading_mag", None),
    "sim/flightmodel/position/hpath": ("track_true", None),
    "sim/flightmodel/position/groundspeed": ("groundspeed_kt", None),       # m/s
    "sim/flightmodel/position/indicated_airspeed": ("ias_kt", None),
    "sim/flightmodel/position/vh_ind_fpm": ("vs_fpm", None),
    "sim/flightmodel/failures/onground_any": ("on_ground", None),
    "sim/cockpit2/radios/actuators/com1_frequency_hz_833": ("com1_khz", None),
    "sim/cockpit2/radios/actuators/com2_frequency_hz_833": ("com2_khz", None),
    "sim/cockpit2/radios/actuators/audio_com_selection": ("tx_com", None),  # 6=COM1 7=COM2
    "sim/cockpit2/radios/actuators/transponder_code": ("xpdr_code", None),
    "sim/cockpit2/radios/actuators/transponder_mode": ("xpdr_mode", None),
    "sim/cockpit2/controls/parking_brake_ratio": ("parking_brake", None),
    "sim/flightmodel/engine/ENGN_running": ("engines_running", list(range(8))),
    "sim/weather/aircraft/wind_now_direction_degt": ("wind_dir_true", None),
    "sim/weather/aircraft/wind_now_speed_msc": ("wind_speed_kt", None),     # m/s
    "sim/weather/aircraft/qnh_pas": ("qnh_hpa", None),                      # Pa
    "sim/weather/aircraft/temperature_ambient_deg_c": ("oat_c", None),
    "sim/aircraft/view/acf_ICAO": ("acf_icao", None),                       # base64 data
    "sim/aircraft/view/acf_tailnum": ("tailnum", None),
    "sim/time/paused": ("paused", None),
    "sim/time/zulu_time_sec": ("zulu_sec", None),
}

# Built-in "Contact ATC" command: bind it to a joystick button in X-Plane to use as PTT.
PTT_COMMAND = "sim/operation/contact_atc"


def _convert(field: str, value):
    if field in ("elevation_ft", "agl_ft"):
        return float(value) * M_TO_FT
    if field in ("groundspeed_kt", "wind_speed_kt"):
        return float(value) * MS_TO_KT
    if field == "qnh_hpa":
        return float(value) / 100.0
    if field in ("on_ground", "paused"):
        return bool(value)
    if field == "tx_com":
        return 2 if int(value) == 7 else 1
    if field in ("acf_icao", "tailnum"):
        return base64.b64decode(value).split(b"\x00")[0].decode("ascii", "ignore").strip()
    if field == "engines_running":
        return [int(v) for v in value]
    if isinstance(getattr(SimState, field, None), int) or field in (
        "com1_khz", "com2_khz", "xpdr_code", "xpdr_mode"
    ):
        return int(value)
    return float(value)


class SimLink:
    def __init__(self, host: str = "localhost", port: int = 8086,
                 on_command: Callable[[str, bool], None] | None = None,
                 watch_commands: tuple[str, ...] = (PTT_COMMAND,)):
        self.rest = f"http://{host}:{port}/api/v3"
        self.ws_url = f"ws://{host}:{port}/api/v3"
        self.on_command = on_command
        self.watch_commands = watch_commands
        self._ids: dict[int, str] = {}          # dataref id -> SimState field
        self._cmd_ids: dict[int, str] = {}
        self._name_to_cmd: dict[str, int] = {}
        self._state = SimState()
        self._req = itertools.count(1)
        self._ws = None
        self.ready = asyncio.Event()
        self.connected = False

    # ---------- REST ----------
    async def _resolve(self, client: httpx.AsyncClient) -> None:
        self._ids.clear()
        missing = []
        for name, (fld, _) in DATAREFS.items():
            r = await client.get(f"{self.rest}/datarefs", params={"filter[name]": name})
            if r.status_code != 200:
                missing.append(name)
                continue
            self._ids[r.json()["data"][0]["id"]] = fld
        if "lat" not in self._ids.values():
            # X-Plane answers before a flight is loaded (or with incoming traffic disabled -> 403)
            raise ConnectionError(f"X-Plane not ready (HTTP {r.status_code}); still loading a flight?")
        for name in missing:
            log.warning("dataref missing in sim: %s", name)
        self._cmd_ids.clear()
        for name in self.watch_commands:
            r = await client.get(f"{self.rest}/commands", params={"filter[name]": name})
            if r.status_code == 200:
                cid = r.json()["data"][0]["id"]
                self._cmd_ids[cid] = name
                self._name_to_cmd[name] = cid

    async def capabilities(self) -> dict:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as c:
            r = await c.get(self.rest.rsplit("/", 1)[0] + "/capabilities")
            r.raise_for_status()
            return r.json()

    # ---------- WebSocket ----------
    async def _send(self, type_: str, params: dict) -> None:
        await self._ws.send(json.dumps({"req_id": next(self._req), "type": type_, "params": params}))

    async def _subscribe(self) -> None:
        refs = []
        idx_by_field = {fld: idx for _, (fld, idx) in DATAREFS.items()}
        for did, fld in self._ids.items():
            item = {"id": did}
            if idx_by_field.get(fld) is not None:
                item["index"] = idx_by_field[fld]
            refs.append(item)
        await self._send("dataref_subscribe_values", {"datarefs": refs})
        if self._cmd_ids:
            await self._send("command_subscribe_is_active",
                             {"commands": [{"id": c} for c in self._cmd_ids]})

    def _handle(self, msg: dict) -> None:
        mt = msg.get("type")
        if mt == "dataref_update_values":
            for k, v in msg["data"].items():
                fld = self._ids.get(int(k))
                if fld is None:
                    continue
                try:
                    setattr(self._state, fld, _convert(fld, v))
                except (TypeError, ValueError) as e:
                    log.debug("bad value %s=%r: %s", fld, v, e)
            self._state.t = time.time()
            self.ready.set()
        elif mt == "command_update_is_active":
            for k, v in msg["data"].items():
                name = self._cmd_ids.get(int(k))
                if name and self.on_command:
                    self.on_command(name, bool(v))
        elif mt == "result" and not msg.get("success", True):
            log.error("X-Plane API error: %s %s", msg.get("error_code"), msg.get("error_message"))

    async def run(self) -> None:
        """Connect and keep streaming; reconnects if X-Plane restarts."""
        while True:
            try:
                # trust_env=False / proxy=None: X-Plane is local; a system proxy (VPN client)
                # would otherwise swallow localhost requests and answer 503
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    await self._resolve(client)
                async with websockets.connect(self.ws_url, max_size=None, proxy=None) as ws:
                    self._ws = ws
                    await self._subscribe()
                    self.connected = True
                    log.info("connected to X-Plane, %d datarefs", len(self._ids))
                    async for raw in ws:
                        self._handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep retrying
                self.connected = False
                log.warning("X-Plane link down (%s), retry in 3 s", e)
                await asyncio.sleep(3)

    def state(self) -> SimState:
        s = self._state
        return SimState(**{k: (list(v) if isinstance(v, list) else v) for k, v in s.__dict__.items()})

    async def set_command(self, name: str, duration: float = 0.0) -> None:
        cid = self._name_to_cmd.get(name)
        if cid is not None and self._ws is not None:
            await self._send("command_set_is_active",
                             {"commands": [{"id": cid, "is_active": True, "duration": duration}]})


async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    link = SimLink(on_command=lambda n, a: print(f"COMMAND {n} active={a}"))
    print(await link.capabilities())
    task = asyncio.create_task(link.run())
    await link.ready.wait()
    while True:
        s = link.state()
        print(f"{s.acf_icao} {s.tailnum} lat={s.lat:.5f} lon={s.lon:.5f} alt={s.altitude_ft:.0f} "
              f"agl={s.agl_ft:.0f} hdg={s.heading_mag:.0f} gs={s.groundspeed_kt:.0f} "
              f"ias={s.ias_kt:.0f} vs={s.vs_fpm:.0f} gnd={s.on_ground} "
              f"com1={s.com1_khz} com2={s.com2_khz} tx=COM{s.tx_com} sq={s.xpdr_code:04d}/{s.xpdr_mode} "
              f"pbrake={s.parking_brake:.1f} eng={s.engines_running[:4]} "
              f"wind={s.wind_dir_true:.0f}/{s.wind_speed_kt:.0f} qnh={s.qnh_hpa:.0f}")
        await asyncio.sleep(1)
    task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
