"""SayXP12 entry point.

    python -m atc.main                 # voice: hold PTT, speak, release
    python -m atc.main --text          # type pilot transmissions instead of speaking
    python -m atc.main --text --no-tts # pure text, no audio at all
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
import tomllib
from pathlib import Path

from .brain.fsm import Brain, FlightConfig
from .brain.monitor import Monitor
from .llm.base import LLMError, make_llm
from .llm.intent_parser import IntentParser
from .sim_link import SimLink
from .state import Transmission
from .world.aptdat import AptIndex, default_xp_root

log = logging.getLogger("main")


def load_config(path: str) -> dict:
    # utf-8-sig: Notepad / PowerShell often save with a BOM, which tomllib rejects
    return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))


def setup_logging() -> Path:
    Path("logs").mkdir(exist_ok=True)
    logfile = Path("logs") / time.strftime("session-%Y%m%d-%H%M%S.log")
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(logfile, encoding="utf-8")])
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    logging.getLogger().addHandler(console)
    for noisy in ("httpx", "httpcore", "faster_whisper", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logfile


def flight_config(cfg: dict) -> FlightConfig:
    fl, atc = cfg.get("flight", {}), cfg.get("atc", {})
    return FlightConfig(
        callsign=fl.get("callsign", "N123AB"), telephony=fl.get("telephony") or None,
        destination=(fl.get("destination") or None), cruise_ft=int(fl.get("cruise_ft", 35000)),
        initial_alt_ft=int(fl.get("initial_alt_ft", 5000)), sid=fl.get("sid") or None,
        first_fix=fl.get("first_fix") or None, arrival_gate=fl.get("arrival_gate") or None,
        handoff_departure_agl=int(atc.get("handoff_departure_agl", 1000)),
        handoff_center_ft=int(atc.get("handoff_center_ft", 10000)),
        center_khz=int(atc.get("center_khz", 127900)), region=atc.get("region", "ICAO"),
        transition_alt=int(atc.get("transition_alt", 5000)),
        strict_readback=bool(atc.get("strict_readback", False)),
    )


class App:
    def __init__(self, cfg: dict, text_mode: bool, tts_on: bool):
        self.cfg = cfg
        self.text_mode = text_mode
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue[Transmission] = asyncio.Queue()
        self.ptt_down = False
        self.speaking = False
        xp = cfg.get("xplane", {})
        self.xp_root = Path(xp.get("root") or default_xp_root() or ".")
        v = cfg.get("voice", {})
        self.tts = None
        if tts_on:
            from .voice.tts import TTS
            self.tts = TTS(v.get("voices"), radio=v.get("radio_effect", True),
                           length_scale=v.get("speech_rate", 0.9))
        self.stt = self.recorder = self.ptt = None
        if not text_mode:
            from .voice.ptt import PTT
            from .voice.stt import STT, Recorder
            self.stt = STT(v.get("whisper_model", "small.en"), v.get("whisper_device", "auto"))
            self.recorder = Recorder(v.get("input_device") or None)
            self.ptt = PTT(self._on_ptt, v.get("ptt_key") or None)
        self.link = SimLink(xp.get("host", "localhost"), int(xp.get("port", 8086)),
                            on_command=(self.ptt.sim_command if self.ptt else None))
        try:
            llm = make_llm(cfg.get("llm", {}))
        except LLMError as e:
            print(f"LLM disabled ({e}); using the rule-based parser")
            llm = None
        self.parser = IntentParser(llm)
        self.brain: Brain | None = None
        self.monitor: Monitor | None = None

    # ------------------------------------------------------------- PTT / STT
    def _on_ptt(self, down: bool) -> None:
        """Called from the keyboard thread or the sim websocket task."""
        self.ptt_down = down
        if down:
            self.recorder.start()
            print("\U0001F3A4 ...", flush=True)
        else:
            audio = self.recorder.stop()
            if self.loop:
                self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._transcribe(audio)))

    async def _transcribe(self, audio) -> None:
        prompt = self.brain.stt_prompt() if self.brain else ""
        text = await asyncio.get_running_loop().run_in_executor(None, self.stt.transcribe, audio, prompt)
        if text:
            await self.pilot_says(text)

    async def pilot_says(self, text: str) -> None:
        s = self.link.state()
        print(f"PILOT [{s.tx_khz / 1000:.3f}]: {text}", flush=True)
        intent = await self.parser.parse(text, self.brain.context())
        log.info("intent: %s", intent)
        for tx in self.brain.handle(intent, s):
            await self.queue.put(tx)

    # ---------------------------------------------------------------- output
    async def speaker(self) -> None:
        while True:
            tx = await self.queue.get()
            while self.ptt_down:  # never talk over the pilot
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.6)  # natural gap after the pilot's call
            s = self.link.state()
            tuned = (not tx.freq_khz) or any(abs(tx.freq_khz - f) <= 5 for f in (s.com1_khz, s.com2_khz))
            heard = tuned or not self.cfg.get("atc", {}).get("require_tuned", True) or self.text_mode
            tag = "" if tuned else f"  (not tuned to {tx.freq_khz / 1000:.3f})"
            log.info("SAY %s heard=%s: %s", tx.position.value, heard, tx.text)
            if not heard:
                continue
            print(f"ATC {tx.position.value.upper()}: {tx.text}{tag}", flush=True)
            if self.tts:
                self.speaking = True
                await asyncio.get_running_loop().run_in_executor(None, self.tts.speak, tx.spoken, tx.position.value)
                self.speaking = False
            if self.brain.pending and self.brain.pending.tx is tx:
                self.brain.pending.t = time.time()  # readback timer starts after we finish talking

    async def monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            s = self.link.state()
            if s.paused or self.speaking or self.ptt_down:
                continue
            for tx in self.monitor.tick(s):
                await self.queue.put(tx)

    def _stdin_reader(self) -> None:
        for line in sys.stdin:
            line = line.strip().lstrip("﻿")
            if line in ("/quit", "/exit"):
                self.loop.call_soon_threadsafe(self.loop.stop)
                return
            if line == "/state":
                b = self.brain
                print(f"phase={b.phase.value} expected={b.expected_position().value} "
                      f"rwy={b.runway} alt={b.assigned_alt} sq={b.squawk} pending={bool(b.pending)}")
                continue
            if line:
                asyncio.run_coroutine_threadsafe(self.pilot_says(line), self.loop)

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        print("Connecting to X-Plane Web API...", flush=True)
        asyncio.create_task(self.link.run())
        await self.link.ready.wait()
        s = self.link.state()
        index = AptIndex(self.xp_root)
        icao = self.cfg.get("flight", {}).get("departure") or index.nearest(s.lat, s.lon)
        if not icao:
            print("No airport near the aircraft; set [flight] departure in config.toml")
            return
        airport = index.load(icao)
        self.brain = Brain(flight_config(self.cfg), airport, index, self.xp_root)
        self.monitor = Monitor(self.brain)
        self.brain.update(s)
        b = self.brain
        print(f"{airport.icao} {airport.name}  ({len(airport.runways)} rwy, {len(airport.edges)} taxi edges)")
        for pos in ("DELIVERY", "GROUND", "TOWER", "DEPARTURE"):
            from .state import Position
            f = b.freq_of(Position[pos])
            print(f"  {pos.title():10s} {f / 1000:.3f}" if f else f"  {pos.title():10s} -")
        print(f"Callsign {b.fc.callsign} ({b.phr.cs_spoken}); "
              f"aircraft {s.acf_icao}; COM1 {s.com1_khz / 1000:.3f}")
        if self.tts:
            await self.loop.run_in_executor(None, self.tts.preload)
        if self.stt:
            await self.loop.run_in_executor(None, self.stt.load)
            self.ptt.start()
            print(f"Ready. Hold [{self.cfg['voice'].get('ptt_key')}] or the X-Plane 'Contact ATC' button to talk.")
        else:
            threading.Thread(target=self._stdin_reader, daemon=True).start()
            print("Ready. Type what you say on the radio (/state, /quit).")
        await asyncio.gather(self.speaker(), self.monitor_loop())


def main() -> None:
    ap = argparse.ArgumentParser(description="SayXP12 - AI ATC for X-Plane 12")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--text", action="store_true", help="type instead of speaking")
    ap.add_argument("--no-tts", action="store_true", help="print ATC instead of speaking")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logfile = setup_logging()
    print(f"log: {logfile}")
    app = App(load_config(a.config), a.text, not a.no_tts)
    try:
        asyncio.run(app.run())
    except (KeyboardInterrupt, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "Event loop stopped" not in str(e):
            raise


if __name__ == "__main__":
    main()
