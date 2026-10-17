"""Compare LLMs on pilot transmissions: accuracy, latency, tokens, cost per flight.

Runs every phrase through the LLM only (mode "always"), so the rules don't hide mistakes.

    python tools/bench_llm.py gpt-5-nano:10:80:minimal gpt-6-luna:20:100:minimal deepseek-v4-flash:3.36:33.6

Model spec: name[:input RUB per 1M[:output RUB per 1M[:reasoning_effort]]].
base_url / key are taken from config.toml [llm] and .env.
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atc.llm.base import LLMError  # noqa: E402
from atc.llm.intent_parser import IntentParser  # noqa: E402
from atc.llm.openai_compat import OpenAICompatLLM  # noqa: E402

CTX = {"callsign": "DLH4AB", "phase": "PARKED", "expected_station": "KBTV Delivery",
       "runways": ["15", "33", "01", "19"]}

# (transmission as Whisper might produce it, expected intent, expected slots)
CASES = [
    ("Burlington Delivery, Lufthansa 4 Alpha Bravo, request IFR clearance to Boston", "request_clearance", {}),
    ("Delivery Lufthansa four alpha bravo IFR to kilo bravo oscar sierra", "request_clearance", {}),
    ("Cleared to KBOS, radar vectors, runway 15, climb 5000, squawk 4521, Lufthansa 4AB",
     "readback", {"runway": "15", "squawk": "4521", "altitude": 5000}),
    ("cleared kbos runway one five five thousand squad four five two one lufthansa four alpha bravo",
     "readback", {"squawk": "4521"}),
    ("Ground, Lufthansa 4AB, at gate 1, request push and start", "request_pushback", {}),
    ("Lufthansa four alpha bravo ready to push", "request_pushback", {}),
    ("Ground Lufthansa 4 Alpha Bravo request taxi", "request_taxi", {}),
    ("we're ready to roll out to the runway, Lufthansa 4AB", "request_taxi", {}),
    ("Taxi via Golf Mike, hold short runway 15, Lufthansa 4AB", "readback", {"hold_short": "15"}),
    ("holding short one five lufthansa four alpha bravo", "readback", {"hold_short": "15"}),
    ("Tower, Lufthansa 4AB, holding short runway 15, ready for departure", "ready_for_departure", {}),
    ("Lufthansa four alpha bravo is ready", "ready_for_departure", {}),
    ("Runway 15 cleared for takeoff, Lufthansa 4AB", "readback", {"runway": "15", "cleared_takeoff": True}),
    ("cleared takeoff one five lufthansa four alpha bravo", "readback", {"cleared_takeoff": True}),
    ("Departure, Lufthansa 4AB with you, passing 2000 climbing 5000", "check_in", {}),
    ("Contact departure 121.1, Lufthansa 4AB", "readback", {"frequency": "121.1"}),
    ("over to departure one two one decimal one lufthansa four alpha bravo", "readback", {"frequency": "121.1"}),
    ("Climb flight level 350, Lufthansa 4AB", "readback", {"flight_level": 350}),
    ("Center, Lufthansa 4AB, request descent", "request_descent", {}),
    ("Lufthansa 4AB requesting lower", "request_descent", {}),
    ("request the ILS approach runway 15, Lufthansa 4AB", "request_approach", {}),
    ("Tower, Lufthansa 4AB, established ILS runway 15", "report_final", {}),
    ("Runway 15 cleared to land, Lufthansa 4AB", "readback", {"cleared_land": True}),
    ("Ground, Lufthansa 4AB, runway vacated, request taxi to the gate", "request_taxi", {}),
    ("Say again, Lufthansa 4AB", "say_again", {}),
    ("sorry, could you repeat that last part", "say_again", {}),
    ("Roger, Lufthansa 4AB", "acknowledge", {}),
    ("unable, Lufthansa 4AB", "unable", {}),
    ("uh lufthansa four alpha bravo uh", "other", {}),
    ("the quick brown fox jumps over the lazy dog", "other", {}),
]


def slot_ok(got: dict, want: dict) -> bool:
    for k, v in want.items():
        g = got.get(k)
        if k == "frequency":
            try:
                if abs(float(g) - float(v)) > 0.001:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(g).upper().lstrip("0") != str(v).upper().lstrip("0"):
            return False
    return True


async def bench(spec: str, cfg: dict) -> None:
    name, *rest = spec.split(":")
    pin = float(rest[0]) if len(rest) > 0 and rest[0] else None
    pout = float(rest[1]) if len(rest) > 1 and rest[1] else None
    # explicit (possibly empty = none) in the spec, else the config value
    effort = (rest[2] or None) if len(rest) > 2 else (None if rest else cfg.get("reasoning_effort") or None)
    llm = OpenAICompatLLM(name, cfg.get("base_url", "https://api.aitunnel.ru/v1"),
                          cfg.get("api_key_env", "OPENAI_API_KEY"), timeout=30, reasoning_effort=effort)
    p = IntentParser(llm, mode="always")
    lat, tin, tout, ok_int, ok_slot, errors = [], 0, 0, 0, 0, 0
    print(f"\n=== {name} (reasoning_effort={effort}) ===")
    for text, want_intent, want_slots in CASES:
        await asyncio.sleep(1.0)  # stay under the provider's rate limit
        t0 = time.perf_counter()
        try:
            it = await p.parse_llm(text, CTX, fallback=False)  # errors count, no rules rescue
        except LLMError as e:
            errors += 1
            print(f"  ERR  {text[:50]!r}: {e}")
            continue
        lat.append(time.perf_counter() - t0)
        u = llm.last_usage
        tin += u.get("prompt_tokens", 0)
        tout += u.get("completion_tokens", 0)
        good_i = it.intent == want_intent or (want_intent == "other" and it.unclear)
        good_s = slot_ok(it.readback, want_slots)
        ok_int += good_i
        ok_slot += good_s
        if not (good_i and good_s):
            print(f"  MISS {text[:60]!r}: got {it.intent} {it.readback} "
                  f"(want {want_intent} {want_slots})")
    n = len(CASES)
    print(f"  intent {ok_int}/{n}, slots {ok_slot}/{n}, errors {errors}")
    if lat:
        print(f"  latency median {statistics.median(lat):.2f}s, max {max(lat):.2f}s")
        print(f"  tokens per call: in {tin / len(lat):.0f}, out {tout / len(lat):.0f}")
        if pin is not None and pout is not None:
            per_call = (tin * pin + tout * pout) / len(lat) / 1e6
            print(f"  cost: {per_call * 100:.2f} RUB per 100 calls (worst-case flight)")


async def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    root = Path(__file__).resolve().parent.parent
    cfg = tomllib.loads((root / "config.toml").read_text(encoding="utf-8-sig")).get("llm", {})
    specs = sys.argv[1:] or [cfg.get("model", "gpt-5-nano")]
    for s in specs:
        try:
            await bench(s, cfg)
        except LLMError as e:
            print(f"{s}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
