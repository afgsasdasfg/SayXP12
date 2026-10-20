# SayXP12 — AI ATC for X-Plane 12

Free, local voice ATC for X-Plane 12 (a SayIntentions-style controller): it knows the real
airport data from your X-Plane install, walks you through the phases of flight, and calls you
on its own.

**Principle:** deterministic code decides *what* ATC says (runway, taxi route, altitudes,
squawk, handoffs). The LLM is used only to understand the pilot (speech → JSON intent), and
templates decide *how* it is said. The LLM never invents frequencies, taxiways, runways or altitudes.

## Status: stage 1 (MVP) + part of stage 2

| Feature | State |
|---|---|
| X-Plane Web API link (REST + WebSocket, datarefs, "Contact ATC" command as PTT) | ✅ tested live on XP 12.4.3 |
| apt.dat parser: custom scenery (scenery_packs.ini order) → Global Airports, cached index | ✅ |
| Runways, frequencies (1050–1056 / legacy 50–56), gates, taxi network (1201/1202/1204) | ✅ |
| Taxi routing: networkx A*, one-way, width class, runway-crossing detection, folded instruction | ✅ |
| Active runway by wind | ✅ |
| Delivery: IFR clearance (dest, runway, SID from CIFP by `first_fix` or radar vectors, altitude, squawk) | ✅ |
| Readback checking with corrections ("negative, squawk 4521, read back") | ✅ |
| Ground: push/start, taxi with hold short, missed-turn detection | ✅ |
| Tower: takeoff clearance, runway incursion warning, handoff to Departure | ✅ |
| Departure / Center: radar contact, climb to cruise, altitude deviation warnings | ✅ basic |
| Approach: Center → Approach handoff, vectors to the ILS (CIFP final), landing / vacate / taxi to stand | ✅ (vectors not flown live yet) |
| PTT (keyboard + joystick via X-Plane), faster-whisper STT, Piper TTS with radio effect, voice per position | ✅ |
| Intent parser: Gemini / Ollama / rule-based fallback | ✅ |
| Planned route (SimBrief): vectors end "proceed direct <next route point>"; STAR from CIFP with the descent, vectors to the ILS from its last fix | ✅ (not flown live yet) |
| Top of descent: Center calls "descend FL100" by itself, 3 NM per 1000 ft to the field (or FL100 at the planned STAR entry) | ✅ (not flown live yet) |
| ATIS, traffic (TCAS), VFR, overlay window | ⏳ stage 2–3 |

## Install (Windows)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

X-Plane 12.1.1+ is required (built-in Web API on port 8086). In X-Plane *Settings → Network*
incoming traffic must not be disabled.

LLM (optional, the rule-based parser works without one):

- **OpenAI-compatible API** (default, `provider = "openai"`): AITUNNEL (`api.aitunnel.ru/v1`,
  pay in RUB, works from Russia), OpenRouter, DeepSeek, etc. Create `.env` with
  `OPENAI_API_KEY=...`; set `base_url` / `model` in `config.toml`. For reasoning models (gpt-5*)
  keep `reasoning_effort = "minimal"`, otherwise hidden reasoning tokens cost money and latency.
- **Gemini**: `GEMINI_API_KEY=...` in `.env` (not available from Russia).

`mode = "fallback"` (default) sends a call to the LLM only when the rule-based parser can't
classify it, so standard phraseology is answered instantly and almost nothing is billed.
Without a key the app prints "LLM disabled" and runs on rules only.

Pick a model by measurement (accuracy / latency / RUB per flight on 30 pilot phrases):

```bash
.venv\Scripts\python tools\bench_llm.py gpt-6-luna:20:100:none gpt-5-nano:10:80:minimal
```

(`name:input RUB per 1M:output RUB per 1M:reasoning_effort`; take the prices from the model's
own page, the catalog list shows the discounted cache price.)
- **Ollama**: `ollama pull qwen2.5:7b`, then set `provider = "ollama"` in `config.toml`.
- **rules**: no LLM, regex parser.

On first run Piper voices (~60 MB each) and the Whisper model (`small.en`, ~480 MB) are
downloaded into `models/`.

Whisper on the GPU needs the CUDA 12 runtime libraries; without them it falls back to CPU
automatically (a few seconds per phrase). To enable the GPU:

```bash
.venv\Scripts\python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

## Run

Edit `config.toml` (callsign, destination, cruise altitude, PTT key). To fly the plan from SimBrief, put
your Pilot ID (simbrief.com → Account Settings) in `[simbrief] pilot_id`: at start the latest OFP
gives the destination, cruise level, callsign and first fix, and Delivery clears you via the
"flight planned route" (FAA: "then as filed"). Radar vectors end with "proceed direct" the next point
of the planned route, and the descent comes with a STAR from the destination's CIFP (the planned one
when it is there). Without a plan: "proceed direct <destination>" and vectors without a STAR. Check it with `python -m atc.world.simbrief <pilot_id>`.
Then:

```bash
.venv\Scripts\python -m atc.main
```

Hold the PTT key (default right Ctrl) or a joystick button bound in X-Plane to **Contact ATC**
(`sim/operation/contact_atc`), speak, release. Tune the station's frequency in COM1/COM2 —
ATC only hears and answers on the frequency you transmit on (audio selector COM1/COM2).

Without a microphone / speakers:

```bash
.venv\Scripts\python -m atc.main --text --no-tts
```

Typical departure: first call to Delivery as on VATSIM, "Burlington Delivery, good day, A320
Lufthansa 4AB on stand 1, information A, request IFR clearance to Boston". The callsign, type and
stand come from this call (the config callsign is only the fallback). The destination may be an
ICAO code or a city: with one airport in the city that is enough, with several Delivery asks for
the ICAO. Then read back → Ground "request
taxi" → read back the hold short → at the holding point Ground hands you to Tower → "ready for
departure" → after takeoff Tower hands you to Departure → check in.

## Checking each piece

```bash
.venv\Scripts\python -m atc.sim_link            # live SimState from X-Plane, 1 Hz
.venv\Scripts\python -m atc.world.aptdat KBTV   # runways, frequencies, network size
.venv\Scripts\python -m atc.world.taxi KBTV 15  # taxi instructions from sample gates
.venv\Scripts\python -m atc.voice.tts "Speedbird one two three, cleared for takeoff" --pos Tower
.venv\Scripts\python -m atc.voice.stt           # record from the mic and transcribe
.venv\Scripts\python -m pytest
```

Everything is logged to `logs/session-*.log`: datarefs-derived phases, recognised text,
LLM JSON, every ATC transmission.

## Layout

```
atc/
  sim_link.py        X-Plane Web API (REST resolve + WebSocket stream)
  state.py           SimState, Phase, Position, Transmission, PilotIntent
  phase_tracker.py   airborne/landed edges, runway occupancy
  phraseology.py     ICAO/FAA templates, spoken numbers
  world/             aptdat.py, taxi.py, runways.py, cifp.py
  brain/             fsm.py (phases, positions, clearances, readbacks), monitor.py (proactive rules)
  voice/             ptt.py, stt.py, tts.py
  llm/               base.py, gemini.py, ollama.py, intent_parser.py
  main.py
tests/fixtures/      KBTV taxi network excerpt from the X-Plane 12 apt.dat
```

## Known limitations

- No Center/Approach frequencies in apt.dat: Center uses `atc.center_khz` from the config.
- Approach vectors to the final (downwind, base, 30° intercept) from the CIFP FAF. STARs only with a SimBrief plan; a CIFP STAR without runway transitions (UUEE: BUKU3E/BUKU3F) is taken as serving every runway, so the letter is the planned one, not checked against the runway. Without CIFP the FAF comes from the field elevation and a 3° glideslope.
- A missing or broken taxi network gives "taxi to runway X at your discretion".
- Taxi cost has no sharp-turn penalty yet.
