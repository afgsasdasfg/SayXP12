"""Pilot transmission -> PilotIntent.

The LLM only classifies the call and pulls out slots. Numbers are normalised and
cross-checked by deterministic code; whether a readback is correct is decided by
atc.brain, never by the LLM.
"""
from __future__ import annotations

import difflib
import json
import logging
import re

from ..state import PilotIntent
from .base import LLM, LLMError

log = logging.getLogger("intent")

INTENTS = [
    "request_clearance",    # IFR clearance request
    "readback",             # reading back an instruction
    "request_pushback",
    "request_startup",
    "request_taxi",
    "ready_for_departure",
    "check_in",             # initial call on a new frequency ("with you, passing 3000")
    "request_climb",
    "request_descent",
    "request_approach",
    "report_final",         # "established ILS 27", "on final", request landing
    "report_position",
    "say_again",            # pilot asks ATC to repeat
    "radio_check",          # "radio check, how do you read"
    "acknowledge",          # roger / wilco without readback
    "unable",
    "other",
]

SLOTS = ("runway", "hold_short", "squawk", "altitude", "flight_level", "frequency",
         "destination", "sid", "heading", "taxiways", "cleared_takeoff", "cleared_land",
         "line_up", "gate", "information")

SYSTEM_PROMPT = """You parse pilot radio transmissions for an ATC simulator.
Return ONLY a JSON object:
{"intent": one of %s,
 "callsign": string or null,
 "readback": {slot: value, ...},   // values the pilot read back or stated
 "data": {slot: value, ...},       // values the pilot requested (destination, altitude, gate)
 "unclear": true if the transmission is garbled or you cannot tell what the pilot wants}
Slots: runway ("27L"), hold_short ("27L"), squawk ("4521"), altitude (feet, integer),
flight_level (integer, 350), frequency ("121.905"), destination (ICAO), sid, heading (integer),
taxiways (list of names ["B","A3"]), cleared_takeoff (true), cleared_land (true), line_up (true), gate, information (ATIS letter).
Speech-to-text output may spell numbers as words ("one two one decimal niner") or use the NATO
alphabet ("bravo" = "B"); convert them. Never invent values that were not said.
A transmission that repeats an ATC instruction is "readback" even if it also contains a request.""" % (
    json.dumps(INTENTS),)

WORD_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "tree": "3", "four": "4", "fower": "4",
    "five": "5", "fife": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9",
}
NATO_WORDS = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J",
    "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
    "whiskey": "W", "whisky": "W", "x-ray": "X", "xray": "X", "yankee": "Y", "zulu": "Z",
}
SIDE = {"left": "L", "right": "R", "center": "C", "centre": "C"}


def normalize(text: str) -> str:
    """Lowercase, spoken digits -> digits, digit runs joined, 'thousand' expanded."""
    t = text.lower().replace(",", " ").replace("-", " ")
    t = re.sub(r"[^\w\s\.]", " ", t)
    words = [WORD_DIGITS.get(w, w) for w in t.split()]
    out: list[str] = []
    for w in words:  # join single digits said one by one: "1 5" -> "15"
        if out and re.fullmatch(r"\d", w) and re.fullmatch(r"\d+", out[-1]) and len(out[-1]) < 6:
            out[-1] += w
        else:
            out.append(w)
    t = " ".join(out)
    t = re.sub(r"(\d+) (?:decimal|point|dot) (\d+)", r"\1.\2", t)
    t = re.sub(r"(\d+) thousand (\d) hundred", lambda m: str(int(m.group(1)) * 1000 + int(m.group(2)) * 100), t)
    t = re.sub(r"(\d+) thousand", lambda m: str(int(m.group(1)) * 1000), t)
    t = re.sub(r"(\d+) hundred", lambda m: str(int(m.group(1)) * 100), t)
    return t


def nato_letter(word: str) -> str | None:
    """NATO word -> letter, tolerant to Whisper spellings ("kila", "oskar", "siera")."""
    if word in NATO_WORDS:
        return NATO_WORDS[word]
    if len(word) < 4:
        return None
    close = difflib.get_close_matches(word, NATO_WORDS.keys(), n=1, cutoff=0.75)
    return NATO_WORDS[close[0]] if close else None


def nato_spelling(words: list[str]) -> str | None:
    letters = [nato_letter(w) for w in words]
    return "".join(letters) if all(letters) else None


_DEST_KEY = re.compile(r"\b(?:cleared|clearance|ifr|destination|flight plan)\b")
_NOT_ICAO = {"TAXI", "PUSH", "LAND", "FROM", "WITH", "HOLD", "TAKE", "LINE", "THAT", "THE", "AND",
             "YOU", "FOR", "VIA", "NOW", "OUR", "TO", "IFR"}


def _destination(t: str) -> str | None:
    """Destination after 'clearance/cleared/ifr ... to|for': NATO-spelled (4 words) or a 3-4 letter code."""
    for k in _DEST_KEY.finditer(t):
        words = t[k.end():].split()
        # skip up to two filler words: "clearance to", "ifr clearance for", "cleared to"
        for skip in range(0, 3):
            rest = words[skip:]
            if len(rest) >= 4 and (spelled := nato_spelling(rest[:4])):
                return spelled
            if rest and re.fullmatch(r"[a-z]{3,4}", rest[0]) and rest[0].upper() not in _NOT_ICAO \
                    and (skip == 0 or words[skip - 1] in ("to", "for", "clearance")):
                return rest[0].upper()
            if not rest or rest[0] not in ("to", "for", "clearance", "ifr"):
                break
    return None


def extract_slots(text: str) -> dict:
    """Regex slot extraction from a normalised transmission."""
    t = normalize(text)
    s: dict = {}
    if m := re.search(r"hold(?:ing)? short(?: of)?(?: runway)? (\d{1,2})(?: ?(left|right|center|centre|[lrc]))?\b", t):
        s["hold_short"] = m.group(1).zfill(2) + SIDE.get(m.group(2) or "", (m.group(2) or "").upper())
    if m := re.search(r"runway (\d{1,2})(?: ?(left|right|center|centre|[lrc]))?\b", t):
        s["runway"] = m.group(1).zfill(2) + SIDE.get(m.group(2) or "", (m.group(2) or "").upper())
    if m := re.search(r"(?:squawk(?:ing)?|transponder|code) (\d{4})\b", t):
        s["squawk"] = m.group(1)
    if m := re.search(r"flight level (\d{2,3})\b", t):
        s["flight_level"] = int(m.group(1))
    if m := re.search(r"(?:climb|climbing|descend|descending|maintain|maintaining|passing|level|initial|altitude)"
                      r"(?: and maintain)?(?: to)? (\d{3,5})(?: feet| ft)?\b", t):
        s["altitude"] = int(m.group(1))
    elif m := re.search(r"\b(\d{3,5}) (?:feet|ft)\b", t):
        s["altitude"] = int(m.group(1))
    if m := re.search(r"\b(1[1-3]\d\.\d{1,3})\b", t):
        s["frequency"] = m.group(1)
    elif m := re.search(r"\b([1-3]\d\.\d{1,3})\b", t):
        s["frequency"] = "1" + m.group(1)  # Whisper drops the leading "one": "26 decimal 3"
    if "cleared for take off" in t or "cleared for takeoff" in t or "cleared takeoff" in t:
        s["cleared_takeoff"] = True
    if "cleared to land" in t:
        s["cleared_land"] = True
    if "line up" in t or "lining up" in t or "position and hold" in t:
        s["line_up"] = True
    dest = _destination(t)
    if dest:
        s["destination"] = dest
    if m := re.search(r"(?:information|info) ([a-z]+)\b", t):
        s["information"] = NATO_WORDS.get(m.group(1), m.group(1)[:1].upper())
    # taxiways after "via": NATO words / letter+digits until a keyword
    if m := re.search(r"\bvia (.+?)(?:hold|cross|runway|$)", t):
        twys = []
        for w in m.group(1).split():
            if w in NATO_WORDS:
                twys.append(NATO_WORDS[w])
            elif re.fullmatch(r"\d{1,2}", w) and twys:
                twys[-1] += w
            elif re.fullmatch(r"[a-z]\d{0,2}", w) and w not in ("a",):
                twys.append(w.upper())
        if twys:
            s["taxiways"] = twys
    return s


def classify_rules(text: str) -> str:
    t = normalize(text)
    if not t.strip():
        return "other"
    if re.search(r"\bsay again\b|\brepeat\b", t):
        return "say_again"
    # Whisper often hears "radio check" as "ready check" / "radar check"
    if re.search(r"\b(?:radio|ready|radar|radial) ?check\b|\bhow do you read\b|\bhow do you copy\b", t):
        return "radio_check"
    if re.search(r"\b(clearance|cleared to [a-z]{4}|ifr to|request ifr)\b", t) and "request" in t:
        return "request_clearance"
    if re.search(r"\bapprov", t) and "request" not in t:
        return "acknowledge"  # "push and start approved, DLH4AB" (Whisper: "approve")
    if re.search(r"\bpush ?back\b|\brequest(?:ing)? push\b|\bpush (?:and )?start\b", t):
        return "request_pushback"
    if re.search(r"\b(request(?:ing)? start|ready (?:to|for) start)", t):
        return "request_startup"
    if re.search(r"\b(request(?:ing)? taxi|ready (?:to|for) taxi|taxi to (?:the )?(?:gate|stand|parking)|request taxi)\b", t):
        return "request_taxi"
    if re.search(r"\bready (?:for )?(?:departure|take ?off)\b|\bholding short.*ready\b|\bready at\b", t):
        return "ready_for_departure"
    if re.search(r"\b(established|on final|final runway|request(?:ing)? landing|inbound for landing)\b", t):
        return "report_final"
    if re.search(r"\brequest(?:ing)? (?:the )?(?:ils|rnav|visual|approach)\b", t):
        return "request_approach"
    if re.search(r"\brequest(?:ing)? (?:lower|descent)\b", t):
        return "request_descent"
    if re.search(r"\brequest(?:ing)? (?:higher|climb)\b", t):
        return "request_climb"
    if re.search(r"\bunable\b", t):
        return "unable"
    slots = extract_slots(text)
    if any(k in slots for k in ("squawk", "hold_short", "cleared_takeoff", "cleared_land", "line_up",
                                "frequency")) or re.search(r"\b(cleared to|climb|descend|taxi via)\b", t):
        return "readback"
    if re.search(r"\b(with you|checking in|good (morning|afternoon|evening|day)|passing|climbing|descending|level)\b", t):
        return "check_in"
    if re.search(r"\b(roger|wilco|copy|will do)\b", t):
        return "acknowledge"
    return "other"


def _norm_value(slot: str, v):
    if v is None or v == "":
        return None
    if slot in ("runway", "hold_short"):
        m = re.match(r"(?:RWY|RUNWAY)?\s*(\d{1,2})\s*([LRC])?", str(v).strip().upper())
        return (m.group(1).zfill(2) + (m.group(2) or "")) if m else str(v).upper()
    if slot == "squawk":
        d = re.sub(r"\D", "", str(v))
        return d.zfill(4) if d else None
    if slot in ("altitude", "flight_level", "heading"):
        try:
            return int(float(str(v).replace(",", "").lower().replace("ft", "").replace("fl", "")))
        except ValueError:
            return None
    if slot == "frequency":
        m = re.search(r"\d{3}\.\d{1,3}", str(v))
        return m.group(0) if m else None
    if slot == "taxiways":
        return [str(x).upper() for x in v] if isinstance(v, list) else [str(v).upper()]
    if slot in ("destination", "gate", "sid", "information"):
        return str(v).upper()
    return v


class IntentParser:
    """mode "fallback": rules first, the LLM only for calls the rules can't classify
    (instant and nearly free for standard phraseology); mode "always": LLM for every call."""

    def __init__(self, llm: LLM | None = None, mode: str = "fallback"):
        self.llm = llm
        self.mode = mode

    def parse_rules(self, text: str) -> PilotIntent:
        intent = classify_rules(text)
        slots = extract_slots(text)
        data = {k: slots[k] for k in ("destination", "gate") if k in slots}
        if intent in ("request_climb", "request_descent"):
            data.update({k: slots[k] for k in ("altitude", "flight_level") if k in slots})
        return PilotIntent(intent=intent, readback=slots, unclear=intent == "other",
                           raw_text=text, data=data)

    async def parse(self, text: str, context: dict) -> PilotIntent:
        text = text.strip()
        if not text:
            return PilotIntent(intent="other", unclear=True, raw_text=text)
        if self.llm is None:
            return self.parse_rules(text)
        if self.mode == "fallback":
            ruled = self.parse_rules(text)
            if ruled.intent != "other":
                return ruled
            log.info("rules could not classify, asking LLM")
        return await self.parse_llm(text, context)

    async def parse_llm(self, text: str, context: dict, fallback: bool = True) -> PilotIntent:
        user = (f"Context: {json.dumps(context)}\n"
                f"Pilot transmission: \"{text}\"")
        try:
            obj = await self.llm.complete_json(SYSTEM_PROMPT, user)
        except LLMError as e:
            if not fallback:
                raise
            log.warning("LLM failed (%s), falling back to rules", e)
            return self.parse_rules(text)
        log.info("LLM JSON: %s", json.dumps(obj))
        intent = obj.get("intent")
        if intent not in INTENTS:
            return PilotIntent(intent="other", unclear=True, raw_text=text)
        rb = {k: _norm_value(k, v) for k, v in (obj.get("readback") or {}).items() if k in SLOTS}
        data = {k: _norm_value(k, v) for k, v in (obj.get("data") or {}).items() if k in SLOTS}
        # deterministic cross-check: fill slots the LLM missed from the regex extractor
        for k, v in extract_slots(text).items():
            rb.setdefault(k, v)
        rb = {k: v for k, v in rb.items() if v is not None}
        data = {k: v for k, v in data.items() if v is not None}
        return PilotIntent(intent=intent, callsign=obj.get("callsign"), readback=rb,
                           unclear=bool(obj.get("unclear")), raw_text=text, data=data)
