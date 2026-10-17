import asyncio

from atc import phraseology as ph
from atc.llm.base import LLM, LLMError, parse_json_loose
from atc.llm.intent_parser import IntentParser, classify_rules, extract_slots, normalize


def test_spoken_formats():
    assert ph.freq_text(118300) == "118.3"
    assert ph.freq_text(121805) == "121.805"
    assert ph.freq_text(118000) == "118.0"
    assert ph.freq(126300) == "one two six decimal three"
    assert ph.runway("09L") == "zero niner left"
    assert ph.altitude(5000) == ("5000 ft", "five thousand")
    assert ph.altitude(2500) == ("2500 ft", "two thousand five hundred")
    assert ph.altitude(35000) == ("FL350", "flight level three five zero")
    assert ph.callsign_spoken("DLH4AB") == "Lufthansa four Alpha Bravo"
    assert ph.callsign_spoken("C-GTLT") == "Charlie Golf Tango Lima Tango"
    assert ph.taxiway("B6") == "Bravo six"
    assert ph.procedure("BTV1") == "B T V one"


def test_phraseology_clearance():
    p = ph.Phraseology("DLH4AB")
    text, spoken = p.ifr_clearance("KBOS", "15", None, 5000, "4521", 121100)
    assert text == ("DLH4AB, cleared to KBOS via radar vectors, runway 15, climb 5000 ft, "
                    "squawk 4521, departure frequency 121.1")
    assert "squawk four five two one" in spoken
    assert ph.Phraseology("N1", region="FAA").qnh(1013.25)[0] == "altimeter 2992"


def test_normalize_and_slots():
    assert normalize("runway one five, squawk four five two one") == "runway 15 squawk 4521"
    assert "121.9" in normalize("one two one decimal niner")
    s = extract_slots("Cleared to KBOS, runway one five, climb five thousand, squawk four five two one, DLH4AB")
    assert s["runway"] == "15" and s["squawk"] == "4521" and s["altitude"] == 5000
    assert s["destination"] == "KBOS"
    assert extract_slots("cleared to kilo bravo oscar sierra runway one five")["destination"] == "KBOS"
    s = extract_slots("taxi via golf mike, hold short runway 15")
    assert s["hold_short"] == "15" and s["taxiways"] == ["G", "M"]
    assert extract_slots("flight level three five zero")["flight_level"] == 350
    assert extract_slots("hold short runway two seven left")["hold_short"] == "27L"


def test_rule_classifier():
    assert classify_rules("Burlington delivery, DLH4AB, request IFR clearance to KBOS") == "request_clearance"
    assert classify_rules("DLH4AB request taxi") == "request_taxi"
    assert classify_rules("request pushback") == "request_pushback"
    assert classify_rules("holding short runway 15, ready for departure") == "ready_for_departure"
    assert classify_rules("runway 15 cleared for takeoff, DLH4AB") == "readback"
    assert classify_rules("Departure, DLH4AB with you passing 2000") == "check_in"
    assert classify_rules("say again") == "say_again"
    assert classify_rules("bla bla") == "other"


class FakeLLM(LLM):
    def __init__(self, reply):
        self.reply = reply

    async def complete_json(self, system, user):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_parser_merges_and_validates():
    llm = FakeLLM({"intent": "readback", "callsign": "DLH4AB", "readback": {"squawk": "4521", "runway": "RWY 15"}})
    it = asyncio.run(IntentParser(llm, mode="always").parse("squawk 4521 runway 15 climb 5000", {}))
    assert it.intent == "readback"
    assert it.readback["runway"] == "15" and it.readback["altitude"] == 5000  # regex filled the gap

    it = asyncio.run(IntentParser(FakeLLM({"intent": "make_coffee"}), mode="always").parse("hello", {}))
    assert it.unclear

    it = asyncio.run(IntentParser(FakeLLM(LLMError("down")), mode="always").parse("request taxi", {}))
    assert it.intent == "request_taxi"  # fallback to rules


class CountingLLM(FakeLLM):
    calls = 0

    async def complete_json(self, system, user):
        CountingLLM.calls += 1
        return self.reply


def test_hybrid_mode_only_calls_llm_when_rules_fail():
    llm = CountingLLM({"intent": "request_taxi", "readback": {}})
    p = IntentParser(llm)  # default mode = fallback
    assert asyncio.run(p.parse("ground DLH4AB request taxi", {})).intent == "request_taxi"
    assert CountingLLM.calls == 0
    assert asyncio.run(p.parse("we'd like to get moving to the runway please", {})).intent == "request_taxi"
    assert CountingLLM.calls == 1


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
