import asyncio

from atc import phraseology as ph
from atc.llm.base import LLM, LLMError, parse_json_loose
from atc.llm.intent_parser import (IntentParser, classify_rules, extract_identity, extract_slots, normalize,
                                   spoken_codes)


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
    # real Whisper output from the live test
    assert extract_slots("Thriving good day, I request IFR clearance to Kila Bravo Oskar Sierra.")["destination"] == "KBOS"
    for said in ("Lufthansa four Alpha Bravo, request IFR clearance for Kila Bravo Oscar Sierra.",
                 "Lufthansa four Alpha Bravo, request IFR clearance for Kilo Bravo Oscar Sierra.",
                 "request ifr to kbos", "cleared to KBOS runway 19"):
        assert extract_slots(said).get("destination") == "KBOS", said
    assert "destination" not in extract_slots("cleared for takeoff runway 19")
    s = extract_slots("taxi via golf mike, hold short runway 15")
    assert s["hold_short"] == "15" and s["taxiways"] == ["G", "M"]
    assert extract_slots("flight level three five zero")["flight_level"] == 350
    assert extract_slots("hold short runway two seven left")["hold_short"] == "27L"
    # real Whisper output from the live test
    assert extract_slots("Climb, flight level, 1, 0, 0. Lufthansa four Alpha Bravo.")["flight_level"] == 100
    assert extract_slots("Lufthansa four Alpha Bravo, run tower on 011, 8 decimals free, thank you.")["frequency"] == "118.3"


def test_rule_classifier():
    assert classify_rules("Burlington delivery, DLH4AB, request IFR clearance to KBOS") == "request_clearance"
    assert classify_rules("DLH4AB request taxi") == "request_taxi"
    assert classify_rules("request pushback") == "request_pushback"
    assert classify_rules("holding short runway 15, ready for departure") == "ready_for_departure"
    assert classify_rules("runway 15 cleared for takeoff, DLH4AB") == "readback"
    assert classify_rules("Departure, DLH4AB with you passing 2000") == "check_in"
    assert classify_rules("say again") == "say_again"
    assert classify_rules("Push start to approve for Alpha Bravo.") == "acknowledge"
    assert classify_rules("Lufthansa four Bravo, request push and start.") == "request_pushback"
    # real Whisper output from the first live test
    assert classify_rules("Living good day, Lufthansa four Alpha Bravo, ready check.") == "radio_check"
    assert classify_rules("delivery good deal, Lufthansa four Alpha Bravo, ready check.") == "radio_check"
    assert classify_rules("Burlington Delivery, DLH4AB, radio check, how do you read") == "radio_check"
    assert classify_rules("right heading 330, DLH4AB") == "readback"   # vectors: heading only
    assert classify_rules("Approach, DLH4AB, with you heading 330") == "check_in"
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


def test_first_call_identity():
    """VATSIM first call: '[type] [callsign] on stand [stand]'."""
    cases = {
        "Burlington Delivery, good day, A320 Lufthansa 4AB on stand 1, information alpha, "
        "request IFR clearance to Boston": ("DLH4AB", "A320", "1"),
        "delivery good day airbus 320 lufthansa four alpha bravo on stand alpha one two":
            ("DLH4AB", "A320", "A12"),
        "Delivery, good day, DLH4AB, A20N, stand 1": ("DLH4AB", "A20N", "1"),
        "Delivery, good day, Boeing 738 Speedbird one two on gate 3": ("BAW12", "B738", "3"),
        "delivery good day C172 november one two three alpha bravo at parking 5": ("N123AB", "C172", "5"),
    }
    for said, (cs, typ, stand) in cases.items():
        assert extract_identity(said) == {"callsign": cs, "acf_type": typ, "stand": stand}, said
    assert extract_identity("cleared KBOS runway 15 squawk 4521 DLH4AB") == {"callsign": "DLH4AB"}
    assert extract_identity("taxi via G M hold short runway 15") == {}
    it = IntentParser().parse_rules("Delivery, good day, A320 Lufthansa 4AB on stand 1, request clearance to KBOS")
    assert it.callsign == "DLH4AB" and it.data["acf_type"] == "A320" and it.data["stand"] == "1"


def test_destination_city():
    assert extract_slots("request IFR clearance to Boston, Lufthansa 4AB")["destination"] == "BOSTON"
    assert extract_slots("request clearance to New York information alpha")["destination"] == "NEW YORK"
    assert extract_slots("request clearance for kilo bravo oscar sierra")["destination"] == "KBOS"
    assert extract_slots("request ifr to kbos")["destination"] == "KBOS"
    assert spoken_codes("kilo juliett foxtrot kilo") == ["KJFK"]
    assert spoken_codes("destination KLGA, Lufthansa 4AB") == ["KLGA"]


def test_misheard_airline_names():
    """Live test ULLI: Whisper wrote 'Aeroflot 51' as 'Airflot 51'."""
    assert extract_identity("Terrific, good day, Airflot 51, radio check.") == {"callsign": "AFL51"}
    assert extract_identity("Luftansa four alpha bravo")["callsign"] == "DLH4AB"
    assert extract_identity("contact ground 121.7, hold short runway 28L") == {}
