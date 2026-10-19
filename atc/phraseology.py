"""Phraseology templates. Every call returns (text for the screen, text for TTS).

Numbers, names and frequencies come in from the deterministic logic; this module
only formats them (ICAO or FAA style). It never decides values.
"""
from __future__ import annotations

import re

# ICAO "tree"/"fife" sound odd through TTS; plain English except "niner"
SPOKEN_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                 "6": "six", "7": "seven", "8": "eight", "9": "niner"}
NATO = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot",
    "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliett", "K": "Kilo", "L": "Lima",
    "M": "Mike", "N": "November", "O": "Oscar", "P": "Papa", "Q": "Quebec", "R": "Romeo",
    "S": "Sierra", "T": "Tango", "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray",
    "Y": "Yankee", "Z": "Zulu",
}
AIRLINES = {
    "AAL": "American", "AFL": "Aeroflot", "AFR": "Airfrance", "BAW": "Speedbird", "DAL": "Delta",
    "DLH": "Lufthansa", "EZY": "Easy", "KLM": "KLM", "RYR": "Ryanair", "SWR": "Swiss",
    "THY": "Turkish", "UAE": "Emirates", "UAL": "United", "WZZ": "Wizz Air", "SBI": "Siberian Air",
    "SDM": "Rossiya", "UTA": "UTair", "AUA": "Austrian", "IBE": "Iberia", "TAP": "Air Portugal",
    "SAS": "Scandinavian", "FIN": "Finnair", "ACA": "Air Canada", "JBU": "JetBlue",
    "SWA": "Southwest", "ASA": "Alaska", "QTR": "Qatari", "ETD": "Etihad", "CPA": "Cathay",
    "SIA": "Singapore", "QFA": "Qantas", "ANA": "All Nippon", "JAL": "Japan Air", "CCA": "Air China",
    "CES": "China Eastern", "CSN": "China Southern", "EIN": "Shamrock", "VLG": "Vueling",
}
RWY_SIDE = {"L": "left", "R": "right", "C": "center"}


def digits(s: str | int) -> str:
    return " ".join(SPOKEN_DIGITS.get(c, c) for c in str(s) if not c.isspace())


def letters(s: str) -> str:
    out = []
    for c in s.upper():
        if c.isdigit():
            out.append(SPOKEN_DIGITS[c])
        elif c in NATO:
            out.append(NATO[c])
    return " ".join(out)


def taxiway(name: str) -> str:
    """'B6' -> 'Bravo six', 'N11' -> 'November one one'."""
    return letters(name) if name else name


def procedure(name: str) -> str:
    """'BTV1' -> 'B T V one', 'LAM3A' -> 'lam three Alpha' (fix names said as words)."""
    m = re.match(r"([A-Z]+)(\d)([A-Z]?)$", name)
    if not m:
        return letters(name)
    base = " ".join(m.group(1)) if len(m.group(1)) <= 3 else m.group(1).capitalize()
    return " ".join(x for x in (base, digits(m.group(2)), NATO.get(m.group(3), "")) if x)


def runway(name: str) -> str:
    m = re.match(r"(\d+)([LRC]?)", name)
    if not m:
        return letters(name)
    s = digits(m.group(1))
    return s + (" " + RWY_SIDE[m.group(2)] if m.group(2) else "")


def freq_text(khz: int) -> str:
    mhz, rem = divmod(int(khz), 1000)
    return f"{mhz}.{f'{rem:03d}'.rstrip('0') or '0'}"


def freq(khz: int) -> str:
    mhz, dec = freq_text(khz).split(".")
    return f"{digits(mhz)} decimal {digits(dec)}"


def _num_words(n: int) -> str:
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else " " + ones[n % 10])
    return str(n)


def altitude(ft: int, transition_alt: int = 5000) -> tuple[str, str]:
    """(text, spoken). Above transition altitude -> flight level."""
    ft = int(round(ft / 100.0) * 100)
    if ft > transition_alt:
        fl = ft // 100
        return f"FL{fl:03d}", f"flight level {digits(f'{fl:03d}' if fl >= 100 else str(fl))}"
    th, hu = divmod(ft, 1000)
    hu //= 100
    parts = []
    if th:
        parts.append(f"{_num_words(th) if th < 20 else digits(th)} thousand")
    if hu:
        parts.append(f"{_num_words(hu)} hundred")
    return f"{ft} ft", " ".join(parts) or "zero"


def heading(deg: float) -> str:
    return digits(f"{int(round(deg)) % 360 or 360:03d}")


def wind(dir_true: float, kt: float, magvar: float = 0.0) -> tuple[str, str]:
    if kt < 3:
        return "wind calm", "wind calm"
    d = int(round((dir_true - magvar) / 10.0) * 10) % 360 or 360
    return f"wind {d:03d} at {int(round(kt))}", f"wind {digits(f'{d:03d}')} degrees, {_num_words(int(round(kt))) if kt < 100 else digits(int(kt))} knots"


def callsign_spoken(cs: str, telephony: str | None = None) -> str:
    cs = cs.upper().replace(" ", "")
    if telephony:
        m = re.match(r"^[A-Z]{3}(\w+)$", cs)
        return f"{telephony} {letters(m.group(1)) if m else ''}".strip()
    m = re.match(r"^([A-Z]{3})(\d\w*)$", cs)
    if m and m.group(1) in AIRLINES:
        return f"{AIRLINES[m.group(1)]} {letters(m.group(2))}"
    return letters(cs.replace("-", ""))


class Phraseology:
    def __init__(self, callsign: str, telephony: str | None = None, region: str = "ICAO",
                 transition_alt: int = 5000):
        self.cs = callsign.upper()
        self.cs_spoken = callsign_spoken(callsign, telephony)
        self.region = region.upper()
        self.ta = transition_alt

    # helpers ---------------------------------------------------------------
    def _out(self, text: str, spoken: str, prefix: bool = True) -> tuple[str, str]:
        if prefix:
            return f"{self.cs}, {text}", f"{self.cs_spoken}, {spoken}"
        return text, spoken

    def alt(self, ft: int) -> tuple[str, str]:
        return altitude(ft, self.ta)

    def qnh(self, hpa: float) -> tuple[str, str]:
        if self.region == "FAA":
            inhg = f"{hpa * 0.0295299830714:.2f}".replace(".", "")
            return f"altimeter {inhg}", f"altimeter {digits(inhg)}"
        q = f"{int(round(hpa))}"
        return f"QNH {q}", f"QNH {digits(q)}"

    # delivery --------------------------------------------------------------
    def ifr_clearance(self, dest: str, runway_end: str, sid: str | None, init_alt: int,
                      squawk: str, dep_khz: int | None) -> tuple[str, str]:
        at, asp = self.alt(init_alt)
        route_t = f"{sid} departure" if sid else "radar vectors"
        route_s = f"{procedure(sid)} departure" if sid else "radar vectors"
        t = (f"cleared to {dest} via {route_t}, runway {runway_end}, "
             f"climb {at}, squawk {squawk}")
        s = (f"cleared to {letters(dest)} via {route_s}, runway {runway(runway_end)}, "
             f"climb {asp}, squawk {digits(squawk)}")
        if dep_khz:
            t += f", departure frequency {freq_text(dep_khz)}"
            s += f", departure frequency {freq(dep_khz)}"
        return self._out(t, s)

    def readback_correct(self, next_pos: str | None = None, next_khz: int | None = None,
                         purpose: str = "") -> tuple[str, str]:
        t, s = "readback correct", "readback correct"
        if next_pos and next_khz:
            t += f", contact {next_pos} {freq_text(next_khz)}{(' ' + purpose) if purpose else ''}"
            s += f", contact {next_pos} {freq(next_khz)}{(' ' + purpose) if purpose else ''}"
        return self._out(t, s)

    def readback_wrong(self, items: list[tuple[str, str, str]]) -> tuple[str, str]:
        """items: (label, text value, spoken value) that were wrong/missing."""
        t = "negative, " + ", ".join(f"{lab} {tv}" for lab, tv, _ in items) + ", read back"
        s = "negative, " + ", ".join(f"{lab} {sv}" for lab, _, sv in items) + ", read back"
        return self._out(t, s)

    # ground ----------------------------------------------------------------
    def push_start(self, facing: str | None = None) -> tuple[str, str]:
        f = f", facing {facing}" if facing else ""
        return self._out(f"push and start approved{f}", f"push and start approved{f}")

    def startup(self) -> tuple[str, str]:
        return self._out("start up approved", "start up approved")

    @staticmethod
    def _via(names: list[str]) -> tuple[str, str]:
        if not names:
            return "", ""
        return " via " + ", ".join(names), " via " + ", ".join(taxiway(n) for n in names)

    def taxi(self, route, runway_end: str, qnh_hpa: float | None = None,
             hold_word: str | None = None) -> tuple[str, str]:
        """Taxi clearance for the first leg. hold_word = runway said in "hold short runway X"
        (the departure runway, or a runway the route crosses: the clearance limit)."""
        if route.ok and route.legs:
            leg = route.legs[0]
            hw = hold_word or runway_end
            vt, vs = self._via(leg.taxiways)
            it, is_ = (f", intersection {route.intersection}", f", intersection {taxiway(route.intersection)}") \
                if route.intersection and len(route.legs) == 1 else ("", "")
            if self.region == "FAA":
                t = f"runway {runway_end}{it}, taxi{vt}, hold short runway {hw}"
                s = f"runway {runway(runway_end)}{is_}, taxi{vs}, hold short runway {runway(hw)}"
            else:
                t = f"taxi to holding point runway {runway_end}{it}{vt}, hold short runway {hw}"
                s = f"taxi to holding point runway {runway(runway_end)}{is_}{vs}, hold short runway {runway(hw)}"
        else:
            t = f"taxi to runway {runway_end} at your discretion, hold short runway {runway_end}"
            s = f"taxi to runway {runway(runway_end)} at your discretion, hold short runway {runway(runway_end)}"
        if qnh_hpa:
            qt, qs = self.qnh(qnh_hpa)
            t += f", {qt}"
            s += f", {qs}"
        return self._out(t, s)

    def taxi_continue(self, leg, cross_word: str, hold_word: str | None, gate: str | None = None,
                      intersection: str | None = None) -> tuple[str, str]:
        """Next leg after a hold short: "cross runway 15, continue via A, hold short runway 19"."""
        vt, vs = self._via(leg.taxiways)
        t = f"cross runway {cross_word}"
        s = f"cross runway {runway(cross_word)}"
        if vt:
            t += f", continue{vt}"
            s += f", continue{vs}"
        if hold_word:
            if intersection:
                t += f", intersection {intersection}"
                s += f", intersection {taxiway(intersection)}"
            t += f", hold short runway {hold_word}"
            s += f", hold short runway {runway(hold_word)}"
        elif gate:
            t += f" to stand {gate}"
            s += f" to stand {gate}"
        return self._out(t, s)

    def taxi_to_gate(self, route, gate: str, hold_word: str | None = None) -> tuple[str, str]:
        if route.ok and route.legs:
            vt, vs = self._via(route.legs[0].taxiways)
            t, s = f"taxi to stand {gate}{vt}", f"taxi to stand {gate}{vs}"
            if hold_word:
                t += f", hold short runway {hold_word}"
                s += f", hold short runway {runway(hold_word)}"
            return self._out(t, s)
        return self._out(f"taxi to stand {gate} at your discretion",
                         f"taxi to stand {gate} at your discretion")

    def contact(self, pos: str, khz: int, extra: str = "") -> tuple[str, str]:
        e = f", {extra}" if extra else ""
        return self._out(f"contact {pos} {freq_text(khz)}{e}", f"contact {pos} {freq(khz)}{e}")

    def monitor(self, pos: str, khz: int) -> tuple[str, str]:
        return self._out(f"monitor {pos} {freq_text(khz)}", f"monitor {pos} {freq(khz)}")

    # tower -----------------------------------------------------------------
    def line_up(self, runway_end: str) -> tuple[str, str]:
        return self._out(f"runway {runway_end}, line up and wait",
                         f"runway {runway(runway_end)}, line up and wait")

    def cleared_takeoff(self, runway_end: str, w: tuple[str, str]) -> tuple[str, str]:
        return self._out(f"{w[0]}, runway {runway_end}, cleared for takeoff",
                         f"{w[1]}, runway {runway(runway_end)}, cleared for takeoff")

    def cleared_land(self, runway_end: str, w: tuple[str, str]) -> tuple[str, str]:
        return self._out(f"{w[0]}, runway {runway_end}, cleared to land",
                         f"{w[1]}, runway {runway(runway_end)}, cleared to land")

    def continue_approach(self, runway_end: str) -> tuple[str, str]:
        return self._out(f"continue approach runway {runway_end}",
                         f"continue approach runway {runway(runway_end)}")

    def vacate(self, pos: str, khz: int) -> tuple[str, str]:
        return self._out(f"vacate when able, contact {pos} {freq_text(khz)}",
                         f"vacate when able, contact {pos} {freq(khz)}")

    # radar -----------------------------------------------------------------
    def radar_contact(self, alt_ft: int, verb: str = "climb") -> tuple[str, str]:
        at, asp = self.alt(alt_ft)
        return self._out(f"radar contact, {verb} {at}", f"radar contact, {verb} {asp}")

    def climb(self, alt_ft: int, verb: str = "climb") -> tuple[str, str]:
        at, asp = self.alt(alt_ft)
        return self._out(f"{verb} {at}", f"{verb} {asp}")

    def cleared_approach(self, appr: str, runway_end: str, alt_ft: int, tower_khz: int | None) -> tuple[str, str]:
        at, asp = self.alt(alt_ft)
        t = f"descend {at}, cleared {appr} approach runway {runway_end}"
        s = f"descend {asp}, cleared {appr} approach runway {runway(runway_end)}"
        if tower_khz:
            t += f", contact tower {freq_text(tower_khz)} when established"
            s += f", contact tower {freq(tower_khz)} when established"
        return self._out(t, s)

    # generic ---------------------------------------------------------------
    def radio_check(self, station: str) -> tuple[str, str]:
        return self._out(f"{station}, read you five", f"{station}, read you five")

    def go_ahead(self, station: str) -> tuple[str, str]:
        return self._out(f"{station}, go ahead", f"{station}, go ahead")

    def say_again(self) -> tuple[str, str]:
        return self._out("say again", "say again")

    def standby(self) -> tuple[str, str]:
        return self._out("standby", "standby")

    def roger(self) -> tuple[str, str]:
        return self._out("roger", "roger")

    def wrong_frequency(self, pos: str, khz: int) -> tuple[str, str]:
        return self._out(f"you are on the wrong frequency, contact {pos} {freq_text(khz)}",
                         f"you are on the wrong frequency, contact {pos} {freq(khz)}")

    def custom(self, text: str, spoken: str | None = None) -> tuple[str, str]:
        return self._out(text, spoken or text)

    def how_do_you_read(self, repeat: tuple[str, str]) -> tuple[str, str]:
        t, s = repeat
        return f"{self.cs}, how do you read? I say again, {t.split(', ', 1)[-1]}", \
               f"{self.cs_spoken}, how do you read? I say again, {s.split(', ', 1)[-1]}"

    def stop_hold_position(self, runway_end: str) -> tuple[str, str]:
        return self._out(f"STOP immediately, hold position, you are not cleared onto runway {runway_end}",
                         f"stop immediately, hold position, you are not cleared onto runway {runway(runway_end)}")

    def missed_turn(self) -> tuple[str, str]:
        return self._out("it appears you missed your turn, hold position and advise",
                         "it appears you missed your turn, hold position and advise")

    def check_altitude(self, alt_ft: int) -> tuple[str, str]:
        at, asp = self.alt(alt_ft)
        return self._out(f"check altitude, maintain {at}", f"check altitude, maintain {asp}")
