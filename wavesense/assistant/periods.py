"""Turn everyday time phrases ("this morning", "last 2 hours", "between 2 pm and 4 pm",
"yesterday at 9:30") into exact time ranges. Used by both the LLM planner (which only has to
copy the phrase) and the rule-based planner, so dates are never computed by the language model.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Optional, Tuple

WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
            "forty": 40, "forty-five": 45, "sixty": 60, "few": 3, "couple of": 2, "half an": 0.5, "half a": 0.5}
UNIT_S = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}
# named parts of a day: (start hour, end hour); end may be > 24 for "night"
PARTS = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (22, 30)}

NUM = r"(\d+(?:\.\d+)?|half an|half a|couple of|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|" \
      r"fifteen|twenty|thirty|forty-five|forty|sixty|few)"
CLOCK = r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?|(noon|midday|midnight)"


def _num(s: str) -> float:
    s = s.strip().lower()
    return float(s) if re.match(r"^\d", s) else float(WORD_NUM.get(s, 1))


def _day_start(now: float, days_ago: int = 0) -> float:
    d = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    return (d - dt.timedelta(days=days_ago)).timestamp()


def _clock(m, prefer_pm: bool = False) -> Optional[float]:
    """Hours after midnight for a CLOCK regex match."""
    if m.group(4):
        return {"noon": 12.0, "midday": 12.0, "midnight": 0.0}[m.group(4).lower()]
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "").lower()
    if h > 23 or mi > 59:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and prefer_pm and 1 <= h <= 7:
        h += 12  # "between 2 and 4" in an occupancy question almost always means afternoon
    return h + mi / 60.0


def _which_day(text: str) -> int:
    t = text.lower()
    if "day before yesterday" in t:
        return 2
    if "yesterday" in t or "last night" in t:
        return 1
    return 0


def resolve_period(phrase: Optional[str], now: float) -> Tuple[float, float, str]:
    """(start, end, label). Unknown or empty phrases mean "today so far"."""
    p = (phrase or "").strip().lower()
    p = re.sub(r"\s+", " ", p)
    if not p or p in ("today", "today so far", "so far today", "this day"):
        return _day_start(now), now, "today"
    if p in ("all", "ever", "all time", "everything", "so far", "overall", "since the start", "entire history"):
        return 0.0, now, "all recorded time"
    if p in ("now", "right now", "currently", "current", "at the moment"):
        return now - 10, now, "now"

    # last / past N units
    m = re.search(r"(?:last|past|previous|recent|within the last|in the last)\s+" + NUM + r"?\s*(second|sec|minute|min|hour|hr|day)s?\b", p)
    if m:
        n = _num(m.group(1)) if m.group(1) else 1.0
        sec = n * UNIT_S[m.group(2)]
        if m.group(2) == "day" and n >= 1:
            return now - sec, now, "the last %g day%s" % (n, "" if n == 1 else "s")
        return now - sec, now, "the last %s" % _fmt_dur(sec)
    m = re.fullmatch(r"(?:over |in |for |during )?(?:the )?" + NUM + r"\s*(second|sec|minute|min|hour|hr|day)s?", p)
    if m:  # a bare "30 minutes" means the last 30 minutes
        sec = _num(m.group(1)) * UNIT_S[m.group(2)]
        return now - sec, now, "the last %s" % _fmt_dur(sec)
    if re.search(r"\b(last|past) (hour|minute)\b", p):
        sec = 3600 if "hour" in p else 60
        return now - sec, now, "the last " + _fmt_dur(sec)

    day = _which_day(p)
    base = _day_start(now, day)
    day_lbl = {0: "today", 1: "yesterday", 2: "the day before yesterday"}[day]

    # explicit clock range: between X and Y / from X to Y / X-Y
    m = re.search(r"(?:between|from)?\s*(" + CLOCK + r")\s*(?:and|to|till|until|-|–)\s*(" + CLOCK + r")", p)
    if m:
        m1 = re.match(CLOCK, m.group(1).strip())
        m2 = re.match(CLOCK, m.group(6).strip())
        if m1 and m2:
            pm = bool(m2.group(3) and "p" in m2.group(3).lower()) and not m1.group(3)
            h1 = _clock(m1, prefer_pm=True)
            h2 = _clock(m2, prefer_pm=True)
            if pm and h1 is not None and h1 < 12 and h2 is not None and h2 >= 12 and h1 + 12 < h2:
                h1 += 12
            if h1 is not None and h2 is not None:
                if h2 <= h1:
                    h2 += 24
                s, e = base + h1 * 3600, min(base + h2 * 3600, now if day == 0 else base + h2 * 3600)
                return s, max(s, e), "%s between %s and %s" % (day_lbl, _hm(h1), _hm(h2 % 24))
    # since X
    m = re.search(r"since\s+(" + CLOCK + r")", p)
    if m:
        h = _clock(re.match(CLOCK, m.group(1).strip()), prefer_pm=False)
        if h is not None:
            s = base + h * 3600
            if s > now:  # "since 9" said in the evening with no am/pm
                s -= 12 * 3600
            return s, now, "the time since %s" % _hm((s - base) / 3600.0)
    # named part of the day
    if "last night" in p or "tonight" in p:
        b = _day_start(now, 1) if "last night" in p else _day_start(now)
        s, e = b + 22 * 3600, b + 30 * 3600
        return s, min(e, now), "last night" if "last night" in p else "tonight"
    for name, (h1, h2) in PARTS.items():
        if name in p:
            s, e = base + h1 * 3600, base + h2 * 3600
            return s, min(e, now) if day == 0 else e, "%s %s" % ("this" if day == 0 else day_lbl, name)
    if day:
        return base, base + 86400, day_lbl
    if "this week" in p or "week" in p:
        return now - 7 * 86400, now, "the last 7 days"
    return _day_start(now), now, "today"


def resolve_time(phrase: Optional[str], now: float) -> Tuple[float, str]:
    """A single moment: "3 pm", "yesterday at 9:30", "10 minutes ago", "noon"."""
    p = (phrase or "").strip().lower()
    if not p or p in ("now", "right now", "currently"):
        return now, "now"
    m = re.search(NUM + r"\s*(second|sec|minute|min|hour|hr|day)s?\s+ago", p)
    if m:
        sec = _num(m.group(1)) * UNIT_S[m.group(2)]
        return now - sec, "%s ago" % _fmt_dur(sec)
    day = _which_day(p)
    base = _day_start(now, day)
    m = re.search(CLOCK, p)
    if m and (m.group(1) or m.group(4)):
        h = _clock(m)
        if h is not None:
            if not m.group(3) and not m.group(4) and h < 12:
                # no am/pm: take the most recent past occurrence ("at 3" in the evening = 15:00)
                if day == 0 and base + (h + 12) * 3600 <= now:
                    h += 12
                elif day > 0 and 1 <= h <= 7:
                    h += 12
            t = base + h * 3600
            return t, ("%s %s" % ("yesterday" if day == 1 else "", _hm(h))).strip()
    return now, "now"


def find_period(text: str) -> Optional[str]:
    """Pull a time-range phrase out of a whole question (used by the rule-based planner)."""
    t = text.lower()
    pats = [r"(?:in the |over the |during the |for the )?(?:last|past|previous)\s+(?:(?:" + NUM[1:-1] + r")\s*)?(?:second|sec|minute|min|hour|hr|day)s?",
            r"(?:yesterday|today|day before yesterday)?\s*(?:between|from)\s+(?:" + CLOCK + r")\s*(?:and|to|till|until|-)\s*(?:" + CLOCK + r")",
            r"since\s+(?:" + CLOCK + r")",
            r"(?:this|yesterday|today)\s+(?:morning|afternoon|evening|night)|last night|tonight",
            r"day before yesterday|yesterday|today|this week|all time|ever|so far"]
    for pat in pats:
        m = re.search(pat, t)
        if m:
            return m.group(0).strip()
    return None


def find_time(text: str) -> Optional[str]:
    t = text.lower()
    m = re.search(NUM + r"\s*(?:second|sec|minute|min|hour|hr)s?\s+ago", t)
    if m:
        return m.group(0)
    m = re.search(r"(?:yesterday\s+)?(?:at|around|by)\s+(?:" + CLOCK + r")(?:\s+yesterday)?", t)
    if m:
        return m.group(0)
    return None


def _hm(h: float) -> str:
    h = h % 24
    return "%02d:%02d" % (int(h), int(round((h - int(h)) * 60)) % 60)


def _fmt_dur(sec: float) -> str:
    if sec % 3600 == 0:
        n = int(sec // 3600)
        return "hour" if n == 1 else "%d hours" % n
    if sec >= 60 and sec % 60 == 0:
        n = int(sec // 60)
        return "minute" if n == 1 else "%d minutes" % n
    return "%d seconds" % sec


def hm(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M")
