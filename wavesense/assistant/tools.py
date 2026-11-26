"""Query tools over the sensing history.

Every tool takes (store, args, now) and returns a dict of exact values plus an `answer` written
from a template. The language model only picks a tool and copies the time phrase; the numbers
always come from here.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, Dict, List

import numpy as np

from .periods import hm, resolve_period, resolve_time
from .store import HistoryStore

ToolFn = Callable[[HistoryStore, Dict, float], Dict]


def _period(args: Dict, now: float):
    if args.get("start") is not None and args.get("end") is not None:  # exact range (evaluation, API)
        return float(args["start"]), float(args["end"]), args.get("label") or "the selected period"
    return resolve_period(args.get("period"), now)


def _min(sec: float) -> int:
    return int(round(sec / 60.0))


def _runs(mask: np.ndarray):
    """(start index, end index exclusive) of each run of True."""
    if not len(mask):
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _no_data(label: str) -> Dict:
    return {"no_data": True, "period": label,
            "answer": "I have no sensor data for %s, so I can't tell." % label}


def _clock_span(s: float, e: float, label: str) -> Dict:
    return {"period": label, "from": hm(s), "to": hm(e)}


# ---- tools ------------------------------------------------------------------------------------
def current_status(store: HistoryStore, args: Dict, now: float) -> Dict:
    r = store.latest()
    if r is None or now - r["t"] > 60:
        return {"no_data": True, "answer": "The sensors are not reporting right now."}
    d = store.load(now - 3600, now + 1)
    same = (d["total"] == r["total"]) & (d["activity"] == r["activity"])
    since = r["t"]
    if len(same):
        idx = np.where(~same)[0]
        since = d["t"][idx[-1] + 1] if len(idx) and idx[-1] + 1 < len(same) else d["t"][0]
    out = {"people": int(r["total"]), "moving": int(r["moving"]), "still": int(r["still"]),
           "activity": r["activity"], "zone": r["zone"], "breathing_bpm": None if r["breath"] is None
           else round(float(r["breath"]), 1), "for_minutes": _min(r["t"] - since), "since": hm(since)}
    if not r["presence"]:
        out["answer"] = "The room is empty right now (for the last %d minutes)." % out["for_minutes"]
    else:
        txt = "%d %s in the room now: %d moving and %d still" % (
            out["people"], "person is" if out["people"] == 1 else "people are", out["moving"], out["still"])
        if out["zone"]:
            txt += ", main person in zone %s" % out["zone"]
        if out["breathing_bpm"]:
            txt += ", breathing about %.1f breaths per minute" % out["breathing_bpm"]
        out["answer"] = txt + "."
    return out


def occupancy(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    dt, tot = d["dt"], d["total"]
    data = float(dt.sum())
    occ = float(dt[d["presence"] > 0].sum())
    peak = int(np.nanmax(tot))
    i = int(np.argmax(tot >= peak))
    out = dict(_clock_span(s, e, label), occupied_minutes=_min(occ), occupied_pct=round(100 * occ / data, 1),
               avg_people=round(float((tot * dt).sum() / data), 2),
               avg_people_when_occupied=round(float((tot * dt)[d["presence"] > 0].sum() / occ), 2) if occ else 0.0,
               peak_people=peak, peak_time=hm(d["t"][i]), data_minutes=_min(data))
    out["answer"] = ("For %s the room was occupied for %d minutes (%.1f%% of the time). On average %.2f people "
                     "were present, and the most at once was %d at %s." % (
                         label, out["occupied_minutes"], out["occupied_pct"], out["avg_people"], peak,
                         out["peak_time"]))
    return out


def people_at(store: HistoryStore, args: Dict, now: float) -> Dict:
    if args.get("at") is not None:
        t = float(args["at"])
    else:
        t, _label = resolve_time(args.get("time"), now)
    if t > now + 30:
        return {"no_data": True, "time": hm(t), "answer": "%s is still in the future." % hm(t)}
    r = store.nearest(t)
    if r is None:
        return {"no_data": True, "time": hm(t), "answer": "I have no sensor data around %s." % hm(t)}
    out = {"time": hm(r["t"]), "people": int(r["total"]), "moving": int(r["moving"]), "still": int(r["still"]),
           "activity": r["activity"], "zone": r["zone"]}
    if not r["presence"]:
        out["answer"] = "At %s the room was empty." % out["time"]
    else:
        out["answer"] = "At %s there %s %d %s in the room (%d moving, %d still)." % (
            out["time"], "was" if out["people"] == 1 else "were", out["people"],
            "person" if out["people"] == 1 else "people", out["moving"], out["still"])
    return out


def presence_times(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    pres = d["presence"] > 0
    out = dict(_clock_span(s, e, label))
    if not pres.any():
        out.update(entries=0, first_seen=None, last_seen=None, longest_empty_minutes=_min(float(d["dt"].sum())))
        out["answer"] = "Nobody was detected %s." % ("today" if label == "today" else "during " + label)
        return out
    runs = _runs(pres)
    # merge occupied runs separated by less than a minute of "empty" (detector flicker)
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if d["t"][a] - d["t"][merged[-1][1] - 1] < 60:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    gaps = [d["t"][a] - d["t"][pb - 1] for (_, pb), (a, _) in zip(merged[:-1], merged[1:])]
    first, last = d["t"][merged[0][0]], d["t"][merged[-1][1] - 1]
    out.update(entries=len(merged), first_seen=hm(first), last_seen=hm(last),
               present_at_end=bool(pres[-1]) and e >= now - 5,
               longest_empty_minutes=_min(max(gaps)) if gaps else 0)
    out["answer"] = ("During %s someone was first detected at %s and last at %s%s. The room became occupied %d "
                     "time%s; the longest empty gap in between was %d minutes." % (
                         label, out["first_seen"], out["last_seen"], " (still there)" if out["present_at_end"] else "",
                         out["entries"], "" if out["entries"] == 1 else "s", out["longest_empty_minutes"]))
    return out


def zone_usage(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    pres = d["presence"] > 0
    mins: Dict[str, float] = {}
    for z, dt, p in zip(d["zone"], d["dt"], pres):
        if p and z:
            mins[z] = mins.get(z, 0.0) + dt
    out = dict(_clock_span(s, e, label))
    if not mins:
        out.update(zones={}, top_zone=None)
        out["answer"] = "Nobody was located in any zone during %s." % label
        return out
    zs = {k: _min(v) for k, v in sorted(mins.items(), key=lambda kv: -kv[1])}
    occ = sum(mins.values())
    out.update(zones=zs, top_zone=next(iter(zs)), top_zone_minutes=next(iter(zs.values())))
    zone = (args.get("zone") or "").strip().upper() or None
    if zone:
        out.update(zone=zone, zone_minutes=zs.get(zone, 0), zone_pct=round(100 * mins.get(zone, 0.0) / occ, 1))
        out["answer"] = "During %s the main person spent %d minutes in zone %s (%.1f%% of the occupied time)." % (
            label, out["zone_minutes"], zone, out["zone_pct"])
    else:
        top3 = ", ".join("%s %d min" % kv for kv in list(zs.items())[:3])
        out["answer"] = "During %s the most used zone was %s (%d minutes). Top zones: %s." % (
            label, out["top_zone"], out["top_zone_minutes"], top3)
    return out


def breathing(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    b = d["breath"]
    ok = np.isfinite(b)
    out = dict(_clock_span(s, e, label))
    if not ok.any():
        out.update(measured_minutes=0)
        out["answer"] = ("No breathing rate was measured during %s. It is only measured while exactly one "
                         "person is still." % label)
        return out
    out.update(avg_bpm=round(float(np.average(b[ok], weights=d["dt"][ok])), 1), min_bpm=round(float(b[ok].min()), 1),
               max_bpm=round(float(b[ok].max()), 1), measured_minutes=_min(float(d["dt"][ok].sum())))
    out["answer"] = ("During %s the breathing rate averaged %.1f breaths per minute (range %.1f to %.1f), "
                     "measured over %d minutes of stillness." % (label, out["avg_bpm"], out["min_bpm"],
                                                                 out["max_bpm"], out["measured_minutes"]))
    return out


def activity_breakdown(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    out = dict(_clock_span(s, e, label))
    for a in ("empty", "still", "moving"):
        out["%s_minutes" % a] = _min(float(d["dt"][d["activity"] == a].sum()))
    still = d["activity"] == "still"
    best, best_s = 0.0, None
    for a, b in _runs(still):
        dur = d["t"][b - 1] - d["t"][a] + d["dt"][b - 1]
        if dur > best:
            best, best_s = dur, d["t"][a]
    out.update(longest_still_minutes=_min(best), longest_still_start=hm(best_s) if best_s else None)
    out["answer"] = ("During %s: %d minutes moving, %d minutes still and %d minutes empty. The longest still "
                     "stretch was %d minutes%s." % (label, out["moving_minutes"], out["still_minutes"],
                                                    out["empty_minutes"], out["longest_still_minutes"],
                                                    " starting at %s" % out["longest_still_start"]
                                                    if best_s else ""))
    return out


def node_health(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    n = int(np.nanmax(d["nodes_n"])) if len(d["nodes_n"]) else 0
    full = d["nodes_up"] >= d["nodes_n"]
    data = float(d["dt"].sum())
    outages = _runs(~full)
    longest = max([d["t"][b - 1] - d["t"][a] + d["dt"][b - 1] for a, b in outages], default=0.0)
    out = dict(_clock_span(s, e, label), nodes=n, all_up_pct=round(100 * float(d["dt"][full].sum()) / data, 1),
               dropouts=len(outages), longest_outage_minutes=round(longest / 60.0, 1))
    out["answer"] = ("During %s all %d boards were streaming %.1f%% of the time. There were %d dropout%s; the "
                     "longest lasted %.1f minutes." % (label, n, out["all_up_pct"], out["dropouts"],
                                                       "" if out["dropouts"] == 1 else "s",
                                                       out["longest_outage_minutes"]))
    return out


def alerts(store: HistoryStore, args: Dict, now: float) -> Dict:
    s, e, label = _period(args, now)
    d = store.load(s, e)
    if not len(d["t"]):
        return _no_data(label)
    limit = float(args.get("min_still_minutes") or 20) * 60
    items: List[Dict] = []
    for a, b in _runs(d["activity"] == "still"):
        dur = d["t"][b - 1] - d["t"][a] + d["dt"][b - 1]
        if dur >= limit:
            items.append({"type": "long_inactivity", "start": hm(d["t"][a]), "minutes": _min(dur)})
    hours = np.array([_dt.datetime.fromtimestamp(t).hour for t in d["t"]])
    night = (hours < 5) & (d["presence"] > 0)
    for a, b in _runs(night):
        dur = d["t"][b - 1] - d["t"][a] + d["dt"][b - 1]
        if dur >= 120:
            items.append({"type": "night_activity", "start": hm(d["t"][a]), "minutes": _min(dur)})
    out = dict(_clock_span(s, e, label), count=len(items), alerts=items)
    if not items:
        out["answer"] = "Nothing unusual during %s: no still period over %d minutes and no night-time activity." % (
            label, int(limit / 60))
    else:
        parts = ["%s at %s (%d min)" % (i["type"].replace("_", " "), i["start"], i["minutes"]) for i in items[:5]]
        out["answer"] = "%d alert%s during %s: %s." % (len(items), "" if len(items) == 1 else "s", label,
                                                     "; ".join(parts))
    return out


TOOLS: Dict[str, Dict] = {
    "current_status": {"fn": current_status, "args": [],
                       "desc": "who is in the room right now: people count, moving/still, zone, breathing"},
    "occupancy": {"fn": occupancy, "args": ["period"],
                  "desc": "how long anyone was in the room (occupied minutes), occupancy %, average and peak number of people"},
    "people_at": {"fn": people_at, "args": ["time"],
                  "desc": "how many people were present at one moment in the past"},
    "presence_times": {"fn": presence_times, "args": ["period"],
                       "desc": "first arrival, last departure, how many times someone came in, empty gaps"},
    "zone_usage": {"fn": zone_usage, "args": ["period", "zone"],
                   "desc": "which zones (A1..C3) were used and for how long; time spent in one zone"},
    "breathing": {"fn": breathing, "args": ["period"],
                  "desc": "average / min / max breathing rate"},
    "activity_breakdown": {"fn": activity_breakdown, "args": ["period"],
                           "desc": "minutes with movement vs minutes still vs empty, longest time sitting still"},
    "node_health": {"fn": node_health, "args": ["period"],
                    "desc": "whether the ESP32 sensor boards were online, dropouts"},
    "alerts": {"fn": alerts, "args": ["period"],
               "desc": "unusual events: very long inactivity (possible fall), activity at night"},
}
REFUSAL = ("I can only answer questions about what the WaveSense sensors recorded in this room "
           "(occupancy, people counts, movement, zones, breathing and sensor health).")


def run_tool(store: HistoryStore, name: str, args: Dict, now: float) -> Dict:
    if name not in TOOLS:
        return {"refused": True, "answer": REFUSAL}
    return TOOLS[name]["fn"](store, args or {}, now)
