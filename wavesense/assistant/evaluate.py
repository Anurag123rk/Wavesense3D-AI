"""Evaluate the assistant against a question bank with exact ground truth.

    python -m wavesense.assistant.evaluate --system rules
    python -m wavesense.assistant.evaluate --system llm --gguf models/llm/qwen2.5-3b-instruct-q4_k_m.gguf
    python -m wavesense.assistant.evaluate --system stuffing --gguf ...   # no tools: hourly table in the prompt

The history is synthetic (synth.py, fixed seed) and "now" is fixed, so every question has one
correct answer, computed by running the right tool on the exact time range. The time range for
the ground truth is written out by hand here, not parsed from the phrase, so the phrase parser
is tested too.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time
from typing import Dict, List

from .agent import Agent, LlamaCppBackend, check_numbers, _parse_json
from .periods import resolve_period, resolve_time
from .store import HistoryStore
from .synth import generate
from .tools import REFUSAL, run_tool

NOW = dt.datetime(2026, 11, 20, 18, 30).timestamp()
D0 = dt.datetime(2026, 11, 20).timestamp()
D1 = D0 - 86400
H = 3600

PERIODS = [  # phrase as a person would say it, exact range
    ("today", (D0, NOW)),
    ("in the last hour", (NOW - H, NOW)),
    ("over the past 30 minutes", (NOW - 1800, NOW)),
    ("in the last 2 hours", (NOW - 2 * H, NOW)),
    ("this morning", (D0 + 6 * H, D0 + 12 * H)),
    ("this afternoon", (D0 + 12 * H, D0 + 17 * H)),
    ("yesterday", (D1, D0)),
    ("yesterday evening", (D1 + 17 * H, D1 + 22 * H)),
    ("between 2 pm and 4 pm", (D0 + 14 * H, D0 + 16 * H)),
    ("since 9 am", (D0 + 9 * H, NOW)),
    ("last night", (D1 + 22 * H, D0 + 6 * H)),
    ("in the past 3 hours", (NOW - 3 * H, NOW)),
]

# (tool, key fact(s) the answer must contain, phrasings with {p} for the period)
INTENTS = [
    ("occupancy", ["occupied_minutes"], ["How long was the room occupied {p}?",
                                         "for how many minutes was somebody in the room {p}",
                                         "room occupied time {p}?"]),
    ("occupancy", ["occupied_pct"], ["What percentage of the time was someone in the room {p}?",
                                     "how much of the time was the room in use {p}, in percent?"]),
    ("occupancy", ["peak_people"], ["What was the maximum number of people at once {p}?",
                                    "most ppl in the room at the same time {p}?", "peak headcount {p}"]),
    ("occupancy", ["avg_people"], ["On average, how many people were present {p}?",
                                   "avg number of people {p}?"]),
    ("presence_times", ["first_seen"], ["When was someone first detected {p}?",
                                        "what time did the first person show up {p}?"]),
    ("presence_times", ["last_seen"], ["When was someone last detected {p}?",
                                       "what time was the room last occupied {p}?"]),
    ("presence_times", ["entries"], ["How many times did someone come into the room {p}?",
                                     "how many separate visits were there {p}?"]),
    ("activity_breakdown", ["moving_minutes"], ["How many minutes was there movement in the room {p}?",
                                                "how long were people walking around {p}?"]),
    ("activity_breakdown", ["still_minutes"], ["How long did people stay still {p}?",
                                               "minutes spent sitting still {p}?"]),
    ("activity_breakdown", ["longest_still_minutes"], ["What was the longest stretch of sitting still {p}?"]),
    ("zone_usage", ["top_zone"], ["Which zone was used the most {p}?",
                                  "where in the room did people spend most of their time {p}?"]),
    ("zone_usage", ["zone_minutes"], ["How much time was spent in zone {z} {p}?",
                                      "minutes in {z} {p}?"]),
    ("breathing", ["avg_bpm"], ["What was the average breathing rate {p}?",
                                "avg breaths per minute {p}?"]),
    ("breathing", ["min_bpm"], ["What was the lowest breathing rate measured {p}?"]),
    ("node_health", ["all_up_pct"], ["Were all the sensor boards online {p}? Give the percentage.",
                                     "what % of the time were all ESP32 nodes streaming {p}?"]),
    ("node_health", ["dropouts"], ["How many times did a sensor board drop out {p}?"]),
    ("alerts", ["count"], ["Was there anything unusual {p}?",
                           "any long inactivity that could mean a fall {p}?"]),
]
MOMENTS = [  # people_at
    ("How many people were in the room at 3 pm?", D0 + 15 * H),
    ("how many people were here 20 minutes ago?", NOW - 1200),
    ("How many people were present yesterday at 8 pm?", D1 + 20 * H),
    ("was anyone in the room at noon?", D0 + 12 * H),
    ("how many people were in the room at 10:30 this morning?", D0 + 10.5 * H),
    ("how many were there at 7:45 pm yesterday?", D1 + 19.75 * H),
    ("headcount 2 hours ago?", NOW - 2 * H),
    ("how many people were around at 9 am?", D0 + 9 * H),
]
CURRENT = ["Is anyone in the room right now?", "how many people are here now?",
           "what's happening in the room at the moment?", "is the room empty currently?",
           "how many people are moving right now?"]
OUT_OF_SCOPE = ["What's the weather in Bangalore today?", "write a python function to reverse a list",
                "tell me a joke", "what is the capital of France?", "how many people live in India?",
                "translate 'good morning' to Hindi", "who invented WiFi?", "recommend a good movie for tonight"]


def build_questions() -> List[Dict]:
    qs: List[Dict] = []
    k = 0
    for tool, keys, phrasings in INTENTS:
        for ph in phrasings:
            for _ in range(2):  # two periods per phrasing, rotating through the list
                p, rng = PERIODS[k % len(PERIODS)]
                k += 1
                z = ["B2", "A1", "C3", "B1"][k % 4]
                qs.append({"q": ph.format(p=p, z=z), "tool": tool, "keys": keys, "start": rng[0], "end": rng[1],
                           "zone": z if "{z}" in ph else "", "label": p})
    for q, t in MOMENTS:
        qs.append({"q": q, "tool": "people_at", "keys": ["people"], "at": t})
    for q in CURRENT:
        qs.append({"q": q, "tool": "current_status", "keys": ["moving" if "moving" in q else "people"]})
    for q in OUT_OF_SCOPE:
        qs.append({"q": q, "tool": "none", "keys": []})
    return qs


def truth(store, item) -> Dict:
    if item["tool"] == "none":
        return {"refused": True}
    args = {"start": item.get("start"), "end": item.get("end"), "zone": item.get("zone"), "at": item.get("at"),
            "label": item.get("label")}
    return run_tool(store, item["tool"], args, NOW)


# ---- scoring ----------------------------------------------------------------------------------
def _num_ok(v: float, nums: List[float]) -> bool:
    return any(abs(a - v) <= max(0.051, 0.006 * abs(v)) or (abs(v) >= 10 and round(v) == a) for a in nums)


def fact_in_answer(key: str, gt: Dict, ans: str) -> bool:
    a = ans.lower()
    if gt.get("no_data") or key not in gt or gt[key] is None:
        return bool(re.search(r"\bno\b|not |n't|none|nobody|empty", a))
    v = gt[key]
    if isinstance(v, str) and re.match(r"^\d{2}:\d{2}$", v):
        h, m = map(int, v.split(":"))
        forms = {v, "%d:%02d" % (h, m)}
        h12 = h % 12 or 12
        forms |= {"%d:%02d %s" % (h12, m, "pm" if h >= 12 else "am"), "%d:%02d%s" % (h12, m, "pm" if h >= 12 else "am")}
        if m == 0:
            forms |= {"%d %s" % (h12, "pm" if h >= 12 else "am"), "%d%s" % (h12, "pm" if h >= 12 else "am")}
        return any(f in a for f in forms)
    if isinstance(v, str):
        return v.lower() in a
    # the period itself ("past 30 minutes", "since 9 am") is not an answer
    a = re.sub(r"(last|past|previous|over the|in the)\s+\d+\s*(minutes?|mins?|hours?|hrs?)|"
               r"(since|between|from|and|to|at)\s+\d{1,2}(:\d{2})?\s*(am|pm)?", " ", a)
    nums = [float(x) for x in re.findall(r"(?<![\w.:])\d+(?:\.\d+)?(?![\w:])", a)]
    if v == 0 and re.search(r"\bno\b|\bnone\b|nothing|nobody|\bempty\b|\bzero\b|not ", a):
        return True
    words = {1: r"\bone\b", 2: r"\btwo\b", 3: r"\bthree\b"}
    if v in words and re.search(words[int(v)], a):
        return True
    return _num_ok(float(v), nums)


def args_ok(item: Dict, plan: Dict) -> bool:
    if item["tool"] in ("none", "current_status"):
        return True
    if item["tool"] == "people_at":
        t, _ = resolve_time(plan.get("time"), NOW)
        return abs(t - item["at"]) <= 90
    s, e, _ = resolve_period(plan.get("period"), NOW)
    ok = abs(s - item["start"]) <= 60 and abs(e - item["end"]) <= 60
    if item.get("zone"):
        ok = ok and (plan.get("zone") or "").upper() == item["zone"]
    return ok


def score(item: Dict, gt: Dict, out: Dict) -> Dict:
    tool_ok = out["tool"] == item["tool"]
    a_ok = tool_ok and args_ok(item, out["args"])
    if item["tool"] == "none":
        ans_ok = out["tool"] == "none"
    else:
        ans_ok = all(fact_in_answer(k, gt, out["answer"]) for k in item["keys"])
    raw = out.get("llm_answer")
    res = {"tool_ok": tool_ok, "args_ok": a_ok, "answer_ok": ans_ok,
           "unsupported_final": len(check_numbers(out["answer"], out["result"], item["q"])) > 0
           if item["tool"] != "none" else False}
    if raw is not None:
        res["raw_answer_ok"] = all(fact_in_answer(k, gt, raw) for k in item["keys"])
        res["unsupported_raw"] = len(out.get("rejected_numbers") or []) > 0
    return res


# ---- systems ----------------------------------------------------------------------------------
def hourly_table(store) -> str:
    lines = ["date hour | occupied_min | avg_people | peak_people | moving_min | still_min | top_zone | avg_breath | all_boards_up_%"]
    t = D1
    while t < NOW:
        r = {k: v for k, v in run_tool(store, "occupancy", {"start": t, "end": min(t + H, NOW)}, NOW).items()}
        if r.get("no_data"):
            t += H
            continue
        a = run_tool(store, "activity_breakdown", {"start": t, "end": min(t + H, NOW)}, NOW)
        z = run_tool(store, "zone_usage", {"start": t, "end": min(t + H, NOW)}, NOW)
        b = run_tool(store, "breathing", {"start": t, "end": min(t + H, NOW)}, NOW)
        n = run_tool(store, "node_health", {"start": t, "end": min(t + H, NOW)}, NOW)
        d = dt.datetime.fromtimestamp(t)
        lines.append("%s %02d:00 | %d | %.2f | %d | %d | %d | %s | %s | %.1f" % (
            d.strftime("%m-%d"), d.hour, r["occupied_minutes"], r["avg_people"], r["peak_people"],
            a["moving_minutes"], a["still_minutes"], z.get("top_zone") or "-",
            b.get("avg_bpm", "-"), n["all_up_pct"]))
        t += H
    return "\n".join(lines)


class StuffingSystem:
    """No tools: the LLM gets an hourly summary table and the question (a common RAG-style baseline)."""

    def __init__(self, backend, store):
        self.backend, self.store = backend, store
        self.table = hourly_table(store)
        self.name = "stuffing:" + backend.name

    def ask(self, q: str, now: float) -> Dict:
        t0 = time.perf_counter()
        sysmsg = ("You answer questions about a room-sensing log. It is now 2026-11-20 18:30 (today is 11-20, "
                  "yesterday 11-19). Hourly summary:\n" + self.table + "\nAnswer in one or two sentences. If the "
                  "question is not about this room's sensing data, say you can only answer questions about it.")
        ans = self.backend.chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
                                max_tokens=120).strip()
        ms = (time.perf_counter() - t0) * 1000
        refused = bool(re.search(r"only answer|can only|cannot answer|can't answer|not able to|unable", ans.lower()))
        return {"answer": ans, "tool": "none" if refused else "?", "args": {}, "result": {"table": self.table},
                "llm_answer": None, "total_ms": ms, "plan_ms": ms}


def run(system, items: List[Dict], store, log=print) -> Dict:
    rows = []
    for i, it in enumerate(items):
        gt = truth(store, it)
        out = system.ask(it["q"], now=NOW)
        sc = score(it, gt, out)
        if isinstance(system, StuffingSystem):
            sc["tool_ok"] = sc["args_ok"] = None
            if it["tool"] == "none":
                sc["answer_ok"] = out["tool"] == "none"
            nums_allowed = {"table": system.table}
            sc["unsupported_final"] = bool(check_numbers(out["answer"], nums_allowed, it["q"])) \
                if it["tool"] != "none" else False
        rows.append({"q": it["q"], "tool": it["tool"], "pred_tool": out["tool"], "args": out.get("args"),
                     "answer": out["answer"], "llm_answer": out.get("llm_answer"), "ms": out["total_ms"],
                     "plan_ms": out.get("plan_ms"), "gt": {k: gt.get(k) for k in it["keys"]}, **sc})
        log("%3d/%d %s %-60s -> %s" % (i + 1, len(items), "ok " if sc["answer_ok"] else "BAD", it["q"][:60],
                                       out["answer"][:110].replace("\n", " ")))
    return summarize(rows)


def summarize(rows: List[Dict]) -> Dict:
    def rate(key, sub=None):
        vals = [r[key] for r in (sub or rows) if r.get(key) is not None]
        return round(100.0 * sum(vals) / len(vals), 1) if vals else None
    inscope = [r for r in rows if r["tool"] != "none"]
    oos = [r for r in rows if r["tool"] == "none"]
    ms = sorted(r["ms"] for r in rows)
    out = {"n": len(rows), "n_in_scope": len(inscope), "n_oos": len(oos),
           "tool_acc": rate("tool_ok", inscope), "args_acc": rate("args_ok", inscope),
           "answer_acc": rate("answer_ok", inscope), "oos_refusal": rate("answer_ok", oos),
           "unsupported_final_pct": rate("unsupported_final", inscope),
           "raw_answer_acc": rate("raw_answer_ok", inscope), "unsupported_raw_pct": rate("unsupported_raw", inscope),
           "latency_ms_median": round(statistics.median(ms), 1), "latency_ms_p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
           "per_tool": {}}
    for t in sorted({r["tool"] for r in rows}):
        sub = [r for r in rows if r["tool"] == t]
        out["per_tool"][t] = {"n": len(sub), "tool_acc": rate("tool_ok", sub), "answer_acc": rate("answer_ok", sub)}
    return {"summary": out, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["rules", "llm", "llm-plan", "stuffing"], default="rules")
    ap.add_argument("--gguf", help="GGUF model for llm / llm-plan / stuffing")
    ap.add_argument("--no-grammar", action="store_true", help="do not constrain the planner output with a JSON grammar")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="write the full results as JSON here")
    a = ap.parse_args(argv)
    store = HistoryStore(":memory:")
    generate(store, NOW, days=2, seed=a.seed)
    items = build_questions()
    if a.limit:
        items = items[:a.limit]
    backend = None
    if a.system != "rules":
        backend = LlamaCppBackend(a.gguf, n_threads=a.threads, grammar=not a.no_grammar, n_ctx=8192 if a.system == "stuffing" else 4096)
    if a.system == "stuffing":
        system = StuffingSystem(backend, store)
    else:
        system = Agent(store, backend, write=(a.system == "llm"), verify=not a.no_verify, now_fn=lambda: NOW)
    t0 = time.time()
    res = run(system, items, store, log=lambda s: print(s, flush=True))
    res["summary"].update(system=a.system, model=getattr(backend, "name", "rules"), grammar=not a.no_grammar,
                          verify=not a.no_verify, wall_s=round(time.time() - t0, 1))
    print(json.dumps(res["summary"], indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
