import datetime as dt

import pytest

from wavesense.assistant import Agent, HistoryStore
from wavesense.assistant.agent import check_numbers, rule_plan
from wavesense.assistant.periods import resolve_period, resolve_time
from wavesense.assistant.synth import generate
from wavesense.assistant.tools import run_tool

NOW = dt.datetime(2026, 11, 20, 18, 30).timestamp()
D0 = dt.datetime(2026, 11, 20).timestamp()


@pytest.fixture(scope="module")
def store():
    s = HistoryStore(":memory:")
    generate(s, NOW, days=2, seed=0)
    return s


@pytest.mark.parametrize("phrase,start,end", [
    ("today", D0, NOW),
    ("last 2 hours", NOW - 7200, NOW),
    ("in the past 30 minutes", NOW - 1800, NOW),
    ("30 minutes", NOW - 1800, NOW),
    ("this morning", D0 + 6 * 3600, D0 + 12 * 3600),
    ("yesterday", D0 - 86400, D0),
    ("between 2 pm and 4 pm", D0 + 14 * 3600, D0 + 16 * 3600),
    ("between 2 and 4", D0 + 14 * 3600, D0 + 16 * 3600),
    ("since 9 am", D0 + 9 * 3600, NOW),
    ("last night", D0 - 2 * 3600, D0 + 6 * 3600),
])
def test_periods(phrase, start, end):
    s, e, _ = resolve_period(phrase, NOW)
    assert abs(s - start) < 1 and abs(e - end) < 1


@pytest.mark.parametrize("phrase,t", [
    ("at 3 pm", D0 + 15 * 3600), ("at 3", D0 + 15 * 3600), ("20 minutes ago", NOW - 1200),
    ("yesterday at 9:30", D0 - 86400 + 9.5 * 3600), ("at noon", D0 + 12 * 3600),
])
def test_times(phrase, t):
    assert abs(resolve_time(phrase, NOW)[0] - t) < 1


def test_tools_match_raw_data(store):
    d = store.load(D0, NOW)
    occ = run_tool(store, "occupancy", {"period": "today"}, NOW)
    assert occ["occupied_minutes"] == round(float(d["dt"][d["presence"] > 0].sum()) / 60)
    assert 1 <= occ["peak_people"] <= 3
    act = run_tool(store, "activity_breakdown", {"period": "today"}, NOW)
    assert abs(act["moving_minutes"] + act["still_minutes"] + act["empty_minutes"] - occ["data_minutes"]) <= 2
    assert run_tool(store, "node_health", {"period": "yesterday"}, NOW)["dropouts"] >= 1


def test_no_data_and_refusal(store):
    r = run_tool(store, "occupancy", {"start": NOW + 10, "end": NOW + 20}, NOW)
    assert r.get("no_data")
    a = Agent(store, now_fn=lambda: NOW)
    assert a.ask("tell me a joke")["tool"] == "none"


@pytest.mark.parametrize("q,tool", [
    ("is anyone in the room right now?", "current_status"),
    ("how long was the room occupied this morning?", "occupancy"),
    ("what was the average breathing rate today?", "breathing"),
    ("were the boards online yesterday?", "node_health"),
    ("which zone was used most in the last 2 hours?", "zone_usage"),
    ("how many people were here 20 minutes ago?", "people_at"),
])
def test_rule_planner(q, tool):
    assert rule_plan(q)["tool"] == tool


def test_number_check():
    res = {"occupied_minutes": 74, "occupied_pct": 61.9, "peak_time": "15:28", "period": "the last 2 hours"}
    assert check_numbers("Occupied 74 minutes (61.9%) in the last 2 hours, peak at 3:28 pm.", res, "q") == []
    assert check_numbers("Occupied 80 minutes.", res, "q") == ["80"]
    assert check_numbers("Peak at 16:00.", res, "q") == ["16:00"]


class FakeLLM:
    """Stands in for an LLM: plans correctly, then writes a reply with an invented number."""
    name = "fake"
    grammar = True

    def chat(self, messages, schema=None, max_tokens=120):
        if schema is not None:
            return '{"tool": "occupancy", "period": "today", "time": "", "zone": ""}'
        return "The room was busy for 999 minutes."


def test_verifier_falls_back_to_template(store):
    r = Agent(store, FakeLLM(), now_fn=lambda: NOW).ask("how busy was it today?")
    assert r["llm_plan"] and r["fallback"] and r["rejected_numbers"] == ["999"]
    assert r["answer"] == r["result"]["answer"]


def test_store_logs_engine_state(tmp_path):
    s = HistoryStore(str(tmp_path / "h.sqlite"))
    for i in range(5):
        s.add_state({"t": 1000.0 + i, "presence": True, "activity": "still",
                     "people": {"total": 1, "moving": 0, "still": 1}, "zone": "B2", "x": 1, "y": 2,
                     "confidence": .8, "vitals": {"breathing_bpm": 14.5}, "nodes": [{"alive": True}] * 3})
    s.flush()
    assert s.span()["samples"] == 5 and s.latest()["breath"] == 14.5
