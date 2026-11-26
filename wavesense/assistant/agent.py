"""The question-answering agent.

    question --(planner: LLM or rules)--> {tool, period/time/zone}
             --(tool over SQLite history)--> exact values + template answer
             --(writer: LLM, optional)--> fluent answer
             --(number check)--> fluent answer, or the template answer if the LLM added a number
                                 that is not in the tool result

The LLM never sees raw CSI and never does arithmetic or date maths; it routes and phrases.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from .periods import find_period, find_time
from .tools import REFUSAL, TOOLS, run_tool

TOOL_NAMES = list(TOOLS) + ["none"]
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": TOOL_NAMES},
        "period": {"type": "string"},
        "time": {"type": "string"},
        "zone": {"type": "string"},
    },
    "required": ["tool", "period", "time", "zone"],
}

PLAN_SYSTEM = """You route questions about a WiFi room-sensing system to exactly one tool.
The system logs, every second, how many people are in one room, whether they move or sit still,
which zone (A1..C3) they are in, their breathing rate, and whether the sensor boards are online.

Tools:
%s
- none: the question is not about this room's sensing data (weather, jokes, coding, other places).

Reply with JSON only: {"tool": "...", "period": "...", "time": "...", "zone": "..."}
- period: copy the time-range words from the question exactly (e.g. "last 2 hours", "yesterday",
  "this morning", "between 2 pm and 4 pm", "since 9 am"). Use "" if there are none.
- time: for people_at only, copy the moment (e.g. "at 3 pm", "20 minutes ago", "yesterday at 9:30").
- zone: a zone name like "B2" if the question names one, else "".
Never compute dates or numbers yourself.

Examples:
Q: is anyone in the room right now?
{"tool": "current_status", "period": "", "time": "", "zone": ""}
Q: how busy was the room this morning?
{"tool": "occupancy", "period": "this morning", "time": "", "zone": ""}
Q: how many people were here at 3 pm?
{"tool": "people_at", "period": "", "time": "at 3 pm", "zone": ""}
Q: when did someone first come in yesterday?
{"tool": "presence_times", "period": "yesterday", "time": "", "zone": ""}
Q: how long was spent near zone C1 in the last 2 hours?
{"tool": "zone_usage", "period": "last 2 hours", "time": "", "zone": "C1"}
Q: was there any long period without movement today?
{"tool": "alerts", "period": "today", "time": "", "zone": ""}
Q: write me a poem about cats
{"tool": "none", "period": "", "time": "", "zone": ""}"""

WRITE_SYSTEM = """You answer a question about a room-sensing log. You get the QUESTION and a DRAFT
answer that was computed exactly from the log. Reply with one short, natural sentence that answers
the QUESTION using only facts from the DRAFT. Copy every number and clock time exactly as written in
the DRAFT. Do not add, round or calculate any numbers. If the DRAFT says there is no data, say so."""


def plan_system() -> str:
    lines = "\n".join("- %s(%s): %s" % (k, ", ".join(v["args"]), v["desc"]) for k, v in TOOLS.items())
    return PLAN_SYSTEM % lines


# ---- backends -------------------------------------------------------------------------------
class LlamaCppBackend:
    """Local GGUF model through llama-cpp-python (CPU, or CUDA on a Jetson build)."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: Optional[int] = None,
                 n_gpu_layers: int = -1, grammar: bool = True):
        from llama_cpp import Llama  # optional dependency
        self.name = "llama.cpp:" + os.path.basename(model_path)
        self.grammar = grammar
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(),
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def chat(self, messages: List[Dict], schema: Optional[Dict] = None, max_tokens: int = 120) -> str:
        kw = {}
        if schema is not None and self.grammar:
            kw["response_format"] = {"type": "json_object", "schema": schema}
        r = self.llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens, **kw)
        return r["choices"][0]["message"]["content"] or ""


class OpenAICompatBackend:
    """Any OpenAI-compatible server: Ollama (http://127.0.0.1:11434/v1), llama-server, vLLM ..."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: float = 120.0):
        self.name = "openai:" + model
        self.base, self.model, self.key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.grammar = True

    def chat(self, messages: List[Dict], schema: Optional[Dict] = None, max_tokens: int = 120) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"] or ""


# ---- rule-based planner ------------------------------------------------------------------------
RULES: List[Tuple[str, List[str], float]] = [
    ("breathing", [r"breath", r"respirat", r"\bbpm\b", r"breathing rate"], 3.0),
    ("node_health", [r"\bboards?\b", r"\bnodes?\b", r"\bsensors?\b.*(online|offline|work|drop|down|up)",
                     r"\besp", r"offline", r"dropout", r"disconnect", r"signal"], 3.0),
    ("alerts", [r"alert", r"unusual", r"\bfall", r"anomal", r"worr", r"concern", r"strange", r"at night",
                r"inactiv", r"without (any )?mov", r"didn'?t move", r"no movement"], 2.5),
    ("zone_usage", [r"\bzones?\b", r"\bwhere\b", r"corner", r"\barea\b", r"\bspot\b", r"part of the room",
                    r"\b[a-c][1-3]\b", r"which side"], 2.0),
    ("presence_times", [r"\bfirst\b", r"\barriv", r"\bleave\b", r"\bleft\b", r"came in", r"come in",
                        r"\benter", r"when did", r"how many times", r"last seen", r"\blast (time|detected)"], 2.0),
    ("activity_breakdown", [r"\bmoving\b", r"\bmove(d)?\b", r"\bstill\b", r"sitting", r"\bsat\b", r"active",
                            r"walk", r"sedentary", r"\bidle\b", r"activity"], 2.5),
    ("occupancy", [r"occupied", r"occupancy", r"how long", r"busy", r"\bpeak\b", r"most people",
                   r"\baverage\b", r"how many people", r"\bempty\b", r"crowded", r"\bused\b", r"anyone",
                   r"someone", r"people"], 1.0),
]
NOW_WORDS = r"\b(right now|now|currently|at the moment|at present|presently)\b"


def rule_plan(q: str) -> Dict:
    t = q.lower()
    plan = {"tool": "none", "period": find_period(q) or "", "time": "", "zone": ""}
    z = re.search(r"\b([a-c][1-3])\b", t)
    if z:
        plan["zone"] = z.group(1).upper()
    tm = find_time(q)
    scores = {}
    for name, pats, w in RULES:
        hits = sum(1 for p in pats if re.search(p, t))
        if hits:
            scores[name] = hits * w
    if re.search(NOW_WORDS, t) and not plan["period"] and "breathing" not in scores and "node_health" not in scores:
        plan["tool"] = "current_status"
        return plan
    if tm and not plan["period"] and not ({"breathing", "node_health", "zone_usage"} & set(scores)):
        plan.update(tool="people_at", time=tm)
        return plan
    if scores:
        plan["tool"] = max(scores, key=lambda k: (scores[k], -[r[0] for r in RULES].index(k)))
    return plan


# ---- number check -----------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\w.])(\d{1,2}:\d{2}|\d+(?:\.\d+)?)(?:\s*(am|pm|a\.m\.|p\.m\.))?", re.I)


def _allowed(result: Dict, question: str):
    nums, times = set(), set()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "period" and isinstance(x, str):
                    nums.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", x))
                if k != "answer":
                    walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float)):
            nums.add(float(v))
        elif isinstance(v, str):
            for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", v):
                h, mi = int(m.group(1)), int(m.group(2))
                times.add((h, mi))
            for m in re.finditer(r"[A-C]([1-3])", v):
                nums.add(float(m.group(1)))
    walk(result)
    for m in NUM_RE.finditer(question):
        tok = m.group(1)
        if ":" in tok:
            h, mi = map(int, tok.split(":"))
            times.add((h, mi))
        else:
            nums.add(float(tok))
    return nums, times


def check_numbers(answer: str, result: Dict, question: str) -> List[str]:
    """Numbers / clock times in `answer` that are not backed by the tool result or the question."""
    nums, times = _allowed(result, question)
    bad = []
    for m in NUM_RE.finditer(answer):
        tok, ap = m.group(1), (m.group(2) or "").replace(".", "").lower()
        if ":" in tok:
            h, mi = map(int, tok.split(":"))
            cands = {(h, mi)}
            if ap == "pm" and h < 12:
                cands = {(h + 12, mi)}
            elif ap == "am" and h == 12:
                cands = {(0, mi)}
            elif not ap and h < 12:
                cands.add((h + 12, mi))
            if not (cands & times):
                bad.append(m.group(0))
            continue
        v = float(tok)
        if ap:  # "3 pm"
            h = int(v) % 12 + (12 if ap == "pm" else 0)
            if (h, 0) in times or float(int(v)) in nums:
                continue
            bad.append(m.group(0))
            continue
        if any(abs(v - n) <= max(0.051, 0.006 * abs(n)) or round(n) == v for n in nums):
            continue
        # "zone B2" -> 2, ordinals etc. tolerated only for 1..3 when a zone is present
        bad.append(m.group(0))
    return bad


def _parse_json(txt: str) -> Optional[Dict]:
    try:
        return json.loads(txt)
    except ValueError:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


# ---- agent ------------------------------------------------------------------------------------
class Agent:
    def __init__(self, store, backend=None, write: bool = True, verify: bool = True, now_fn=time.time):
        self.store = store
        self.backend = backend
        self.write = write and backend is not None
        self.verify = verify
        self.now_fn = now_fn

    @property
    def name(self) -> str:
        return self.backend.name if self.backend else "rules"

    def plan(self, q: str) -> Tuple[Dict, bool]:
        """(plan, used_llm)."""
        if self.backend is None:
            return rule_plan(q), False
        msgs = [{"role": "system", "content": plan_system()}, {"role": "user", "content": "Q: " + q.strip()}]
        try:
            p = _parse_json(self.backend.chat(msgs, schema=PLAN_SCHEMA, max_tokens=80))
        except Exception:  # server down, bad output ... fall back to rules
            p = None
        if not p or p.get("tool") not in TOOL_NAMES:
            return rule_plan(q), False
        return {"tool": p.get("tool"), "period": str(p.get("period") or ""), "time": str(p.get("time") or ""),
                "zone": str(p.get("zone") or "").upper()}, True

    def phrase(self, q: str, result: Dict) -> str:
        msgs = [{"role": "system", "content": WRITE_SYSTEM},
                {"role": "user", "content": "QUESTION: %s\nDRAFT: %s" % (q.strip(), result["answer"])}]
        return self.backend.chat(msgs, max_tokens=110).strip()

    def ask(self, q: str, now: Optional[float] = None) -> Dict:
        now = self.now_fn() if now is None else now
        t0 = time.perf_counter()
        plan, used_llm = self.plan(q)
        t1 = time.perf_counter()
        if plan["tool"] == "none":
            result = {"refused": True}
            answer, raw, bad = REFUSAL, None, []
        else:
            result = run_tool(self.store, plan["tool"], plan, now)
            answer, raw, bad = result["answer"], None, []
            if self.write and not result.get("no_data"):  # no-data replies stay as templates
                try:
                    raw = self.phrase(q, result)
                except Exception:
                    raw = None
                if raw:
                    bad = check_numbers(raw, result, q) if self.verify else []
                    answer = raw if not bad else result["answer"]
        t2 = time.perf_counter()
        return {"question": q, "answer": answer, "tool": plan["tool"], "args": plan, "result": result,
                "backend": self.name, "llm_plan": used_llm, "llm_answer": raw, "rejected_numbers": bad,
                "fallback": bool(bad), "plan_ms": round((t1 - t0) * 1000, 1), "total_ms": round((t2 - t0) * 1000, 1)}


def make_backend(cfg: Dict, root: str):
    """Build the backend described by cfg["assistant"]; returns None for the rule-based planner."""
    a = (cfg or {}).get("assistant", {})
    kind = a.get("backend", "auto")
    if kind == "rules":
        return None
    if kind == "openai":
        return OpenAICompatBackend(a.get("base_url", "http://127.0.0.1:11434/v1"), a.get("model", "qwen2.5:3b"))
    path = a.get("model_path", "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf")
    path = path if os.path.isabs(path) else os.path.join(root, path)
    if os.path.exists(path):
        try:
            return LlamaCppBackend(path, n_threads=a.get("threads"), grammar=a.get("grammar", True))
        except ImportError:
            if kind == "llama_cpp":
                raise
            return None
    if kind == "llama_cpp":
        raise FileNotFoundError(path)
    return None
