"""Synthetic sensing history for testing and evaluating the assistant.

Writes per-second samples that look like what the live engine logs for a lived-in room:
nights mostly empty, occupied blocks during the day with 1-3 people, still/moving phases,
zone changes, breathing while one person is still, one long inactivity per day, a few short
board dropouts, short detector flickers and one server-off gap. Because the log *is* the
ground truth for "what did the system record", answers can be checked exactly.
"""

from __future__ import annotations

import datetime as dt
from typing import List

import numpy as np

ZONES = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
ZXY = {z: ((i % 3 + 0.5) * 2.0, (i // 3 + 0.5) * 5.0 / 3) for i, z in enumerate(ZONES)}


def _day(rng, day0: float, end: float, rows: List[tuple]) -> None:
    t = day0
    occ_blocks = []
    # optional night-time visit (e.g. kitchen at 2 am)
    if rng.random() < 0.6:
        s = day0 + rng.uniform(1.5, 3.5) * 3600
        occ_blocks.append((s, s + rng.uniform(4, 12) * 60, "night"))
    t = day0 + rng.uniform(7.0, 8.2) * 3600
    long_still_done = False
    while t < day0 + 23 * 3600:
        dur = rng.uniform(20, 95) * 60
        kind = "day"
        if not long_still_done and t > day0 + 13 * 3600:
            kind, long_still_done = "inactive", True
            dur = rng.uniform(26, 45) * 60
        occ_blocks.append((t, t + dur, kind))
        t += dur + rng.uniform(4, 40) * 60
    state = {"zone": rng.choice(ZONES), "breath": rng.uniform(13, 17)}
    cur = day0
    for s, e, kind in occ_blocks:
        _empty(rows, cur, min(s, end))
        if s >= end:
            return
        _occupied(rng, rows, s, min(e, end), kind, state)
        cur = min(e, end)
    _empty(rows, cur, min(day0 + 86400, end))


def _empty(rows, s, e):
    for t in np.arange(np.ceil(s), e, 1.0):
        rows.append([float(t), 0, "empty", 0, 0, 0, None, None, None, 0.9, None, 3, 3])


def _occupied(rng, rows, s, e, kind, state):
    t = float(np.ceil(s))
    if kind == "night":
        n_people = 1
    elif kind == "inactive":
        n_people = 1
    else:
        n_people = int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
    while t < e:
        seg = rng.uniform(2, 14) * 60
        if kind == "inactive":
            seg, act = e - t, "still"
        elif kind == "night":
            act = "moving"
        else:
            act = "moving" if rng.random() < 0.45 else "still"
        if act == "moving":
            mv = int(rng.integers(1, n_people + 1))
        else:
            mv = 0
        st = n_people - mv
        if rng.random() < 0.5 or act == "moving":
            state["zone"] = ZONES[int(np.clip(ZONES.index(state["zone"]) + rng.choice([-3, -1, 1, 3]), 0, 8))]
        state["breath"] = float(np.clip(state["breath"] + rng.normal(0, 1.0), 10, 22))
        end = min(e, t + seg)
        while t < end:
            z = state["zone"]
            x, y = ZXY[z]
            br = None
            if act == "still" and n_people == 1:
                br = round(float(state["breath"] + rng.normal(0, 0.4)), 1)
            rows.append([t, 1, act, n_people, mv, st, z, round(x + rng.normal(0, .3), 2),
                         round(y + rng.normal(0, .3), 2), round(float(rng.uniform(.55, .95)), 2), br, 3, 3])
            t += 1.0


def generate(store, now: float, days: int = 2, seed: int = 0) -> dict:
    """Fill `store` with `days` days of history ending at `now`. Returns a few facts about it."""
    rng = np.random.default_rng(seed)
    today = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    rows: List[list] = []
    for d in range(days - 1, -1, -1):
        day0 = today - d * 86400
        _day(rng, day0, min(day0 + 86400, now), rows)
    arr_t = np.array([r[0] for r in rows])
    # board dropouts: one board stops for 10 s .. 6 min, ~5 per day
    for _ in range(5 * days):
        s = rng.uniform(arr_t[0], arr_t[-1])
        idx = np.where((arr_t >= s) & (arr_t < s + rng.uniform(10, 360)))[0]
        for i in idx:
            rows[i][11] = 2
    # detector flicker: 1-2 s count glitches
    for _ in range(40 * days):
        i = int(rng.integers(0, len(rows)))
        if rows[i][1]:
            rows[i][3] = int(np.clip(rows[i][3] + int(rng.choice([-1, 1])), 1, 3))
            rows[i][4] = min(rows[i][4], rows[i][3])
            rows[i][5] = rows[i][3] - rows[i][4]
    # one 12-minute server-off gap yesterday afternoon
    gap0 = today - 86400 + 15.5 * 3600
    rows = [r for r in rows if not (gap0 <= r[0] < gap0 + 720)]
    store.add_rows([tuple(r) for r in rows])
    return {"rows": len(rows), "first": rows[0][0], "last": rows[-1][0]}
