"""SQLite history of what the sensing pipeline saw, one row per second.

The live engine produces a state dict 10 times a second. `HistoryStore.add_state` keeps one
sample per second (presence, activity, people counts, zone, position, breathing rate, node
health) so the assistant can answer questions about the past without keeping raw CSI.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    t        REAL PRIMARY KEY,   -- unix time, seconds
    presence INTEGER NOT NULL,
    activity TEXT NOT NULL,      -- empty / still / moving
    total    INTEGER NOT NULL,
    moving   INTEGER NOT NULL,
    still    INTEGER NOT NULL,
    zone     TEXT,
    x        REAL,
    y        REAL,
    conf     REAL,
    breath   REAL,               -- breaths per minute, only while one person is still
    nodes_up INTEGER NOT NULL,
    nodes_n  INTEGER NOT NULL
);
"""

COLUMNS = ["t", "presence", "activity", "total", "moving", "still", "zone", "x", "y", "conf", "breath",
           "nodes_up", "nodes_n"]
MAX_GAP_S = 5.0  # a gap longer than this between samples counts as "no data"


class HistoryStore:
    def __init__(self, path: str, retention_days: float = 30.0, interval: float = 1.0):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.retention_s = retention_days * 86400.0
        self.interval = interval
        self._last = 0.0
        self._pending: List[tuple] = []
        self._last_flush = time.time()

    # ---- writing ---------------------------------------------------------------------------
    @staticmethod
    def row_from_state(st: Dict) -> tuple:
        ppl = st.get("people") or {}
        vit = st.get("vitals") or {}
        nodes = st.get("nodes") or []
        br = vit.get("breathing_bpm") if isinstance(vit, dict) else None
        return (round(float(st["t"]), 1), int(bool(st.get("presence"))), st.get("activity") or "empty",
                int(ppl.get("total", 0)), int(ppl.get("moving", 0)), int(ppl.get("still", 0)),
                st.get("zone"), st.get("x"), st.get("y"), st.get("confidence"),
                float(br) if br else None,
                sum(1 for n in nodes if n.get("alive")), len(nodes))

    def add_state(self, st: Dict) -> bool:
        """Keep at most one sample per `interval` seconds; flush to disk every 10 s."""
        t = float(st.get("t", time.time()))
        if t - self._last < self.interval:
            return False
        self._last = t
        self._pending.append(self.row_from_state(st))
        if t - self._last_flush > 10.0:
            self.flush()
        return True

    def add_rows(self, rows: Iterable[tuple]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO samples VALUES (%s)" % ",".join("?" * len(COLUMNS)), rows)
            self.db.commit()

    def flush(self) -> None:
        rows, self._pending = self._pending, []
        self._last_flush = time.time()
        if rows:
            self.add_rows(rows)
        if self.retention_s > 0 and int(self._last_flush) % 600 < 10:
            with self.lock:
                self.db.execute("DELETE FROM samples WHERE t < ?", (self._last_flush - self.retention_s,))
                self.db.commit()

    # ---- reading ---------------------------------------------------------------------------
    def load(self, start: float, end: float) -> Dict[str, np.ndarray]:
        """Samples with start <= t < end as column arrays, plus `dt` (seconds each sample covers)."""
        self.flush()
        with self.lock:
            rows = self.db.execute("SELECT %s FROM samples WHERE t >= ? AND t < ? ORDER BY t" % ",".join(COLUMNS),
                                   (start, end)).fetchall()
        out: Dict[str, np.ndarray] = {}
        for i, c in enumerate(COLUMNS):
            vals = [r[i] for r in rows]
            if c in ("activity", "zone"):
                out[c] = np.array(vals, dtype=object)
            else:
                out[c] = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
        t = out["t"]
        if len(t):
            dt = np.diff(t, append=t[-1] + self.interval)
            dt[dt > MAX_GAP_S] = self.interval  # a gap: the sample only stands for itself
            out["dt"] = np.minimum(dt, MAX_GAP_S)
        else:
            out["dt"] = np.zeros(0)
        return out

    def nearest(self, t: float, window: float = 90.0) -> Optional[Dict]:
        self.flush()
        with self.lock:
            row = self.db.execute("SELECT %s FROM samples WHERE t BETWEEN ? AND ? ORDER BY ABS(t - ?) LIMIT 1"
                                  % ",".join(COLUMNS), (t - window, t + window, t)).fetchone()
        return dict(zip(COLUMNS, row)) if row else None

    def latest(self) -> Optional[Dict]:
        self.flush()
        with self.lock:
            row = self.db.execute("SELECT %s FROM samples ORDER BY t DESC LIMIT 1" % ",".join(COLUMNS)).fetchone()
        return dict(zip(COLUMNS, row)) if row else None

    def span(self) -> Dict:
        self.flush()
        with self.lock:
            lo, hi, n = self.db.execute("SELECT MIN(t), MAX(t), COUNT(*) FROM samples").fetchone()
        return {"first": lo, "last": hi, "samples": n}

    def close(self) -> None:
        self.flush()
        self.db.close()
