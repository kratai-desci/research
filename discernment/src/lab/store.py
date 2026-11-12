"""SQLite results store: runs → episodes → steps and checks. The dashboard reads this file."""

import json
import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(os.environ.get("LAB_DB", Path(__file__).parent / "data" / "lab.db"))
# Each VM numbers its rows from its own base (e.g. 1_000_000_000 on the second), so the dashboard can merge
# the databases of several VMs without clashes.
ID_BASE = int(os.environ.get("LAB_ID_BASE", "0"))
_based: set[str] = set()  # tables whose ids are already at or above ID_BASE

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  name TEXT, system TEXT, model TEXT, track TEXT, condition TEXT,
  status TEXT, total INTEGER, done INTEGER DEFAULT 0,
  started REAL, finished REAL, config TEXT, experiment TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY,
  run_id INTEGER, day INTEGER, task TEXT, slot TEXT, split TEXT, message TEXT,
  answer TEXT, status TEXT, score INTEGER, max_score INTEGER,
  input_tokens INTEGER DEFAULT 0, thinking_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
  tool_calls INTEGER DEFAULT 0, steps INTEGER DEFAULT 0, latency_ms INTEGER DEFAULT 0,
  started REAL, finished REAL
);
CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY,
  episode_id INTEGER, n INTEGER, thought TEXT, action TEXT, input TEXT, observation TEXT,
  input_tokens INTEGER, thinking_tokens INTEGER, output_tokens INTEGER, latency_ms INTEGER
);
CREATE TABLE IF NOT EXISTS checks (
  id INTEGER PRIMARY KEY,
  episode_id INTEGER, n INTEGER, name TEXT, kind TEXT, passed INTEGER, evidence TEXT
);
CREATE INDEX IF NOT EXISTS episodes_by_run ON episodes(run_id, split, day);
CREATE INDEX IF NOT EXISTS steps_by_episode ON steps(episode_id);
CREATE INDEX IF NOT EXISTS checks_by_episode ON checks(episode_id, kind);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    if "experiment" not in {r[1] for r in db.execute("PRAGMA table_info(runs)")}:
        try:
            db.execute("ALTER TABLE runs ADD COLUMN experiment TEXT")  # databases from before experiments
        except sqlite3.OperationalError as e:  # another life starting at the same moment added it first
            if "duplicate column" not in str(e):
                raise
    return db


def insert(db: sqlite3.Connection, table: str, **values) -> int:
    values = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in values.items()}
    if ID_BASE and table not in _based:
        if db.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0] < ID_BASE:
            values["id"] = ID_BASE
        _based.add(table)
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cur = db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))
    db.commit()
    return cur.lastrowid


def update(db: sqlite3.Connection, table: str, row_id: int, **values) -> None:
    sets = ", ".join(f"{k} = ?" for k in values)
    db.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])
    db.commit()


def now() -> float:
    return time.time()
