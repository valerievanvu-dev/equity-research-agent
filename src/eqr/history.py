"""Append-only log of every rating the agent produces (data/ratings_history.jsonl, git-ignored).

Why: nothing tests whether the ratings predict returns. With the date, price and target of each call on file, a later
backtest can ask "what did the Buys do over the next 12 months?" without having to re-run anything.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import store
from .schemas import ResearchNote


def default_path() -> Path:
    return store.data_dir() / "ratings_history.jsonl"


def log_rating(note: ResearchNote, trace: dict, path: str | Path | None = None, today: date | None = None) -> bool:
    """Append one line per rated note. Returns False when the note has no recommendation."""
    rec = note.recommendation
    if rec is None:
        return False
    row = {
        "date": (today or date.today()).isoformat(), "ticker": note.ticker, "company": note.company,
        "rating": rec.rating, "price_target": rec.price_target, "current_price": rec.current_price,
        "price_date": rec.price_date, "expected_total_return_pct": rec.expected_total_return_pct,
        "horizon_months": rec.horizon_months, "method": rec.method, "conviction": rec.conviction,
        "flags": rec.flags, "run_id": trace.get("run_id"), "engine": trace.get("engine", "loop"),
        "model": trace.get("model"),
    }
    path = Path(path) if path else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return True


def load(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else default_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_by_ticker(path: str | Path | None = None) -> dict[str, dict]:
    """Most recent rating per ticker (a later Hold supersedes an earlier Buy). Same-day ties: the later line wins."""
    out: dict[str, dict] = {}
    for row in sorted(load(path), key=lambda r: r["date"]):  # stable sort keeps file order within a date
        out[row["ticker"]] = row
    return out
