"""Append-only log of every rating the agent produces (data/ratings_history.jsonl, git-ignored).

Why: nothing tests whether the ratings predict returns. With the date, price and target of each call on file, a later
backtest can ask "what did the Buys do over the next 12 months?" without having to re-run anything.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .schemas import ResearchNote

DEFAULT_PATH = Path("data/ratings_history.jsonl")


def log_rating(note: ResearchNote, trace: dict, path: str | Path = DEFAULT_PATH, today: date | None = None) -> bool:
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return True
