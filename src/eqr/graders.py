"""Deterministic graders. Cheap, reproducible, and reusable as RL/rejection-sampling rewards.

Each grader returns a score in [0, 1]. `reward` is a weighted blend; `passed` is a strict gate.
Add an LLM-judge for thesis quality as a separate, clearly-labelled score (see README exercises).
"""
from __future__ import annotations

from .citations import tool_calls
from .schemas import ResearchNote

DEFAULT_TOL = 0.005  # 0.5% relative


def metric_accuracy(note: ResearchNote, expected: list[dict]) -> tuple[float, list[dict]]:
    """Fraction of required (name, period) metrics present and within tolerance."""
    details = []
    for exp in expected:
        got = next((m for m in note.metrics if m.name == exp["name"] and m.period == exp["period"]), None)
        tol = exp.get("tol", DEFAULT_TOL)
        if got is None:
            ok, val = False, None
        else:
            val = got.value
            ok = abs(got.value - exp["value"]) <= tol * abs(exp["value"])
        details.append({"name": exp["name"], "period": exp["period"], "expected": exp["value"], "got": val, "ok": ok})
    score = sum(d["ok"] for d in details) / len(details) if details else 1.0
    return score, details


def _cited(note: ResearchNote) -> list[str]:
    ids = [s for m in note.metrics for s in m.sources] + [s for c in note.comps for s in c.sources] + [s for v in note.valuation for s in v.sources] + (note.recommendation.sources if note.recommendation else [])
    for group in (note.bull_case, note.bear_case, note.key_risks):
        ids += [s for c in group for s in c.sources]
    return ids


def citation_grounded(note: ResearchNote, sources_seen: set[str]) -> float:
    """Fraction of cited source ids that really appeared in a tool result (catches fabricated citations)."""
    ids = _cited(note)
    return sum(i in sources_seen for i in ids) / len(ids) if ids else 0.0


def recommendation_consistent(note: ResearchNote, trace: dict) -> float | None:
    """1.0 if the note's rating/target match the tool's computed result exactly, 0.0 if not, None if not applicable.
    A note that hands back a different rating or target than the code computed is a transcription or integrity failure."""
    results = [r for name, _inp, r in tool_calls(trace) if name == "get_price_target"]
    rec = note.recommendation
    if rec is None:
        return 0.0 if results else None  # computed a target but dropped it from the note
    if not results:
        return 0.0  # a rating with no computed basis is invented
    r = results[-1]
    ok = (rec.rating == r["rating"] and abs(rec.price_target - r["price_target"]) < 0.01
          and abs(rec.current_price - r["current_price"]) < 0.01
          and abs(rec.expected_total_return_pct - r["expected_total_return_pct"]) < 0.05)
    return 1.0 if ok else 0.0


def grade(case: dict, note: ResearchNote | None, trace: dict) -> dict:
    if note is None:
        return {"schema_ok": 0.0, "metric_accuracy": 0.0, "citation_grounded": 0.0, "reward": 0.0, "passed": False,
                "details": []}
    acc, details = metric_accuracy(note, case.get("expected", []))
    cit = citation_grounded(note, set(trace.get("sources_seen", [])))
    rec = recommendation_consistent(note, trace)
    reward = 0.6 * acc + 0.4 * cit
    return {
        "schema_ok": 1.0,
        "metric_accuracy": acc,
        "citation_grounded": cit,
        "recommendation_consistent": rec,
        "reward": round(reward, 4),
        "passed": acc >= 0.999 and cit >= 0.999 and rec in (None, 1.0),
        "details": details,
    }
