"""Thesis tracker: turn a note's thesis into testable pillars, then re-test them against fresh data.

A note says "margins are expanding and the stock is cheap". A tracker asks, later, "is that still true?". Each thesis is
a handful of **pillars**: a metric, a comparison and a threshold (e.g. operating_margin_pct >= 17). Pillars are seeded
from the numbers in the note, and you can add your own. Checking is pure code on SEC + price data: no model call.

Status rule (fixed, so it is the same every time):
- Broken : a `broken`-severity pillar failed (default for pillars you add, and the -20% drawdown pillar).
- Watch  : a `watch`-severity pillar failed, the price reached the target (re-underwrite), or the note is stale.
- Intact : everything measurable passes. Pillars with no data are listed as unknown, never counted as passing.
"""
from __future__ import annotations

import operator
import re
from datetime import date
from pathlib import Path

from . import store
from .schemas import ResearchNote

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}
COMPS_METRICS = {"revenue_growth_pct", "gross_margin_pct", "operating_margin_pct", "net_margin_pct", "fcf_margin_pct"}
VAL_METRICS = {"pe", "ev_ebit", "ev_sales", "fcf_yield_pct"}
METRICS = COMPS_METRICS | VAL_METRICS | {"price"}
STALE_NOTE_DAYS = 120
DRAWDOWN_PCT = 20.0     # default price pillar: review if the stock falls this far below the entry price
MARGIN_CUSHION_PTS = 3.0
KEEP_CHECKS = 20


def thesis_path(ticker: str) -> Path:
    return store.data_dir() / "theses" / f"{ticker.upper()}.json"


def load(ticker: str) -> dict | None:
    return store.read_json(thesis_path(ticker), None)


def load_all() -> list[dict]:
    d = store.data_dir() / "theses"
    return [store.read_json(p, {}) for p in sorted(d.glob("*.json"))] if d.exists() else []


# ------------------------------------------------------------------ pillars
def parse_pillar(text: str, on_fail: str = "broken") -> dict:
    """'operating_margin_pct >= 17' -> a pillar. Raises ValueError with the allowed metrics if it does not parse."""
    m = re.fullmatch(r"\s*([a-z_]+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*", text)
    if not m or m.group(1) not in METRICS:
        raise ValueError(f"cannot parse pillar {text!r}: use '<metric> <op> <number>' with metric in {sorted(METRICS)}")
    metric, op, value = m.group(1), m.group(2), float(m.group(3))
    return {"id": f"user:{metric}{op}{value:g}", "label": f"{metric} {op} {value:g}", "metric": metric, "op": op,
            "value": value, "on_fail": on_fail, "source": "user"}


def seed_pillars(note: ResearchNote) -> list[dict]:
    """Default pillars from the note's own numbers: growth and margin must not deteriorate much, price must hold."""
    pillars: list[dict] = []
    own = next((c for c in note.comps if c.ticker.upper() == note.ticker.upper()), None)
    if own:
        g = own.metrics.get("revenue_growth_pct")
        if g is not None:
            floor = round(g - max(3.0, 0.5 * abs(g)), 1)
            pillars.append({"id": "seed:growth", "label": f"revenue growth stays above {floor:g}% (was {g:.1f}%)",
                            "metric": "revenue_growth_pct", "op": ">=", "value": floor, "on_fail": "watch", "source": "seed"})
        m = own.metrics.get("operating_margin_pct")
        if m is not None:
            floor = round(m - MARGIN_CUSHION_PTS, 1)
            pillars.append({"id": "seed:margin", "label": f"operating margin stays above {floor:g}% (was {m:.1f}%)",
                            "metric": "operating_margin_pct", "op": ">=", "value": floor, "on_fail": "watch", "source": "seed"})
    rec = note.recommendation
    if rec:
        stop = round(rec.current_price * (1 - DRAWDOWN_PCT / 100), 2)
        pillars.append({"id": "seed:drawdown", "label": f"price holds above ${stop:g} (-{DRAWDOWN_PCT:.0f}% from entry)",
                        "metric": "price", "op": ">", "value": stop, "on_fail": "broken", "source": "seed"})
    return pillars


def create(note: ResearchNote, trace: dict | None = None, extra_pillars: list[dict] | None = None,
           today: date | None = None) -> dict:
    """Create or refresh the thesis for a ticker from a note. Re-running keeps your own pillars and the check history."""
    today = today or date.today()
    prev = load(note.ticker) or {}
    rec = note.recommendation
    user = [p for p in prev.get("pillars", []) if p.get("source") == "user"]
    for p in extra_pillars or []:
        if p["id"] not in {u["id"] for u in user}:
            user.append(p)
    th = {
        "ticker": note.ticker.upper(), "company": note.company, "created": prev.get("created", today.isoformat()),
        "updated": today.isoformat(), "run_id": (trace or {}).get("run_id"),
        "thesis": note.thesis, "summary": note.summary,
        "bull": [c.text for c in note.bull_case], "bear": [c.text for c in note.bear_case],
        "risks": [c.text for c in note.key_risks],
        "rating": rec.rating if rec else None, "price_target": rec.price_target if rec else None,
        "entry_price": rec.current_price if rec else None, "horizon_months": rec.horizon_months if rec else None,
        "conviction": rec.conviction if rec else None,
        "pillars": seed_pillars(note) + user, "checks": prev.get("checks", []),
    }
    store.write_json(thesis_path(note.ticker), th)
    return th


# ------------------------------------------------------------------ checking
def _value(snap: dict, metric: str):
    if metric == "price":
        return snap.get("price")
    return snap.get("comps", {}).get(metric) if metric in COMPS_METRICS else snap.get("valuation", {}).get(metric)


def evaluate(th: dict, snap: dict, today: date | None = None) -> dict:
    """Test every pillar against a live snapshot (see live.snapshots). Pure function: no I/O."""
    today = today or date.today()
    results, reasons = [], []
    broken = watch = False
    for p in th["pillars"]:
        v = _value(snap, p["metric"])
        if v is None:
            status = "unknown"
        else:
            status = "ok" if OPS[p["op"]](v, p["value"]) else "failed"
        results.append({**p, "current": v, "status": status})
        if status == "failed":
            reasons.append(f"{p['label']}: now {v:,.2f}")
            broken |= p["on_fail"] == "broken"
            watch |= p["on_fail"] != "broken"
    price, tgt, entry = snap.get("price"), th.get("price_target"), th.get("entry_price")
    progress = None
    if price is not None and tgt and entry and tgt != entry:
        progress = round((price - entry) / (tgt - entry) * 100, 1)
        if price >= tgt and tgt > entry:
            watch = True
            reasons.append(f"price ${price:,.2f} reached the ${tgt:,.2f} target: re-underwrite")
    if price is None:
        reasons.append("no verified price today (" + (", ".join(snap.get("flags", [])[:2]) or "unavailable") + ")")
    age = (today - date.fromisoformat(th["updated"])).days
    if age > STALE_NOTE_DAYS:
        watch = True
        reasons.append(f"note is {age} days old (limit {STALE_NOTE_DAYS}): refresh it")
    status = "Broken" if broken else "Watch" if watch else "Intact"
    return {"ticker": th["ticker"], "date": today.isoformat(), "status": status, "reasons": reasons,
            "pillars": results, "price": price, "progress_to_target_pct": progress, "age_days": age,
            "unknown": [r["label"] for r in results if r["status"] == "unknown"]}


def record_check(th: dict, result: dict) -> None:
    """Append a compact line of this check to the thesis file (drift over time is the point of a tracker)."""
    th["checks"] = (th.get("checks", []) + [{"date": result["date"], "status": result["status"], "price": result["price"],
                                             "failed": [p["label"] for p in result["pillars"] if p["status"] == "failed"]}])[-KEEP_CHECKS:]
    store.write_json(thesis_path(th["ticker"]), th)


def _n(x) -> str:
    return "n/a" if x is None else f"{x:,.2f}"


def render(th: dict, res: dict) -> str:
    icon = {"Intact": "OK", "Watch": "WATCH", "Broken": "BROKEN"}[res["status"]]
    L = [f"## {th['ticker']}: thesis {icon}",
         f"_{th['company']}. Rated {th.get('rating') or 'n/a'}, target ${th['price_target']:,.2f}, entry ${th['entry_price']:,.2f} "
         f"({th['updated']})._" if th.get("price_target") else f"_{th['company']} ({th['updated']})._",
         "", th["thesis"], ""]
    if res["price"] is not None:
        prog = f", {res['progress_to_target_pct']:.0f}% of the way to target" if res["progress_to_target_pct"] is not None else ""
        L.append(f"Price now ${res['price']:,.2f}{prog}.")
    L += ["", "| pillar | threshold | now | status |", "|---|---|---|---|"]
    L += [f"| {p['label']} | {p['op']} {p['value']:g} | {_n(p['current'])} | {p['status']} |" for p in res["pillars"]]
    if res["reasons"]:
        L += [""] + [f"- {r}" for r in res["reasons"]]
    if th.get("checks"):
        L += ["", "History: " + " -> ".join(f"{c['date']} {c['status']}" for c in th["checks"][-6:])]
    return "\n".join(L) + "\n"
