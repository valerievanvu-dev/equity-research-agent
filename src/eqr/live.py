"""One shared "what does the market and the latest filing say today?" lookup for the tracking features.

Wraps get_comps (latest annual operating metrics, from SEC XBRL) and get_valuation (price + multiples) so the thesis
tracker, the monitor and the recommender all see the same numbers and each ticker is fetched once. A ticker whose price
could not be verified comes back with price=None and the reason in flags: nothing is guessed.
"""
from __future__ import annotations

from datetime import date

from . import comps, valuation


def snapshots(tickers: list[str], today: date | None = None, *, valuation_fn=valuation.get_valuation,
              comps_fn=comps.get_comps) -> dict[str, dict]:
    today = today or date.today()
    tickers = [t.upper() for t in tickers]
    if not tickers:
        return {}
    val, cmp_ = valuation_fn(tickers, today), comps_fn(tickers, today)
    vrows = {r["ticker"]: r for r in val["rows"]}
    crows = {r["ticker"]: r for r in cmp_["rows"]}
    errors = {**cmp_.get("errors", {}), **val.get("errors", {})}
    out: dict[str, dict] = {}
    for t in tickers:
        v, c = vrows.get(t), crows.get(t)
        snap = {"ticker": t, "company": (v or c or {}).get("company"), "price": None, "price_date": None, "flags": [],
                "comps": {}, "valuation": {}, "period_end": None, "stale": None, "error": errors.get(t)}
        if c:
            snap.update(period_end=c["period_end"], stale=c["stale"],
                        comps={k: m["value"] for k, m in c["metrics"].items()})
        if v:
            snap["flags"] = list(v["flags"])
            if not v["withheld"] and v.get("price"):
                snap.update(price=float(v["price"]), price_date=v.get("price_date"),
                            valuation={k: m["value"] for k, m in v["metrics"].items()})
        out[t] = snap
    return out
