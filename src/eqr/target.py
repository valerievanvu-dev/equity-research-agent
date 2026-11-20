"""Price target and Buy/Hold/Sell, computed in code from model-proposed scenario assumptions.

The LLM proposes WHAT it believes (growth and a target multiple for bear/base/bull) and explains why. This module
does the math, enforces guardrails, and applies a FIXED rating rule. The model cannot choose or soften the rating.

This is a transparent illustration of a valuation method, not personalized investment advice.

Method (12-month horizon):
- "pe":      target price = current EPS x (1 + growth) x target P/E
- "ev_ebit": equity = EBIT x (1 + growth) x target EV/EBIT - net debt (held constant); price = equity / shares
- price target = probability-weighted average of the three scenario prices (default 25% / 50% / 25%)
- expected total return = (price target / price - 1) + last declared dividend per share / price
"""
from __future__ import annotations

from datetime import date

from . import valuation

BUY_AT, SELL_AT = 15.0, -10.0  # expected 12-month total return thresholds, in percent
WEIGHTS = {"bear": 0.25, "base": 0.50, "bull": 0.25}
MULTIPLE_BOUNDS = {"pe": (3.0, 60.0), "ev_ebit": (3.0, 40.0)}
GROWTH_BOUNDS = (-60.0, 100.0)
RULE = (f"Buy if expected 12-month total return >= {BUY_AT:+.0f}%; Sell if <= {SELL_AT:+.0f}%; otherwise Hold. "
        "Total return = probability-weighted price target vs current price, plus the last declared dividend per "
        "share as a yield.")
LOW_CONVICTION_FLAGS = {"multiple_not_anchored", "wide_dispersion", "dividend_unknown", "few_peers"}
LOW_CONVICTION_ROW_PREFIXES = ("single_source", "stale_financials", "check_outlier", "shares_from_weighted")


def rating_for(total_return_pct: float) -> str:
    if total_return_pct >= BUY_AT:
        return "Buy"
    if total_return_pct <= SELL_AT:
        return "Sell"
    return "Hold"


def implied_price(method: str, f: dict, growth_pct: float, multiple: float) -> float:
    g = 1 + growth_pct / 100
    if method == "pe":
        return f["eps_diluted"] * g * multiple
    return (f["operating_income"] * g * multiple - f["net_debt_proxy"]) / f["shares"]


def _check_fundamentals(method: str, f: dict) -> None:
    if method == "pe":
        if not f.get("eps_diluted") or f["eps_diluted"] <= 0:
            raise ValueError("diluted EPS is missing or not positive: use method 'ev_ebit'")
    else:
        if not f.get("operating_income") or f["operating_income"] <= 0:
            raise ValueError("operating income is missing or not positive: a multiple-based target is not meaningful")
        if f.get("net_debt_proxy") is None:
            raise ValueError("net debt is unavailable for this company: use method 'pe'")
        if not f.get("shares"):
            raise ValueError("share count unavailable")


def _weights(scenarios: dict) -> dict[str, float]:
    given = {k: scenarios[k].get("weight") for k in WEIGHTS}
    if all(v is None for v in given.values()):
        return dict(WEIGHTS)
    if any(v is None for v in given.values()):
        raise ValueError("give a weight for every scenario or for none")
    if abs(sum(given.values()) - 1) > 0.01 or any(not 0.05 <= v <= 0.70 for v in given.values()):
        raise ValueError("weights must sum to 1 and each be between 0.05 and 0.70")
    return {k: float(v) for k, v in given.items()}


def get_price_target(ticker: str, peer_tickers: list[str], method: str, scenarios: dict, horizon_months: int = 12,
                     today: date | None = None, *, valuation_fn=valuation.get_valuation) -> dict:
    today = today or date.today()
    ticker = ticker.upper()
    if method not in MULTIPLE_BOUNDS:
        raise ValueError(f"method must be one of {sorted(MULTIPLE_BOUNDS)}")
    if set(scenarios) != set(WEIGHTS):
        raise ValueError("scenarios must be exactly: bear, base, bull")
    weights = _weights(scenarios)
    lo_m, hi_m = MULTIPLE_BOUNDS[method]
    for name, sc in scenarios.items():
        if not GROWTH_BOUNDS[0] <= float(sc["growth_pct"]) <= GROWTH_BOUNDS[1]:
            raise ValueError(f"{name}: growth_pct must be between {GROWTH_BOUNDS[0]} and {GROWTH_BOUNDS[1]}")
        if not lo_m <= float(sc["multiple"]) <= hi_m:
            raise ValueError(f"{name}: multiple must be between {lo_m} and {hi_m} for method '{method}'")

    peers = [p.upper() for p in peer_tickers if p.upper() != ticker]
    val = valuation_fn([ticker] + peers, today)
    row = next((r for r in val["rows"] if r["ticker"] == ticker), None)
    if row is None:
        raise ValueError(f"no valuation data for {ticker}: {val['errors'].get(ticker, 'unknown error')}")
    if row["withheld"] or not row.get("price"):
        raise ValueError(f"price unavailable for {ticker} (flags: {row['flags']}); cannot set a target")
    f = row["fundamentals"]
    _check_fundamentals(method, f)
    price = float(row["price"])

    out_scen, prices = [], {}
    for name in ("bear", "base", "bull"):
        sc = scenarios[name]
        p = implied_price(method, f, float(sc["growth_pct"]), float(sc["multiple"]))
        if p <= 0:
            raise ValueError(f"{name}: scenario implies a non-positive equity value")
        prices[name] = p
        out_scen.append({"name": name, "weight": weights[name], "growth_pct": float(sc["growth_pct"]),
                         "multiple": float(sc["multiple"]), "price": round(p, 2),
                         "return_pct": round((p / price - 1) * 100, 1), "rationale": str(sc.get("rationale", ""))})
    if not prices["bear"] <= prices["base"] <= prices["bull"]:
        raise ValueError("scenario prices must be ordered bear <= base <= bull")

    target = sum(weights[k] * prices[k] for k in weights)
    dps = f.get("dps")
    div_yield = (dps / price * 100) if dps else 0.0
    price_ret = (target / price - 1) * 100
    total = price_ret + div_yield

    base = scenarios["base"]
    current_mult = row["metrics"].get(method, {}).get("value")
    med = val["medians"].get(method)
    anchors = [a for a in (current_mult, med["value"] if med else None) if a]
    flags: list[str] = []
    if anchors and all(abs(float(base["multiple"]) / a - 1) > 0.35 for a in anchors):
        flags.append("multiple_not_anchored")
    if dps is None:
        flags.append("dividend_unknown")
    if not med or med.get("n", 0) < 2:
        flags.append("few_peers")
    if (prices["bull"] - prices["bear"]) / price > 0.8:
        flags.append("wide_dispersion")
    row_flags = list(row["flags"])
    low = any(x in LOW_CONVICTION_FLAGS for x in flags) or any(x.startswith(LOW_CONVICTION_ROW_PREFIXES) for x in row_flags)

    g0, m0 = float(base["growth_pct"]), float(base["multiple"])
    mults = [round(m0 * k, 1) for k in (0.8, 0.9, 1.0, 1.1, 1.2)]
    grid = []
    for dg in (-10, -5, 0, 5, 10):
        cells = []
        for m in mults:
            try:
                v = implied_price(method, f, g0 + dg, m)
                cells.append(round(v, 2) if v > 0 else None)
            except ZeroDivisionError:
                cells.append(None)
        grid.append({"growth_pct": g0 + dg, "prices": cells})

    sources = sorted(set(row.get("sources", [])))  # leaf ids only: the derived calc:valuation id adds nothing
    src_id = f"calc:target:{ticker}:{today.isoformat()}"
    return {
        "ticker": ticker, "method": method, "horizon_months": horizon_months, "current_price": round(price, 2),
        "price_date": row.get("price_date"), "price_target": round(target, 2),
        "expected_price_return_pct": round(price_ret, 1), "dividend_yield_pct": round(div_yield, 1),
        "expected_total_return_pct": round(total, 1), "rating": rating_for(total), "rating_rule": RULE,
        "conviction": "Low" if low else "Normal", "scenarios": out_scen,
        "anchors": {"current_multiple": current_mult, "peer_median_multiple": med["value"] if med else None,
                    "peers_in_median": med["n"] if med else 0},
        "fundamentals_used": {k: f.get(k) for k in ("eps_diluted", "operating_income", "net_debt_proxy", "dps", "shares")},
        "sensitivity": {"multiples": mults, "rows": grid}, "flags": flags, "data_flags": row_flags,
        "notes": ["Growth is the change in EPS (pe) or EBIT (ev_ebit) from the latest fiscal year to the horizon.",
                  "Net debt and share count are held constant; dividend yield uses the last declared dividend per share.",
                  "Probabilities and scenarios are the model's judgement; the rating is a fixed rule applied to them."],
        "source": src_id, "sources": [src_id] + sources,
    }
