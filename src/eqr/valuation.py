"""Valuation multiples: web price (guarded) x EDGAR shares, plus EDGAR fundamentals. All math in code.

Honest limits, returned with every result:
- Multiples use the latest FISCAL YEAR, not LTM, so they lag for companies with older year-ends.
- EV = market cap + net_debt_proxy (long-term debt - cash). Excludes current debt, leases, minorities, preferred.
- Price comes from web pages. A row whose price is conflicting/stale/unverified, or whose implied market cap
  (price x SEC shares) disagrees by >10% with EVERY quote-verified market cap found, has its multiples withheld and is
  excluded from medians. The market cap printed on a price page is NOT used: it is unverified and pages disagree.
"""
from __future__ import annotations

import statistics as st
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import comps, edgar, fx, prices

MCAP_TOL = 0.10
NOTES = [
    "Multiples use the latest fiscal year (not LTM); companies with older year-ends are less current.",
    "EV = market cap + net_debt_proxy (long-term debt - cash); excludes current debt, leases and minorities.",
    "Prices come from web search with verbatim-quote verification; check `flags` before relying on a row.",
    "Price x SEC shares is cross-checked against a quote-verified market cap; market_cap_unverified means that check could not run.",
    "debt_assumed_zero: no long-term-debt tag in any year, so net debt = -cash; excluded from EV-multiple medians.",
    "check_outlier flags extreme multiples (often one-off items in the fiscal year): inspect before relying on them.",
    "n/m means not meaningful (negative or zero earnings). Withheld rows are excluded from medians.",
]
MULTIPLES = ("ev_sales", "ev_ebit", "pe", "fcf_yield_pct")


def _to_usd(m: dict, ccy: str, rate: float) -> dict:
    """Convert a company's monetary metrics from its reporting currency to USD at one spot rate."""
    out = {}
    for k, v in m.items():
        if v["unit"] == ccy:
            out[k] = {**v, "value": v["value"] * rate, "unit": "USD"}
        elif v["unit"] == f"{ccy}/shares":
            out[k] = {**v, "value": v["value"] * rate, "unit": "USD/shares"}
        else:
            out[k] = v  # percentages and other ratios are currency-neutral
    return out


def _row(ticker: str, today: date, price: dict, shares_fn, comps_fn, fx_fn=fx.get_rate, mc: dict | None = None) -> dict:
    c = comps_fn(ticker, today)
    m = c["metrics"]
    ccy = c.get("currency", "USD")
    sh = shares_fn(ticker)
    flags = list(price["flags"])
    row = {
        "ticker": ticker.upper(), "company": c["company"], "price": price["price"], "price_date": price["as_of"],
        "financials_period_end": c["period_end"], "shares": sh["shares"], "shares_as_of": sh["as_of"],
        "metrics": {}, "flags": flags, "withheld": False, "reporting_currency": ccy,
        "source": f"calc:valuation:{ticker.upper()}:{price['as_of'] or today.isoformat()}",
    }
    srcs = set(price["sources"]) | {sh["source"]}
    if sh["multi_class"]:
        flags.append("multi_class_shares_summed")
    if sh.get("basis") != "cover_page":
        flags.append("shares_from_weighted_avg_diluted")
    flags.extend(c.get("flags", []))
    if c["stale"]:
        flags.append("stale_financials")
    if not price["usable"]:
        row["withheld"] = True
        row["sources"] = sorted(srcs)
        return row

    if ccy != "USD":  # price is in USD (US listing): put the financials in USD at today's spot rate
        try:
            rate = fx_fn(ccy, "USD", None)
        except Exception as e:
            flags.append(f"fx_unavailable:{ccy}:{type(e).__name__}")
            row["withheld"] = True
            row["sources"] = sorted(srcs)
            return row
        m = _to_usd(m, ccy, rate["rate"])
        flags.append(f"converted_at_spot:{ccy}USD={rate['rate']:.4f}")
        if rate["source"]:
            srcs.add(rate["source"])

    mcap = price["price"] * sh["shares"]
    # Cross-check price x SEC shares (this catches share-count and ADR-ratio errors) against a market cap that was itself
    # quote-verified. If ANY verified source agrees, the row passes; sources that disagree with each other are flagged.
    mc = mc or {}
    srcs.update(mc.get("sources", []))
    flags.extend(mc.get("flags", []))
    vals = mc.get("values") or ([mc["market_cap_usd"]] if mc.get("market_cap_usd") else [])
    if any(abs(mcap / v - 1) <= MCAP_TOL for v in vals):
        if max(vals) / min(vals) - 1 > MCAP_TOL:
            flags.append("market_cap_sources_disagree")
    elif vals:
        closest = min(vals, key=lambda v: abs(v / mcap - 1))
        flags.append(f"market_cap_mismatch:implied={mcap:,.0f},verified={closest:,.0f}")
        row["withheld"] = True
        row["sources"] = sorted(srcs)
        return row
    elif ccy != "USD":  # no verified market cap: the ADR-to-share ratio cannot be confirmed
        flags.append("adr_ratio_unverified")
        row["withheld"] = True
        row["sources"] = sorted(srcs)
        return row
    else:
        flags.append("market_cap_unverified")  # domestic row: keep it, but say the cross-check could not run

    def put(name, value, unit, inputs):
        if value is not None:
            row["metrics"][name] = {"value": round(value, 2), "unit": unit}
            for i in inputs:
                srcs.update(m[i]["sources"])

    put("market_cap", mcap, "USD", [])
    nd = m.get("net_debt_proxy")
    if nd is None:
        flags.append("no_net_debt")
    else:
        ev = mcap + nd["value"]
        put("enterprise_value", ev, "USD", ["net_debt_proxy"])
        if "revenue" in m:
            put("ev_sales", ev / m["revenue"]["value"], "x", ["net_debt_proxy", "revenue"])
        oi = m.get("operating_income")
        if oi and oi["value"] > 0:
            put("ev_ebit", ev / oi["value"], "x", ["net_debt_proxy", "operating_income"])
    eps = m.get("eps_diluted")
    if eps and eps["value"] > 0:
        put("pe", price["price"] / eps["value"], "x", ["eps_diluted"])
    if "fcf" in m:
        put("fcf_yield_pct", m["fcf"]["value"] / mcap * 100, "percent", ["fcf"])
    for k in ("eps_diluted", "operating_income", "net_debt_proxy", "revenue", "dps"):
        if k in m:
            srcs.update(m[k]["sources"])
    row["fundamentals"] = {"eps_diluted": m.get("eps_diluted", {}).get("value"),
                           "operating_income": m.get("operating_income", {}).get("value"),
                           "net_debt_proxy": m.get("net_debt_proxy", {}).get("value"),
                           "revenue": m.get("revenue", {}).get("value"),
                           "dps": m.get("dps", {}).get("value"), "shares": sh["shares"]}
    for k, lo, hi in (("pe", 4, 150), ("ev_ebit", 3, 100), ("fcf_yield_pct", -50, 20)):
        v = row["metrics"].get(k, {}).get("value")
        if v is not None and not lo <= v <= hi:
            flags.append(f"check_outlier:{k}={v}")  # not withheld: could be real (one-off gain, depressed earnings)
    row["sources"] = sorted(srcs)
    return row


def get_valuation(tickers: list[str], today: date | None = None, *, price_fn=prices.get_price,
                  shares_fn=edgar.get_shares_outstanding, comps_fn=comps._company, fx_fn=fx.get_rate,
                  mcap_fn=prices.get_market_cap) -> dict:
    today = today or date.today()
    names: dict[str, str] = {}
    errors: dict[str, str] = {}
    for t in tickers:  # company names come from EDGAR; also fails fast on unknown tickers
        try:
            names[t] = edgar.lookup(t)[1]
        except Exception as e:
            errors[t.upper()] = f"{type(e).__name__}: {e}"
    with ThreadPoolExecutor(4) as ex:  # prices and market caps are network-bound and independent
        price_futs = {t: ex.submit(price_fn, t, names[t], today) for t in names}
        mcap_futs = {t: ex.submit(mcap_fn, t, names[t], today) for t in names} if mcap_fn else {}
    rows = []
    for t in names:
        try:
            try:
                mc = mcap_futs[t].result() if t in mcap_futs else None
            except Exception as e:  # a failed market-cap lookup weakens the cross-check, it does not blank the row
                mc = {"flags": [f"mcap_lookup_error:{type(e).__name__}"]}
            rows.append(_row(t, today, price_futs[t].result(), shares_fn, comps_fn, fx_fn, mc))
        except Exception as e:
            errors[t.upper()] = f"{type(e).__name__}: {e}"
    medians = {}
    for k in MULTIPLES:
        used = [r for r in rows if not r["withheld"] and "stale_financials" not in r["flags"] and k in r["metrics"]
                and not (k.startswith("ev_") and "debt_assumed_zero" in r["flags"])]
        if used:
            medians[k] = {"value": round(st.median(r["metrics"][k]["value"] for r in used), 2),
                          "unit": used[0]["metrics"][k]["unit"], "n": len(used),
                          "sources": sorted({s for r in used for s in r["sources"]})}
    return {"rows": rows, "medians": medians, "errors": errors, "notes": NOTES}
