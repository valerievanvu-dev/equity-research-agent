"""Peer universes and operating comps, computed in code from SEC XBRL (not by the LLM).

Why code and not the model: arithmetic and period alignment are exactly where LLMs slip, and comps are
only useful if the numbers are right and reproducible. The model chooses WHICH peers and interprets the table.

Known limits (also returned in every result so the agent can say so):
- Operating comps only. Valuation multiples (EV/EBITDA, P/E) need market prices, which EDGAR doesn't have.
- Fiscal years aren't calendar aligned. Compare via `period_end`; `fy` labels are not consistent across filers.
- US-listed SEC filers only. Foreign leaders (L'Oreal, Shiseido, Beiersdorf, Kao, Amorepacific) are not covered.
"""
from __future__ import annotations

import statistics as st
from datetime import date

from . import edgar

UNIVERSES: dict[str, list[dict]] = {
    "beauty": [
        {"ticker": "EL", "name": "Estee Lauder", "segment": "prestige brands"},
        {"ticker": "COTY", "name": "Coty", "segment": "fragrance and color cosmetics"},
        {"ticker": "IPAR", "name": "Inter Parfums", "segment": "prestige fragrance (licensed brands)"},
        {"ticker": "ELF", "name": "e.l.f. Beauty", "segment": "mass/masstige color cosmetics"},
        {"ticker": "HNST", "name": "Honest Company", "segment": "clean personal care (borderline beauty)"},
        {"ticker": "SKIN", "name": "SkinHealth Systems", "segment": "aesthetics devices (borderline beauty)"},
        {"ticker": "NUS", "name": "Nu Skin", "segment": "direct-selling skincare"},
        {"ticker": "ULTA", "name": "Ulta Beauty", "segment": "specialty retail"},
        {"ticker": "SBH", "name": "Sally Beauty", "segment": "professional/DIY beauty retail"},
        {"ticker": "PG", "name": "Procter & Gamble", "segment": "diversified: beauty is one segment, use with caution"},
    ],
    "tobacco": [
        {"ticker": "MO", "name": "Altria", "segment": "US cigarettes, smokeless, oral nicotine"},
        {"ticker": "PM", "name": "Philip Morris International", "segment": "international cigarettes and smoke-free (IQOS)"},
        {"ticker": "TPB", "name": "Turning Point Brands", "segment": "US smokeless/papers (small cap)"},
        {"ticker": "UVV", "name": "Universal Corp", "segment": "leaf tobacco supplier (B2B, not a brand owner)"},
    ],
}

NOTES = [
    "Operating comps only: no valuation multiples because EDGAR has no market prices.",
    "Fiscal years are not calendar aligned. Compare using period_end, not fiscal-year labels.",
    "net_debt_proxy = long_term_debt - cash. It excludes current debt and lease liabilities.",
    "SEC 10-K filers only: leaders that file 20-F/40-F (e.g. L'Oreal, Shiseido, Beiersdorf, BAT, Imperial) are not covered.",
    "Non-USD filers (IFRS, Form 20-F): operating comps stay in each company's reporting currency (see `currency`); percentages are currency-neutral. Valuation converts to USD at the ECB spot rate.",
    "Medians exclude stale rows (latest annual period older than 15 months).",
]
NEEDED = ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted", "dps", "operating_cash_flow", "capex",
          "cash", "long_term_debt"]
STALE_DAYS = 456  # ~15 months


def peer_universe(sector: str = "beauty") -> dict:
    if sector not in UNIVERSES:
        raise ValueError(f"Unknown sector '{sector}'. Available: {sorted(UNIVERSES)}")
    return {"sector": sector, "companies": UNIVERSES[sector], "notes": NOTES[-2:]}


def _pct(num: float, den: float) -> float | None:
    return round(num / den * 100, 2) if den else None


def _row_for(fin: dict, metric: str, fy: int) -> dict | None:
    return next((r for r in fin["metrics"].get(metric, []) if r["fiscal_year"] == fy), None)


def _company(ticker: str, today: date) -> dict:
    fin = edgar.get_financials(ticker, NEEDED, years=3)
    ccy = fin.get("currency", "USD")  # reporting currency: EUR for an IFRS filer such as Unilever
    revs = sorted(fin["metrics"].get("revenue", []), key=lambda r: r["period_end"], reverse=True)
    if not revs:
        raise ValueError("no annual revenue found in XBRL")
    cur = revs[0]
    fy = cur["fiscal_year"]
    pick = {m: _row_for(fin, m, fy) for m in NEEDED}
    metrics: dict[str, dict] = {}

    def put(name: str, value, unit: str, rows: list[dict | None]):
        if value is not None and all(rows):
            metrics[name] = {"value": value, "unit": unit, "sources": [r["source"] for r in rows]}

    put("revenue", cur["value"], ccy, [cur])
    for name, unit in (("operating_income", ccy), ("net_income", ccy), ("eps_diluted", f"{ccy}/shares"),
                       ("dps", f"{ccy}/shares")):
        if pick[name]:
            put(name, pick[name]["value"], unit, [pick[name]])
    if len(revs) > 1:
        prev = revs[1]
        days = (date.fromisoformat(cur["period_end"]) - date.fromisoformat(prev["period_end"])).days
        if 340 <= days <= 380:  # skip growth if a year is missing: it would be multi-year growth
            put("revenue_growth_pct", _pct(cur["value"] - prev["value"], prev["value"]), "percent", [cur, prev])
    for name, num in (("gross_margin_pct", "gross_profit"), ("operating_margin_pct", "operating_income"),
                      ("net_margin_pct", "net_income")):
        if pick[num]:
            put(name, _pct(pick[num]["value"], cur["value"]), "percent", [pick[num], cur])
    ocf, capex = pick["operating_cash_flow"], pick["capex"]
    if ocf and capex:
        fcf = ocf["value"] - capex["value"]
        put("fcf", fcf, ccy, [ocf, capex])
        put("fcf_margin_pct", _pct(fcf, cur["value"]), "percent", [ocf, capex, cur])
    ltd, cash = pick["long_term_debt"], pick["cash"]
    flags: list[str] = []
    if ltd and cash:
        put("net_debt_proxy", ltd["value"] - cash["value"], ccy, [ltd, cash])
    elif cash and "long_term_debt" in fin["not_found"]:
        # No long-term-debt tag in ANY year: usually a debt-free filer, but absence is not proof. Assume zero, flag it.
        put("net_debt_proxy", -cash["value"], ccy, [cash])
        flags.append("debt_assumed_zero")

    stale = (today - date.fromisoformat(cur["period_end"])).days > STALE_DAYS
    return {
        "ticker": ticker.upper(),
        "company": fin["company"],
        "currency": ccy,
        "fiscal_year_label": fy,
        "period_end": cur["period_end"],
        "stale": stale,
        "flags": flags,
        "source": f"edgar:comps:{ticker.upper()}:{cur['period_end']}",
        "metrics": metrics,
        "missing": [m for m in ("revenue_growth_pct", "gross_margin_pct", "operating_margin_pct", "net_margin_pct",
                                "fcf_margin_pct", "net_debt_proxy") if m not in metrics],
    }


def get_comps(tickers: list[str], today: date | None = None) -> dict:
    today = today or date.today()
    rows, errors = [], {}
    for t in tickers:
        try:
            rows.append(_company(t, today))
        except Exception as e:  # one bad ticker must not sink the table
            errors[t.upper()] = f"{type(e).__name__}: {e}"
    medians = {}
    for name in ("revenue_growth_pct", "gross_margin_pct", "operating_margin_pct", "net_margin_pct", "fcf_margin_pct"):
        used = [r for r in rows if not r["stale"] and name in r["metrics"]]
        if used:
            medians[name] = {"value": round(st.median(r["metrics"][name]["value"] for r in used), 2),
                             "unit": "percent", "n": len(used), "sources": [r["source"] for r in used]}
    return {"rows": rows, "medians": medians, "errors": errors, "notes": NOTES}
