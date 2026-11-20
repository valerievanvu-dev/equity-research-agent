"""SEC EDGAR access: XBRL company facts, filing lists, filing text.

Everything is cached on disk. That is polite to the SEC and it makes evals
reproducible (delete .cache/ to refresh).
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import httpx

CACHE_DIR = Path(os.environ.get("EQR_CACHE", ".cache/edgar"))
MIN_INTERVAL = 0.15  # SEC allows 10 req/s; stay well under
_last_call = 0.0

# friendly name -> (fact kind, candidate us-gaap concepts in priority order)
METRICS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("duration", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"]),
    "gross_profit": ("duration", ["GrossProfit"]),
    "operating_income": ("duration", ["OperatingIncomeLoss"]),
    "net_income": ("duration", ["NetIncomeLoss"]),
    "eps_diluted": ("duration", ["EarningsPerShareDiluted"]),
    "dps": ("duration", ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"]),
    "operating_cash_flow": ("duration", ["NetCashProvidedByUsedInOperatingActivities"]),
    "capex": ("duration", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "total_assets": ("instant", ["Assets"]),
    "total_liabilities": ("instant", ["Liabilities"]),
    "shareholders_equity": ("instant", ["StockholdersEquity"]),
    "cash": ("instant", ["CashAndCashEquivalentsAtCarryingValue",
                         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"]),
    "long_term_debt": ("instant", ["LongTermDebtNoncurrent", "LongTermDebt"]),
}

# Foreign private issuers file 20-F / 40-F (usually IFRS, in their reporting currency) instead of the 10-K.
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F", "40-F/A")

# Same friendly metric names, IFRS (ifrs-full) concepts. Used when a company has no US-GAAP revenue data.
METRICS_IFRS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("duration", ["Revenue", "RevenueFromContractsWithCustomers"]),
    "gross_profit": ("duration", ["GrossProfit"]),
    "operating_income": ("duration", ["ProfitLossFromOperatingActivities"]),
    "net_income": ("duration", ["ProfitLossAttributableToOwnersOfParent", "ProfitLoss"]),
    "eps_diluted": ("duration", ["DilutedEarningsLossPerShare"]),
    "dps": ("duration", ["DividendsRecognisedAsDistributionsToOwnersPerShare"]),  # often absent: derived below
    "operating_cash_flow": ("duration", ["CashFlowsFromUsedInOperatingActivities"]),
    "capex": ("duration", ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"]),
    "total_assets": ("instant", ["Assets"]),
    "total_liabilities": ("instant", ["Liabilities"]),
    "shareholders_equity": ("instant", ["EquityAttributableToOwnersOfParent", "Equity"]),
    "cash": ("instant", ["CashAndCashEquivalents"]),
    "long_term_debt": ("instant", ["NoncurrentPortionOfNoncurrentBondsIssued", "NoncurrentBorrowings"]),
}
DIVIDENDS_PAID_IFRS = ["DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities", "DividendsPaid"]
WEIGHTED_SHARES_IFRS = ["WeightedAverageShares", "AdjustedWeightedAverageShares"]


# ---------------------------------------------------------------- http + cache
def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT to 'Your Name your@email.com'. The SEC requires "
            "automated clients to identify themselves."
        )
    return ua


def _fetch(url: str, cache_name: str, *, as_json: bool):
    global _last_call
    path = CACHE_DIR / cache_name
    if path.exists():
        raw = path.read_text()
        return json.loads(raw) if as_json else raw
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    r = httpx.get(
        url,
        headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=60,
        follow_redirects=True,
    )
    _last_call = time.time()
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.json() if as_json else r.text


def lookup(ticker: str) -> tuple[int, str]:
    """ticker -> (cik, company name)"""
    data = _fetch("https://www.sec.gov/files/company_tickers.json", "company_tickers.json", as_json=True)
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"]), row["title"]
    raise ValueError(f"Unknown ticker: {ticker}")


def _cik10(cik: int) -> str:
    return f"{cik:010d}"


# ------------------------------------------------------------------ financials
def select_annual(entries: list[dict], kind: str) -> dict[int, dict]:
    """Pick one entry per fiscal year from raw XBRL facts.

    A 10-K reports the current year AND prior-year comparatives, all tagged with
    the filing's fiscal year. The current period is the one with the latest end
    date. Getting this wrong silently returns last year's number.
    """
    best: dict[int, dict] = {}
    for e in entries:
        if e.get("form") not in ANNUAL_FORMS or e.get("fp") != "FY" or e.get("fy") is None:
            continue
        if kind == "duration":
            if "start" not in e:
                continue
            days = (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
            if not 340 <= days <= 380:  # skip quarterly / multi-year facts
                continue
        cur = best.get(e["fy"])
        if cur is None or (e["end"], e["filed"]) > (cur["end"], cur["filed"]):
            best[e["fy"]] = e
    return best


def _currency(unit: str) -> str:
    """'EUR' -> EUR, 'EUR/shares' -> EUR, 'USD' -> USD."""
    return unit.split("/")[0].upper()


def _derive_dps(ifrs: dict, years: int, ticker: str) -> list[dict]:
    """Approximate dividends per share = cash dividends paid / weighted average shares, when no DPS concept exists."""
    div = next((ifrs[c] for c in DIVIDENDS_PAID_IFRS if c in ifrs), None)
    sh = next((ifrs[c] for c in WEIGHTED_SHARES_IFRS if c in ifrs), None)
    if not div or not sh:
        return []
    div_by: dict[int, dict] = {}
    for unit, entries in div["units"].items():
        for fy, e in select_annual(entries, "duration").items():
            div_by.setdefault(fy, {**e, "_unit": unit})
    sh_by: dict[int, dict] = {}
    for entries in sh["units"].values():
        for fy, e in select_annual(entries, "duration").items():
            sh_by.setdefault(fy, e)
    rows = []
    for fy in sorted(set(div_by) & set(sh_by), reverse=True)[:years]:
        d, w = div_by[fy], sh_by[fy]
        if not w["val"]:
            continue
        rows.append({"fiscal_year": fy, "period_end": d["end"], "value": abs(d["val"]) / w["val"],
                     "unit": f"{_currency(d['_unit'])}/shares", "concept": "derived:cash_dividends_paid/weighted_shares",
                     "source": f"edgar:facts:{ticker.upper()}:DividendsPaid:FY{fy}:{d['accn']}"})
    return rows


def get_financials(ticker: str, metrics: list[str] | None = None, years: int = 5) -> dict:
    cik, name = lookup(ticker)
    all_facts = _fetch(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik10(cik)}.json",
        f"facts-{cik}.json",
        as_json=True,
    )["facts"]
    us, ifrs = all_facts.get("us-gaap", {}), all_facts.get("ifrs-full", {})
    # US-GAAP when the company reports it; otherwise IFRS (foreign private issuers filing 20-F / 40-F)
    use_ifrs = bool(ifrs) and not any(c in us for c in METRICS["revenue"][1])
    facts, spec = (ifrs, METRICS_IFRS) if use_ifrs else (us, METRICS)
    wanted = metrics or list(METRICS)
    out: dict[str, list[dict]] = {}
    missing: list[str] = []
    notes: list[str] = []
    for m in wanted:
        if m not in METRICS:
            raise ValueError(f"Unknown metric '{m}'. Valid: {sorted(METRICS)}")
        kind, concepts = spec[m]
        by_fy: dict[int, dict] = {}
        for concept in concepts:
            node = facts.get(concept)
            if not node:
                continue
            for unit, entries in node["units"].items():
                for fy, e in select_annual(entries, kind).items():
                    by_fy.setdefault(fy, {**e, "_unit": unit, "_concept": concept})
        if not by_fy and use_ifrs and m == "dps":
            derived = _derive_dps(ifrs, years, ticker)
            if derived:
                out[m] = derived
                notes.append("dps is derived: cash dividends paid / weighted average shares (no per-share concept reported).")
                continue
        if not by_fy:
            missing.append(m)
            continue
        rows = []
        for fy in sorted(by_fy, reverse=True)[:years]:
            e = by_fy[fy]
            rows.append(
                {
                    "fiscal_year": fy,
                    "period_end": e["end"],
                    "value": e["val"],
                    "unit": e["_unit"],
                    "concept": e["_concept"],
                    "source": f"edgar:facts:{ticker.upper()}:{e['_concept']}:FY{fy}:{e['accn']}",
                }
            )
        out[m] = rows
    rev = out.get("revenue") or next(iter(out.values()), [])
    currency = _currency(rev[0]["unit"]) if rev else "USD"
    if use_ifrs:
        notes.append(f"IFRS filer (Form 20-F/40-F): values are in {currency}, not USD.")
    return {"ticker": ticker.upper(), "company": name, "taxonomy": "ifrs-full" if use_ifrs else "us-gaap",
            "currency": currency, "metrics": out, "not_found": missing, "notes": notes}


def get_shares_outstanding(ticker: str) -> dict:
    """Latest cover-page share count (dei:EntityCommonStockSharesOutstanding).

    Multi-class issuers report one value per class on the same filing; we sum distinct values and flag it.
    Callers should sanity-check the resulting market cap, since dimensional facts can be missing.
    """
    cik, _ = lookup(ticker)
    facts = _fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik10(cik)}.json", f"facts-{cik}.json",
                   as_json=True)["facts"].get("dei", {})
    node = facts.get("EntityCommonStockSharesOutstanding")
    if not node:
        return _shares_from_weighted_average(ticker, cik)
    entries = [e for es in node["units"].values() for e in es]
    latest = max(entries, key=lambda e: (e["end"], e["filed"]))
    same = {e["val"] for e in entries if e["accn"] == latest["accn"] and e["end"] == latest["end"]}
    return {
        "shares": float(sum(same)),
        "as_of": latest["end"],
        "multi_class": len(same) > 1,
        "basis": "cover_page",
        "source": f"edgar:facts:{ticker.upper()}:dei:EntityCommonStockSharesOutstanding:{latest['accn']}",
    }


def _shares_from_weighted_average(ticker: str, cik: int) -> dict:
    """Fallback for issuers (often dual-class) with no usable cover-page count: latest annual diluted
    weighted-average shares. Less current than the cover page, so callers must flag it."""
    all_facts = _fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik10(cik)}.json", f"facts-{cik}.json",
                       as_json=True)["facts"]
    node = all_facts.get("us-gaap", {}).get("WeightedAverageNumberOfDilutedSharesOutstanding") or next(
        (all_facts.get("ifrs-full", {})[c] for c in WEIGHTED_SHARES_IFRS if c in all_facts.get("ifrs-full", {})), None)
    if not node:
        raise ValueError("no cover-page or weighted-average share count in XBRL")
    best = None
    for entries in node["units"].values():
        for fy, e in select_annual(entries, "duration").items():
            if best is None or (e["end"], e["filed"]) > (best["end"], best["filed"]):
                best = e
    if best is None:
        raise ValueError("no annual weighted-average share count in XBRL")
    return {"shares": float(best["val"]), "as_of": best["end"], "multi_class": False, "basis": "weighted_avg_diluted",
            "source": f"edgar:facts:{ticker.upper()}:WeightedAverageNumberOfDilutedSharesOutstanding:{best['accn']}"}


# --------------------------------------------------------------------- filings
def _submissions(cik: int) -> dict:
    return _fetch(f"https://data.sec.gov/submissions/CIK{_cik10(cik)}.json", f"subs-{cik}.json", as_json=True)


def list_filings(ticker: str, form: str = "10-K", limit: int = 5) -> dict:
    cik, name = lookup(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    rows = []
    for i, f in enumerate(recent["form"]):
        if f == form:
            acc = recent["accessionNumber"][i]
            rows.append(
                {
                    "accession": acc,
                    "form": f,
                    "filed": recent["filingDate"][i],
                    "period": recent["reportDate"][i],
                    "source": f"edgar:filing:{ticker.upper()}:{acc}",
                }
            )
        if len(rows) >= limit:
            break
    return {"ticker": ticker.upper(), "company": name, "filings": rows}


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
    SKIP = {"script", "style", "ix:header"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    p.close()
    text = "".join(p.out).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


_SEP = r"\s*[\.\:\-–—]?\s*"
SECTION_START = {
    "business": rf"^\s*item\s*1{_SEP}business\b",
    "risk_factors": rf"^\s*item\s*1a{_SEP}risk\s+factors",
    "mdna": rf"^\s*item\s*7{_SEP}management",
    "financial_statements": rf"^\s*item\s*8{_SEP}(consolidated\s+)?financial\s+statements",
}
SECTION_END = {
    "business": r"^\s*item\s*1a\b",
    "risk_factors": r"^\s*item\s*1b\b",
    "mdna": r"^\s*item\s*7a\b",
    "financial_statements": r"^\s*item\s*9\b",
}
SECTIONS = ["business", "risk_factors", "mdna", "financial_statements", "full"]


def extract_section(text: str, section: str) -> str | None:
    """Heuristic 10-K section splitter.

    The table of contents and cross-references also match the heading regex, so
    we take the longest span that has a proper end marker.
    """
    if section == "full":
        return text
    start, end = re.compile(SECTION_START[section], re.I | re.M), re.compile(SECTION_END[section], re.I | re.M)
    best = ""
    for m in start.finditer(text):
        e = end.search(text, m.end())
        if not e:
            continue
        chunk = text[m.start() : e.start()]
        if len(chunk) > len(best):
            best = chunk
    return best or None


def read_filing_section(
    ticker: str, accession: str, section: str = "risk_factors", offset: int = 0, max_chars: int = 12000
) -> dict:
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {SECTIONS}")
    max_chars = min(max_chars, 20000)
    cik, _ = lookup(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    if accession not in recent["accessionNumber"]:
        raise ValueError(f"Accession {accession} not found in recent filings for {ticker}; use list_filings")
    doc = recent["primaryDocument"][recent["accessionNumber"].index(accession)]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}"
    text_path = CACHE_DIR / f"text-{cik}-{accession}.txt"
    if text_path.exists():
        text = text_path.read_text()
    else:
        text = html_to_text(_fetch(url, f"raw-{cik}-{accession}.html", as_json=False))
        text_path.write_text(text)
    body = extract_section(text, section)
    if body is None:
        raise ValueError(f"Could not locate section '{section}'. Try section='full' and page with offset.")
    chunk = body[offset : offset + max_chars]
    return {
        "source": f"edgar:filing:{ticker.upper()}:{accession}:{section}",
        "section": section,
        "offset": offset,
        "total_chars": len(body),
        "has_more": offset + max_chars < len(body),
        "text": chunk,
    }
