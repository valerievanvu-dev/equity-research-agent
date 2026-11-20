"""Check which tickers resolve on EDGAR and have usable XBRL financials.

    uv run python scripts/check_universe.py EL COTY ELF
"""
import sys

from eqr import edgar

METRICS = ["revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex", "cash",
           "long_term_debt"]

for t in sys.argv[1:]:
    try:
        r = edgar.get_financials(t, METRICS, years=2)
        rev = r["metrics"].get("revenue", [])
        latest = (rev[0]["fiscal_year"], rev[0]["period_end"], round(rev[0]["value"] / 1e9, 2)) if rev else None
        print(f"{t:5} {r['company'][:34]:34} rev(FY,end,$B)={latest} missing={r['not_found']}")
    except Exception as e:
        print(f"{t:5} ERROR {type(e).__name__}: {str(e)[:80]}")
