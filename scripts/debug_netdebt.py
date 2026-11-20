"""Why is net_debt_proxy missing? Shows the cash / long_term_debt rows by fiscal year.

    uv run python scripts/debug_netdebt.py IPAR
"""
import sys

from eqr import edgar

for t in sys.argv[1:]:
    fin = edgar.get_financials(t, ["revenue", "cash", "long_term_debt"], years=3)
    print(t, "not_found:", fin["not_found"])
    for m in ("revenue", "cash", "long_term_debt"):
        print("  ", m, [(r["fiscal_year"], r["period_end"], r["concept"].replace("CashAndCashEquivalentsAtCarryingValue", "Cash..")) for r in fin["metrics"].get(m, [])])
