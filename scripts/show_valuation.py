"""Print valuation multiples for a sector or tickers (no agent LLM; uses Parallel + a small extraction model).

    uv run python scripts/show_valuation.py beauty
    uv run python scripts/show_valuation.py ELF EL COTY
"""
import sys

from eqr.comps import UNIVERSES, peer_universe
from eqr.valuation import get_valuation

args = sys.argv[1:] or ["beauty"]
tickers = [c["ticker"] for c in peer_universe(args[0])["companies"]] if args[0] in UNIVERSES else args
out = get_valuation(tickers)

COLS = [("market_cap", "mcap $B"), ("ev_sales", "EV/S"), ("ev_ebit", "EV/EBIT"), ("pe", "P/E"), ("fcf_yield_pct", "FCFy%")]
print(f"{'tkr':5} {'price':>8} {'price date':11} " + " ".join(f"{h:>8}" for _, h in COLS) + "  flags")
for r in out["rows"]:
    cells = []
    for k, _ in COLS:
        v = r["metrics"].get(k, {}).get("value")
        cells.append("n/a" if v is None else (f"{v / 1e9:.2f}" if k == "market_cap" else f"{v:.1f}"))
    price = f"{r['price']:.2f}" if r["price"] else "n/a"
    print(f"{r['ticker']:5} {price:>8} {str(r['price_date']):11} " + " ".join(f"{c:>8}" for c in cells)
          + ("  WITHHELD " if r["withheld"] else "  ") + ", ".join(r["flags"]))
print("medians:", {k: v["value"] for k, v in out["medians"].items()})
print("errors:", out["errors"])
