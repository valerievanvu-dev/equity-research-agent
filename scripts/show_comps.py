"""Print the operating comps table for a sector or a list of tickers (no LLM involved).

    uv run python scripts/show_comps.py beauty
    uv run python scripts/show_comps.py ELF EL COTY
"""
import sys

from eqr.comps import UNIVERSES, get_comps, peer_universe

args = sys.argv[1:] or ["beauty"]
tickers = [c["ticker"] for c in peer_universe(args[0])["companies"]] if args[0] in UNIVERSES else args
out = get_comps(tickers)

COLS = [("revenue", "rev $B"), ("revenue_growth_pct", "growth%"), ("gross_margin_pct", "gross%"),
        ("operating_margin_pct", "op%"), ("net_margin_pct", "net%"), ("fcf_margin_pct", "fcf%")]
print(f"{'tkr':5} {'period_end':11} {'stale':5} " + " ".join(f"{h:>8}" for _, h in COLS))
for r in out["rows"]:
    cells = []
    for key, _ in COLS:
        v = r["metrics"].get(key, {}).get("value")
        cells.append("n/a" if v is None else (f"{v / 1e9:.2f}" if key == "revenue" else f"{v:.1f}"))
    print(f"{r['ticker']:5} {r['period_end']:11} {str(r['stale']):5} " + " ".join(f"{c:>8}" for c in cells))
print("medians:", {k: v["value"] for k, v in out["medians"].items()})
print("errors:", out["errors"])
