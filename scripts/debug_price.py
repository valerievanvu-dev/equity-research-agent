"""Show the verified price candidates behind one ticker's price (why was it conflicting / withheld?).

    uv run python scripts/debug_price.py COTY
"""
import json
import sys

from eqr import edgar, prices

t = sys.argv[1].upper()
r = prices.get_price(t, edgar.lookup(t)[1])
print(json.dumps({k: r.get(k) for k in ("price", "as_of", "usable", "flags", "market_cap_reported", "candidates")}, indent=1))
