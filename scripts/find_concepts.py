"""List XBRL concepts matching a keyword with their latest annual period: debug missing/stale metrics.

    uv run python scripts/find_concepts.py IPAR Revenue
"""
import json
import sys

from eqr import edgar

ticker, keyword = sys.argv[1], sys.argv[2]
cik, _ = edgar.lookup(ticker)
edgar.get_financials(ticker, ["revenue"], years=1)  # ensures the facts file is cached
facts = json.load(open(edgar.CACHE_DIR / f"facts-{cik}.json"))["facts"]["us-gaap"]
for name, node in sorted(facts.items()):
    if keyword.lower() in name.lower():
        for unit, entries in node["units"].items():
            ann = [e for e in entries if e.get("form") in edgar.ANNUAL_FORMS and e.get("fp") == "FY"]
            if ann:
                print(f"{name:75} {unit:8} latest FY-reported end {max(e['end'] for e in ann)}")
