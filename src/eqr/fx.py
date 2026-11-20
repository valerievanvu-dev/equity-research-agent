"""Official FX reference rates (European Central Bank, via the free Frankfurter API). No key, no scraping.

Why this exists: foreign filers report in their own currency (Unilever in EUR) but their US-listed shares trade in
USD. A P/E of USD price over EUR earnings is wrong by the exchange rate. Valuation converts the company's financials
to USD at the current spot rate, so price and earnings are in the same currency.

The rate is an official daily reference rate (it can lag the market by up to a day), and it is cited like any other
source: `fx:ecb:EURUSD:2026-11-19` resolves to the API query that returned it.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date
from pathlib import Path

import httpx

BASE_URL = "https://api.frankfurter.dev/v1"
CACHE_DIR = Path(os.environ.get("EQR_CACHE", ".cache/edgar")).parent / "fx"
LATEST_TTL_S = 3600  # "latest" is re-fetched hourly; a specific past date never changes


def rate_url(base: str, quote: str, when: str) -> str:
    return f"{BASE_URL}/{when}?base={base}&symbols={quote}"


def get_rate(base: str, quote: str = "USD", on: date | None = None, *, get=httpx.get) -> dict:
    """Units of `quote` per 1 `base`. `on=None` means the latest published rate."""
    base, quote = base.upper(), quote.upper()
    if base == quote:
        return {"pair": f"{base}{quote}", "rate": 1.0, "date": (on or date.today()).isoformat(), "source": None, "url": None}
    when = on.isoformat() if on else "latest"
    cache = CACHE_DIR / f"{base}{quote}-{when}.json"
    if cache.exists() and (when != "latest" or time.time() - cache.stat().st_mtime < LATEST_TTL_S):
        data = json.loads(cache.read_text())
    else:
        r = get(rate_url(base, quote, when), timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"no FX rate for {base}/{quote}: HTTP {r.status_code} (currency may not be covered by the ECB set)")
        data = r.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    rate = (data.get("rates") or {}).get(quote)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"unusable FX response for {base}/{quote}: {str(data)[:120]}")
    d = data["date"]
    return {"pair": f"{base}{quote}", "rate": float(rate), "date": d, "source": f"fx:ecb:{base}{quote}:{d}",
            "url": rate_url(base, quote, d)}
