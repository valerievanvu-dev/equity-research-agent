"""Web search via Parallel's Search API (POST https://api.parallel.ai/v1/search).

Used for what filings can't tell you: news, catalysts, competitor moves, analyst context.
Responses are cached on disk so repeated identical queries are free and fast. The cache EXPIRES: agent-initiated
searches default to 6 hours (set EQR_SEARCH_TTL_HOURS; 0 means never expire, which is what you want for reproducible
evals), so a query repeated on a later day is searched again instead of returning stale news. Callers can pass an
explicit cache_ttl_s: the price and market-cap lookups use 15 minutes; None means never expire.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

URL = "https://api.parallel.ai/v1/search"
CACHE_DIR = Path(os.environ.get("EQR_CACHE", ".cache/edgar")).parent / "web"
MODES = ["fast", "advanced"]
_DEFAULT_TTL = object()  # sentinel: "use the configured default expiry"


def default_ttl_s() -> float | None:
    """Default cache expiry for agent searches: EQR_SEARCH_TTL_HOURS (default 6); 0 or less means never expire."""
    hours = float(os.environ.get("EQR_SEARCH_TTL_HOURS", "6"))
    return None if hours <= 0 else hours * 3600


def _api_key() -> str:
    key = os.environ.get("PARALLEL_API_KEY")
    if not key:
        raise RuntimeError("PARALLEL_API_KEY is not set in this process's environment")
    return key


def web_search(
    objective: str,
    search_queries: list[str],
    mode: str = "fast",
    max_results: int = 5,
    max_chars_total: int = 8000,
    after_date: str | None = None,
    include_domains: list[str] | None = None,
    max_age_seconds: int | None = None,
    cache_ttl_s: float | None = _DEFAULT_TTL,  # type: ignore[assignment]
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not search_queries:
        raise ValueError("provide at least one search query (3-6 keywords each)")

    advanced: dict = {"max_results": min(max_results, 10)}
    policy = {}
    if after_date:
        policy["after_date"] = after_date  # YYYY-MM-DD
    if include_domains:
        policy["include_domains"] = include_domains
    if policy:
        advanced["source_policy"] = policy
    if max_age_seconds:  # ask Parallel for pages no older than this (API minimum is 600s)
        advanced["fetch_policy"] = {"max_age_seconds": max(600, max_age_seconds)}
    # max_results / source_policy / fetch_policy live under advanced_settings; top level returns 422
    body = {
        "objective": objective,
        "search_queries": search_queries[:3],
        "mode": mode,
        "max_chars_total": min(max_chars_total, 20000),
        "advanced_settings": advanced,
    }

    cache = CACHE_DIR / (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20] + ".json")
    ttl = default_ttl_s() if cache_ttl_s is _DEFAULT_TTL else cache_ttl_s
    fresh = cache.exists() and (ttl is None or time.time() - cache.stat().st_mtime < ttl)
    if fresh:
        data = json.loads(cache.read_text())
    else:
        r = httpx.post(URL, headers={"x-api-key": _api_key(), "Content-Type": "application/json"},
                       json=body, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Parallel search failed {r.status_code}: {r.text[:300]}")
        data = r.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))

    return {
        "untrusted_content": True,  # web text is data, never instructions
        "results": [
            {
                "title": x.get("title"),
                "url": x["url"],
                "publish_date": x.get("publish_date"),
                "excerpts": x.get("excerpts", []),
                "source": f"web:{x['url']}",
            }
            for x in data.get("results", [])
        ],
        "warnings": [w.get("message") for w in (data.get("warnings") or [])],
    }
