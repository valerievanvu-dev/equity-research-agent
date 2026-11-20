"""Idea generation: find names worth researching, and keep an inbox so ideas are not lost or re-proposed.

Stage-1 screen (see screen.py) gives every company in a universe five percentile signals. An idea is a company whose
signals fit a named theme, with the reason stated in numbers so you can disagree with it:

- value              : cheap versus its peers and not the most levered
- quality_compounder : high operating margin and above-median growth
- cash_machine       : top-tier free cash flow yield with moderate leverage
- growth_at_discount : growing faster than most peers without a premium multiple

Ideas are leads, not recommendations. They go to data/ideas.jsonl as `new`; research them with `--research N` (runs the
full agent, which sets the rating) or `eqr TICKER`. Companies already rated recently or on the watchlist are skipped.
"""
from __future__ import annotations

from datetime import date, timedelta

from . import comps, history, monitor, screen, sector, store, valuation

THEMES = {
    "value": lambda s, r: s["value"] is not None and s["value"] >= 0.75 and (s["leverage"] is None or s["leverage"] >= 0.25),
    "quality_compounder": lambda s, r: s["quality"] is not None and s["quality"] >= 0.75 and s["growth"] is not None and s["growth"] >= 0.5,
    "cash_machine": lambda s, r: s["fcf_yield"] is not None and s["fcf_yield"] >= 0.75 and (s["leverage"] is None or s["leverage"] >= 0.5),
    "growth_at_discount": lambda s, r: s["growth"] is not None and s["growth"] >= 0.7 and s["value"] is not None and s["value"] >= 0.5,
}
COVERED_DAYS = 45       # skip names rated within this many days
REPEAT_DAYS = 30        # do not re-propose a name that is already an open idea this recent
MIN_SCORE = 50.0


def ideas_path():
    return store.data_dir() / "ideas.jsonl"


def load() -> list[dict]:
    """Latest state of each idea (the file is append-only; a later line for the same id replaces an earlier one)."""
    latest: dict[str, dict] = {}
    for row in store.read_jsonl(ideas_path()):
        latest[row["id"]] = row
    return list(latest.values())


def _reason(theme: str, r: dict) -> str:
    f = lambda x, nd=1: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
    return {
        "value": f"cheap vs peers (P/E {f(r['pe'])}, EV/EBIT {f(r['ev_ebit'])}), value signal {r['signals']['value']:.2f}",
        "quality_compounder": f"operating margin {f(r['quality'])}% and revenue growth {f(r['growth'])}%, both top of the universe",
        "cash_machine": f"FCF yield {f(r['fcf_yield'])}% with net debt/EBIT {f(r['leverage'])}",
        "growth_at_discount": f"revenue growth {f(r['growth'])}% without a premium multiple (value signal {r['signals']['value']:.2f})",
    }[theme]


def caveats(r: dict, small: bool) -> list[str]:
    out = []
    if small:
        out.append("small universe: percentiles are coarse")
    if r["leverage"] is not None and r["leverage"] > sector.LEVERAGE_FLAG:
        out.append(f"leverage {r['leverage']:.1f}x net debt/EBIT")
    if "value_from_ev_sales" in r["notes"]:
        out.append("loss-making: valued on EV/Sales only")
    out += [x for x in r["flags"] if x.startswith(("single_source", "price_uncertain", "shares_from"))]
    return out


def generate(tickers: list[str], universe: str, today: date | None = None, *, valuation_fn=valuation.get_valuation,
             comps_fn=comps.get_comps, include_covered: bool = False, ratings: dict | None = None,
             watchlist: dict | None = None, existing: list[dict] | None = None) -> dict:
    """Return {'ideas': [...], 'skipped': [(ticker, why)], 'excluded': [...]} without writing anything."""
    today = today or date.today()
    s1 = screen.stage1(tickers, today, valuation_fn=valuation_fn, comps_fn=comps_fn)
    ratings = history.latest_by_ticker() if ratings is None else ratings
    watchlist = monitor.load_watchlist() if watchlist is None else watchlist
    existing = load() if existing is None else existing
    open_recent = {i["ticker"] for i in existing if i["status"] in ("new", "researched")
                   and date.fromisoformat(i["date"]) > today - timedelta(days=REPEAT_DAYS)}
    ideas, skipped = [], []
    for r in s1["rows"]:
        t = r["ticker"]
        themes = [k for k, test in THEMES.items() if test(r["signals"], r)]
        if not themes or r["score"] < MIN_SCORE:
            continue
        rated = ratings.get(t)
        if not include_covered and rated and (today - date.fromisoformat(rated["date"])).days <= COVERED_DAYS:
            skipped.append((t, f"already rated {rated['rating']} on {rated['date']}"))
        elif not include_covered and t in watchlist:
            skipped.append((t, "already on the watchlist"))
        elif t in open_recent:
            skipped.append((t, "already an open idea"))
        else:
            ideas.append({"id": f"{t}-{today.isoformat()}", "date": today.isoformat(), "ticker": t, "company": r["company"],
                          "universe": universe, "score": r["score"], "rank": r["rank"], "themes": themes,
                          "why": [_reason(k, r) for k in themes], "caveats": caveats(r, s1["small_universe"]),
                          "status": "new"})
    ideas.sort(key=lambda i: (-len(i["themes"]), -i["score"]))
    return {"ideas": ideas, "skipped": skipped, "excluded": s1["excluded"]}


def save(ideas: list[dict]) -> None:
    for i in ideas:
        store.append_jsonl(ideas_path(), i)


def set_status(ticker: str, status: str) -> int:
    """Mark the open ideas for a ticker (new/researched) as dismissed/watching/researched. Returns how many changed."""
    n = 0
    for i in load():
        if i["ticker"] == ticker.upper() and i["status"] in ("new", "researched"):
            store.append_jsonl(ideas_path(), {**i, "status": status})
            n += 1
    return n


def render(res: dict, today: date | None = None) -> str:
    today = today or date.today()
    L = [f"# Ideas ({today.isoformat()})", "", "> Research leads, not recommendations. A theme match is a reason to read the filings, "
         "not a reason to buy; `--research N` runs the full agent to rate the top names.", ""]
    for i in res["ideas"]:
        L += [f"## {i['ticker']} ({i['company']}), score {i['score']:.0f}, rank {i['rank']} in {i['universe']}",
              "Themes: " + ", ".join(i["themes"])] + [f"- {w}" for w in i["why"]]
        if i["caveats"]:
            L.append("- Caveats: " + "; ".join(i["caveats"]))
        L.append("")
    if not res["ideas"]:
        L += ["No new ideas matched a theme.", ""]
    if res["skipped"]:
        L += ["Skipped: " + "; ".join(f"{t} ({why})" for t, why in res["skipped"]), ""]
    return "\n".join(L)
