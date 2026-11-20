"""Buy recommendations: turn the ratings log into a current, price-aware Buy list.

A rating is made at one price on one day. A week later the stock has moved, so "Buy" may no longer be true. This module
takes the latest rating per ticker (data/ratings_history.jsonl), re-prices the *stored* target at today's price with the
same fixed rule the agent uses (target.rating_for: Buy at >= +15% expected total return), and sorts names into:

- Buy now         : still a Buy at today's price, normal conviction, rating fresh, thesis not broken
- Buy, low conv.  : same, but flagged low conviction (the model leaned on weak anchors)
- Wait for a dip  : was a Buy, price rose, and it no longer clears +15% (shows the "buy below" price)
- Refresh first   : the rating is older than max_age_days; re-run the note before acting
- Review by hand  : implausible return, or the thesis tracker says Broken

Only the price is refreshed; the target and scenarios come from the note. Output is a research lead, not personal advice:
it knows nothing about your portfolio, risk tolerance or taxes, and it sizes nothing.
"""
from __future__ import annotations

from datetime import date

from . import history, live, thesis
from .target import BUY_AT, rating_for

PLAUSIBLE_RETURN_PCT = 100.0
MAX_AGE_DAYS = 45


def dividend_yield_pct(r: dict) -> float:
    """Dividend yield implied by the logged row: expected total return minus the price-only return to target."""
    return r["expected_total_return_pct"] - (r["price_target"] / r["current_price"] - 1) * 100


def reprice(r: dict, price: float) -> dict:
    """Re-run the rating rule for a logged rating at a new price (same target, same dividend yield)."""
    dy = dividend_yield_pct(r)
    er = (r["price_target"] / price - 1) * 100 + dy
    return {"price": price, "expected_total_return_pct": round(er, 1), "rating": rating_for(er),
            "buy_below": round(r["price_target"] / (1 + (BUY_AT - dy) / 100), 2)}


def build(today: date | None = None, *, ratings: dict[str, dict] | None = None, snaps: dict[str, dict] | None = None,
          theses: dict[str, dict] | None = None, max_age_days: int = MAX_AGE_DAYS) -> dict[str, list[dict]]:
    """Buckets of recommendation rows. snaps=None keeps the logged price (and says so); pass live snapshots to re-price."""
    today = today or date.today()
    ratings = history.latest_by_ticker() if ratings is None else ratings
    out: dict[str, list[dict]] = {k: [] for k in ("buy", "buy_low", "wait", "refresh", "review")}
    for t, r in ratings.items():
        if r["rating"] != "Buy":
            continue
        age = (today - date.fromisoformat(r["date"])).days
        snap = (snaps or {}).get(t) or {}
        live_price = snap.get("price")
        price = live_price if live_price else r["current_price"]
        rp = reprice(r, price)
        row = {"ticker": t, "company": r["company"], "rating_date": r["date"], "age_days": age, "price_target": r["price_target"],
               "price": price, "price_is_live": bool(live_price), "price_note": "" if live_price or snaps is None else "no verified price today; logged price used",
               "logged_price": r["current_price"], "conviction": r["conviction"], "method": r["method"], "flags": r.get("flags", []),
               "expected_total_return_pct": rp["expected_total_return_pct"], "buy_below": rp["buy_below"], "still_buy": rp["rating"] == "Buy",
               "thesis_status": (theses or {}).get(t, {}).get("status")}
        if abs(rp["expected_total_return_pct"]) > PLAUSIBLE_RETURN_PCT or row["thesis_status"] == "Broken":
            out["review"].append(row)
        elif age > max_age_days:
            out["refresh"].append(row)
        elif not rp["rating"] == "Buy":
            out["wait"].append(row)
        elif r["conviction"] == "Low":
            out["buy_low"].append(row)
        else:
            out["buy"].append(row)
    for k in out:
        out[k].sort(key=lambda x: -x["expected_total_return_pct"])
    return out


def run(today: date | None = None, *, refresh_prices: bool = False, max_age_days: int = MAX_AGE_DAYS,
        snapshot_fn=live.snapshots) -> dict[str, list[dict]]:
    today = today or date.today()
    ratings = history.latest_by_ticker()
    buys = [t for t, r in ratings.items() if r["rating"] == "Buy"]
    snaps = snapshot_fn(buys, today) if refresh_prices and buys else None
    ths = {}
    for t in buys:
        th = thesis.load(t)
        if th and snaps and snaps.get(t, {}).get("price") is not None:
            ths[t] = thesis.evaluate(th, snaps[t], today)
    return build(today, ratings=ratings, snaps=snaps, theses=ths, max_age_days=max_age_days)


def render(b: dict[str, list[dict]], today: date | None = None, max_age_days: int = MAX_AGE_DAYS) -> str:
    today = today or date.today()

    def table(rows, extra=""):
        L = ["| ticker | rating date | price | target | expected return | buy below | conviction | method | flags |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['ticker']} | {r['rating_date']} ({r['age_days']}d) | ${r['price']:,.2f}{'' if r['price_is_live'] else '*'} | "
                     f"${r['price_target']:,.2f} | {r['expected_total_return_pct']:+.1f}% | ${r['buy_below']:,.2f} | {r['conviction']} | "
                     f"{r['method']} | {', '.join(r['flags'][:3]) or '-'} |")
        return L + ([extra] if extra else [])

    L = [f"# Buy recommendations ({today.isoformat()})", "",
         f"> **Research leads, not personal advice.** Each row re-prices the target from the latest note at today's price using the fixed "
         f"rule (Buy at >= +{BUY_AT:.0f}% expected total return). Nothing here knows your portfolio, risk or taxes, and nothing is sized. "
         "Verify against the filings. `*` = logged price from the rating date, not refreshed.", ""]
    sections = [("Buy now", "buy", "Still clears the threshold at the current price, normal conviction, rating fresh."),
                ("Buy, low conviction", "buy_low", "Clears the threshold, but the note flagged weak anchors. Treat with extra caution."),
                ("Wait for a dip", "wait", "Was a Buy; the price has risen so it no longer clears the threshold. 'Buy below' is the price at which it would again."),
                ("Refresh the note first", "refresh", f"Rating older than {max_age_days} days: re-run `eqr TICKER` before acting."),
                ("Review by hand", "review", f"Return beyond +/-{PLAUSIBLE_RETURN_PCT:.0f}% (usually bad inputs) or thesis broken.")]
    for title, key, blurb in sections:
        L += [f"## {title}", "", f"_{blurb}_", ""]
        L += table(b[key]) if b[key] else ["none"]
        L.append("")
    return "\n".join(L)
