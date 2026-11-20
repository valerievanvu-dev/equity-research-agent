"""Stock monitor: a watchlist, and a check that turns "what changed?" into a short list of alerts.

    uv run eqr-track watch add ELF --below 90 --above 140
    uv run eqr-track watch run

What it looks at (all code, no model call):
- price triggers you set, and price versus the last rating (target reached, big drawdown, upside used up)
- new SEC filings since the last check (10-K / 10-Q = refresh the note; 8-K = read it)
- whether the last Buy would still clear the rating threshold at today's price
- the age of the rating, and the status of the thesis tracker if the ticker has a thesis

The first check of a ticker only records the filings it can see (a baseline); it never floods you with old filings.
Alerts carry a severity: high (act or read now), medium (look soon), info (context).
"""
from __future__ import annotations

from datetime import date

from . import edgar, history, live, recommend, store, thesis

PRICE_MOVE_PCT = 7.0         # move since the previous check that deserves a mention
DRAWDOWN_ALERT_PCT = -15.0   # vs the price when the rating was made
RUNUP_ALERT_PCT = 25.0
RATING_STALE_DAYS = 90
FORMS = ("10-K", "10-Q", "8-K")
FILINGS_FRESH_S = 3600


def watchlist_path():
    return store.data_dir() / "watchlist.json"


def state_path():
    return store.data_dir() / "monitor_state.json"


def load_watchlist() -> dict[str, dict]:
    return store.read_json(watchlist_path(), {})


def add(ticker: str, price_above: float | None = None, price_below: float | None = None, note: str = "",
        today: date | None = None) -> dict:
    wl = load_watchlist()
    t = ticker.upper()
    entry = wl.get(t, {"added": (today or date.today()).isoformat()})
    entry.update({"price_above": price_above if price_above is not None else entry.get("price_above"),
                  "price_below": price_below if price_below is not None else entry.get("price_below"),
                  "note": note or entry.get("note", "")})
    wl[t] = entry
    store.write_json(watchlist_path(), wl)
    return entry


def remove(ticker: str) -> bool:
    wl = load_watchlist()
    if wl.pop(ticker.upper(), None) is None:
        return False
    store.write_json(watchlist_path(), wl)
    return True


def fetch_filings(ticker: str, *, list_fn=edgar.list_filings) -> list[dict]:
    """Recent 10-K / 10-Q / 8-K filings, newest first. The filing index is re-fetched if older than an hour."""
    rows: list[dict] = []
    for form in FORMS:
        rows += list_fn(ticker, form, 5, FILINGS_FRESH_S)["filings"]
    return sorted(rows, key=lambda r: r["filed"], reverse=True)


def evaluate(ticker: str, snap: dict, *, entry: dict, rating: dict | None, filings: list[dict], state: dict,
             thesis_result: dict | None = None, today: date | None = None) -> tuple[list[dict], dict]:
    """Alerts for one ticker plus its updated state. Pure function: all inputs are passed in."""
    today = today or date.today()
    alerts: list[dict] = []
    add_ = lambda sev, kind, text: alerts.append({"severity": sev, "kind": kind, "text": text})  # noqa: E731
    price = snap.get("price")
    new_state = {"last_check": today.isoformat(), "seen_filings": sorted({f["accession"] for f in filings}),
                 "last_price": price if price is not None else state.get("last_price")}

    if price is None:
        add_("info", "no_price", "no verified price today (" + (", ".join(snap.get("flags", [])[:2]) or snap.get("error") or "unavailable") + ")")
    else:
        if entry.get("price_below") is not None and price <= entry["price_below"]:
            add_("high", "price_below", f"price ${price:,.2f} is at or below your ${entry['price_below']:,.2f} trigger")
        if entry.get("price_above") is not None and price >= entry["price_above"]:
            add_("high", "price_above", f"price ${price:,.2f} is at or above your ${entry['price_above']:,.2f} trigger")
        last = state.get("last_price")
        if last:
            mv = (price / last - 1) * 100
            if abs(mv) >= PRICE_MOVE_PCT:
                add_("medium", "price_move", f"price moved {mv:+.1f}% since the last check (${last:,.2f} -> ${price:,.2f})")

    if rating is None:
        add_("info", "no_rating", f"no rating on file: run `uv run eqr {ticker}`")
    else:
        age = (today - date.fromisoformat(rating["date"])).days
        if age > RATING_STALE_DAYS:
            add_("medium", "rating_stale", f"last rating ({rating['rating']}, {rating['date']}) is {age} days old")
        if price is not None:
            move = (price / rating["current_price"] - 1) * 100
            if rating["price_target"] and price >= rating["price_target"] > rating["current_price"]:
                add_("high", "target_reached", f"price ${price:,.2f} reached the ${rating['price_target']:,.2f} target")
            elif move <= DRAWDOWN_ALERT_PCT:
                add_("high", "drawdown", f"{move:+.1f}% since the {rating['date']} {rating['rating']} (price then ${rating['current_price']:,.2f})")
            elif move >= RUNUP_ALERT_PCT:
                add_("medium", "run_up", f"{move:+.1f}% since the {rating['date']} {rating['rating']}: much of the upside may be used up")
            if rating["rating"] == "Buy":
                rp = recommend.reprice(rating, price)
                if rp["rating"] != "Buy":
                    add_("medium", "buy_lapsed", f"the Buy no longer clears {recommend.BUY_AT:.0f}% at ${price:,.2f} "
                                                  f"(expected return now {rp['expected_total_return_pct']:+.1f}%, buy below ${rp['buy_below']:,.2f})")

    seen = set(state.get("seen_filings", []))
    if "last_check" in state:  # not the first check: report what is new
        for f in (x for x in filings if x["accession"] not in seen):
            if f["form"] in ("10-K", "10-Q"):
                add_("high", "new_filing", f"new {f['form']} filed {f['filed']} (period {f['period']}): refresh the note and thesis")
            else:
                add_("medium", "new_filing", f"new 8-K filed {f['filed']}: read it ({f['source']})")
    else:
        add_("info", "baseline", f"first check: recorded {len(filings)} recent filings as the baseline")

    if thesis_result and thesis_result["status"] != "Intact":
        add_("high" if thesis_result["status"] == "Broken" else "medium", "thesis",
             f"thesis {thesis_result['status'].upper()}: " + "; ".join(thesis_result["reasons"][:2]))
    return alerts, new_state


SEV = {"high": 0, "medium": 1, "info": 2}


def run(today: date | None = None, *, snapshot_fn=live.snapshots, filings_fn=fetch_filings, save_state: bool = True) -> dict:
    """Check every watchlist ticker. Returns {ticker: {alerts, price, rating}}."""
    today = today or date.today()
    wl = load_watchlist()
    if not wl:
        return {}
    state = store.read_json(state_path(), {})
    ratings = history.latest_by_ticker()
    snaps = snapshot_fn(list(wl), today)
    out: dict[str, dict] = {}
    for t, entry in wl.items():
        try:
            filings = filings_fn(t)
        except Exception as e:  # one SEC hiccup must not hide the price alerts
            filings = []
            filing_err = f"filings unavailable ({type(e).__name__})"
        else:
            filing_err = None
        th = thesis.load(t)
        tr = thesis.evaluate(th, snaps[t], today) if th else None
        if tr and snaps[t].get("price") is not None:
            thesis.record_check(th, tr)
        st_in = state.get(t, {})
        alerts, st_out = evaluate(t, snaps[t], entry=entry, rating=ratings.get(t), filings=filings, state=st_in,
                                  thesis_result=tr, today=today)
        if filing_err:
            alerts.append({"severity": "info", "kind": "filings_error", "text": filing_err})
            st_out["seen_filings"] = st_in.get("seen_filings", [])  # keep the old baseline: a failed fetch must not reset it
            if "last_check" not in st_in:
                st_out.pop("last_check")
        state[t] = st_out
        out[t] = {"alerts": sorted(alerts, key=lambda a: SEV[a["severity"]]), "price": snaps[t].get("price"),
                  "rating": ratings.get(t, {}).get("rating")}
    if save_state:
        store.write_json(state_path(), state)
    return out


def render(results: dict, today: date | None = None) -> str:
    today = today or date.today()
    L = [f"# Watchlist monitor ({today.isoformat()})", "", "> Research leads, not advice.", ""]
    if not results:
        return "\n".join(L + ["Watchlist is empty: `uv run eqr-track watch add TICKER`.", ""])
    ordered = sorted(results.items(), key=lambda kv: (min((SEV[a["severity"]] for a in kv[1]["alerts"]), default=3), kv[0]))
    for t, r in ordered:
        price = "no price" if r["price"] is None else f"${r['price']:,.2f}"
        L += [f"## {t}: {price}, rating {r['rating'] or 'none'}"]
        L += [f"- **{a['severity'].upper()}** {a['text']}" for a in r["alerts"]] or ["- nothing to report"]
        L.append("")
    return "\n".join(L)
