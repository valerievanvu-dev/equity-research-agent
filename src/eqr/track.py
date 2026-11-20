"""eqr-track: the tracking side of the agent. Everything here is code on SEC + price data; none of it calls a model
except `ideas run --research`, which runs the full analyst agent on the names it proposes.

    eqr-track thesis add ELF [--pillar "operating_margin_pct >= 17"] | check [TICKER ...] | list
    eqr-track sector list | add NAME T1 T2 ... | snapshot beauty
    eqr-track watch add ELF --below 90 --above 140 | remove ELF | list | run
    eqr-track ideas run beauty [--research 2] | list | dismiss TICKER | watch TICKER
    eqr-track buys [--refresh-prices]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import ideas, live, monitor, recommend, screen, sector, thesis
from .schemas import ResearchNote


def _save_report(name: str, text: str) -> Path:
    p = Path("reports") / f"{name}-{date.today().isoformat()}.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text(text)
    return p


# ------------------------------------------------------------------ thesis
def cmd_thesis(a) -> None:
    if a.action == "list":
        for th in thesis.load_all():
            last = th["checks"][-1] if th.get("checks") else None
            print(f"{th['ticker']:6} {th.get('rating') or '-':5} target ${th.get('price_target') or 0:,.2f}  "
                  f"updated {th['updated']}  last check: {last['status'] + ' ' + last['date'] if last else 'never'}")
        return
    if a.action == "add":
        t = a.tickers[0].upper()
        trace = screen.find_recent_run(t, 24 * 365)
        if not trace:
            sys.exit(f"no saved run with a note for {t}: run `uv run eqr {t}` first")
        extra = [thesis.parse_pillar(p, "broken") for p in a.pillar] + [thesis.parse_pillar(p, "watch") for p in a.watch_pillar]
        th = thesis.create(ResearchNote.model_validate(trace["note"]), trace, extra)
        print(f"thesis saved for {t} with {len(th['pillars'])} pillars:")
        for p in th["pillars"]:
            print(f"  [{p['on_fail']}] {p['label']}")
        return
    tickers = [t.upper() for t in a.tickers] or [th["ticker"] for th in thesis.load_all()]
    if not tickers:
        sys.exit("no theses yet: `uv run eqr-track thesis add TICKER`")
    snaps = live.snapshots(tickers)
    out = []
    for t in tickers:
        th = thesis.load(t)
        if not th:
            print(f"{t}: no thesis (run `eqr-track thesis add {t}`)")
            continue
        res = thesis.evaluate(th, snaps[t])
        if snaps[t]["price"] is not None:
            thesis.record_check(th, res)
        out.append(thesis.render(th, res))
    text = "# Thesis check\n\n> Research leads, not advice.\n\n" + "\n".join(out)
    print(text)
    print(f"saved: {_save_report('thesis', text)}")


# ------------------------------------------------------------------ sector
def cmd_sector(a) -> None:
    unis = sector.all_universes()
    if a.action == "list":
        for k, v in sorted(unis.items()):
            print(f"{k}: {', '.join(c['ticker'] for c in v)}")
        return
    if a.action == "add":
        print(f"saved {a.name}: {', '.join(sector.add_universe(a.name, a.tickers))}")
        return
    if a.name.lower() not in unis:
        sys.exit(f"unknown sector {a.name!r}; have {sorted(unis)}; add one with `eqr-track sector add NAME TICKERS...`")
    name = a.name.lower()
    snap = sector.snapshot(name, [c["ticker"] for c in unis[name]])
    prev = sector.previous(name, snap["date"])
    text = sector.render(snap, prev)
    if not a.no_save:
        sector.save(snap)
    print(text)
    print(f"saved: {_save_report('sector-' + name, text)}")


# ------------------------------------------------------------------ watchlist
def cmd_watch(a) -> None:
    if a.action == "add":
        print(monitor.add(a.ticker, a.above, a.below, a.note or ""))
    elif a.action == "remove":
        print("removed" if monitor.remove(a.ticker) else "not on the watchlist")
    elif a.action == "list":
        for t, e in monitor.load_watchlist().items():
            print(f"{t:6} above {e.get('price_above')}  below {e.get('price_below')}  added {e['added']}  {e.get('note', '')}")
    else:
        text = monitor.render(monitor.run())
        print(text)
        print(f"saved: {_save_report('monitor', text)}")


# ------------------------------------------------------------------ ideas
def cmd_ideas(a) -> None:
    if a.action == "list":
        for i in ideas.load():
            print(f"{i['date']} {i['ticker']:6} {i['status']:10} {', '.join(i['themes'])}")
        return
    if a.action in ("dismiss", "watch"):
        n = ideas.set_status(a.ticker, "dismissed" if a.action == "dismiss" else "watching")
        if a.action == "watch":
            monitor.add(a.ticker)
        print(f"{n} idea(s) updated" + ("; added to the watchlist" if a.action == "watch" else ""))
        return
    unis = sector.all_universes()
    tickers: list[str] = []
    names = []
    for u in a.universe:
        if u.lower() in unis:
            names.append(u.lower())
            tickers += [c["ticker"] for c in unis[u.lower()]]
        else:
            names.append("custom")
            tickers.append(u.upper())
    if not tickers:
        sys.exit("name a sector or tickers: `eqr-track ideas run beauty`")
    res = ideas.generate(list(dict.fromkeys(tickers)), "+".join(dict.fromkeys(names)), include_covered=a.include_covered)
    ideas.save(res["ideas"])
    text = ideas.render(res)
    print(text)
    if a.research and res["ideas"]:
        top = [i["ticker"] for i in res["ideas"][: a.research]]
        print(f"Researching {', '.join(top)} with the full agent ...", flush=True)
        results = screen.stage2(top, workers=a.workers, progress=print, run_fn=lambda t: screen.run_one(t, universe="ideas"))
        for r in results:
            if r.get("rating"):
                ideas.set_status(r["ticker"], "researched")
        ok, low, bad, odd = screen.buy_lists(results)
        print("Buys:", ", ".join(r["ticker"] for r in ok) or "none", "| low conviction:", ", ".join(r["ticker"] for r in low) or "none")
    print(f"saved: {_save_report('ideas', text)}")


# ------------------------------------------------------------------ buys
def cmd_buys(a) -> None:
    b = recommend.run(refresh_prices=a.refresh_prices, max_age_days=a.max_age_days)
    text = recommend.render(b, max_age_days=a.max_age_days)
    print(text)
    print(f"saved: {_save_report('buys', text)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eqr-track", description="Thesis, sector, watchlist, ideas and Buy-list tracking")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("thesis")
    p.add_argument("action", choices=["add", "check", "list"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--pillar", action="append", default=[], help="kill criterion, e.g. 'operating_margin_pct >= 17' (Broken if it fails)")
    p.add_argument("--watch-pillar", action="append", default=[], help="softer criterion (Watch if it fails)")
    p.set_defaults(fn=cmd_thesis)

    p = sub.add_parser("sector")
    p.add_argument("action", choices=["list", "add", "snapshot"])
    p.add_argument("name", nargs="?")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--no-save", action="store_true", help="do not record this snapshot for later comparison")
    p.set_defaults(fn=cmd_sector)

    p = sub.add_parser("watch")
    p.add_argument("action", choices=["add", "remove", "list", "run"])
    p.add_argument("ticker", nargs="?")
    p.add_argument("--above", type=float)
    p.add_argument("--below", type=float)
    p.add_argument("--note")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("ideas")
    p.add_argument("action", choices=["run", "list", "dismiss", "watch"])
    p.add_argument("universe", nargs="*", help="sector names or tickers (run), or one ticker (dismiss/watch)")
    p.add_argument("--research", type=int, default=0, help="run the full agent on the top N ideas")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--include-covered", action="store_true", help="also propose names already rated or on the watchlist")
    p.set_defaults(fn=cmd_ideas)

    p = sub.add_parser("buys")
    p.add_argument("--refresh-prices", action="store_true", help="look up today's prices and re-price every Buy")
    p.add_argument("--max-age-days", type=int, default=recommend.MAX_AGE_DAYS)
    p.set_defaults(fn=cmd_buys)

    a = ap.parse_args(argv)
    if a.cmd == "watch" and a.action in ("add", "remove") and not a.ticker:
        ap.error("watch add/remove needs a TICKER")
    if a.cmd == "sector" and a.action != "list" and not a.name:
        ap.error("sector add/snapshot needs a NAME")
    if a.cmd == "sector" and a.action == "add" and not a.tickers:
        ap.error("sector add needs tickers")
    if a.cmd == "thesis" and a.action == "add" and len(a.tickers) != 1:
        ap.error("thesis add needs exactly one TICKER")
    if a.cmd == "ideas":
        if a.action in ("dismiss", "watch"):
            if len(a.universe) != 1:
                ap.error(f"ideas {a.action} needs exactly one TICKER")
            a.ticker = a.universe[0]
    a.fn(a)


if __name__ == "__main__":
    main()
