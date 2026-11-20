"""Buy screener: a cheap quantitative screen over a universe, then the full agent on the shortlist, keeping the Buys.

    uv run eqr-screen beauty --top 5
    uv run eqr-screen MO PM TPB UVV --top 3 --hide-low-conviction
    uv run eqr-screen beauty --no-agent            # stage 1 only: no agent runs, no big token cost

Stage 1 (no agent, seconds per ticker) ranks companies against each other on five signals. The weights are below, in
one place, so you can see and change them. Percentiles are computed WITHIN the universe you pass, so the screen answers
"which of these look cheapest and healthiest relative to each other", not "which stocks are good".

Stage 2 runs the full analyst agent on the top N. A Buy is the fixed-rule rating (see target.py) on scenarios the model
proposed. Screening many names and keeping the Buys partly selects for the model's optimism and for noise: the report
lists Low-conviction Buys separately and says so. Output is research leads to verify, not advice.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from . import comps, history, valuation
from .comps import UNIVERSES
from .schemas import ResearchNote

# ---- the screen's weights: edit here. Each signal is a percentile (0 = worst, 1 = best) within the universe.
WEIGHTS = {
    "value": 0.35,       # cheapness: low P/E and EV/EBIT (EV/Sales only if there are no positive earnings)
    "fcf_yield": 0.20,   # free cash flow / market cap, higher is better
    "growth": 0.15,      # latest-year revenue growth, higher is better
    "quality": 0.15,     # operating margin, higher is better
    "leverage": 0.15,    # net debt / EBIT, lower is better (needs positive EBIT)
}
MIN_SIGNALS = 3          # a company with fewer measurable signals is excluded, not guessed at
SMALL_UNIVERSE = 5       # percentiles over fewer names are coarse: the report says so
PLAUSIBLE_RETURN_PCT = 100.0  # |expected total return| above this is more likely bad inputs than an opportunity


# ------------------------------------------------------------------ stage 1: quantitative screen
def pct_rank(x, values, higher_is_better: bool = True):
    """Percentile of x among the valid values: 1.0 = best, 0.0 = worst, ties share rank. None if x is None."""
    if x is None:
        return None
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.5
    worse = sum(1 for v in vals if (v < x if higher_is_better else v > x))
    ties = sum(1 for v in vals if v == x) - 1
    return (worse + 0.5 * ties) / (len(vals) - 1)


def resolve_universe(args: list[str], exclude: list[str] | None = None) -> tuple[str, list[str]]:
    """'beauty' -> the curated list; otherwise treat the arguments as tickers (commas or spaces)."""
    if len(args) == 1 and args[0].lower() in UNIVERSES:
        name, tickers = args[0].lower(), [c["ticker"] for c in UNIVERSES[args[0].lower()]]
    else:
        name, tickers = "custom", [t for a in args for t in re.split(r"[,\s]+", a) if t]
    skip = {e.upper() for e in (exclude or [])}
    seen: list[str] = []
    for t in (x.upper() for x in tickers):
        if t not in seen and t not in skip:
            seen.append(t)
    return name, seen


def stage1(tickers: list[str], today: date | None = None, *, valuation_fn=valuation.get_valuation,
           comps_fn=comps.get_comps, include_suspect: bool = False) -> dict:
    today = today or date.today()
    val, cmp_ = valuation_fn(tickers, today), comps_fn(tickers, today)
    vrows = {r["ticker"]: r for r in val["rows"]}
    crows = {r["ticker"]: r for r in cmp_["rows"]}
    errors = {**cmp_.get("errors", {}), **val.get("errors", {})}
    raw: dict[str, dict] = {}
    excluded: list[tuple[str, str]] = []
    for t in tickers:
        v, c = vrows.get(t), crows.get(t)
        if not v or not c:
            excluded.append((t, errors.get(t, "no data")))
        elif v["withheld"]:
            excluded.append((t, "no verified valuation: " + ", ".join(v["flags"][:3])))
        elif c.get("stale"):
            excluded.append((t, f"stale financials (latest annual period {c['period_end']})"))
        elif not include_suspect and any(x.startswith("check_outlier") for x in v["flags"]):
            # An extreme multiple (e.g. P/E 1.2 from a one-off gain) looks "cheap" but is usually bad data: it would
            # top the ranking and distort everyone else's percentiles. Leave it out and say why; a human can look.
            excluded.append((t, "suspect data, review by hand: " + ", ".join(x for x in v["flags"] if x.startswith("check_outlier"))))
        else:
            m, f, cm = v["metrics"], v.get("fundamentals", {}), c["metrics"]
            nd, ebit = f.get("net_debt_proxy"), f.get("operating_income")
            raw[t] = {
                "company": v["company"], "price": v["price"], "flags": v["flags"],
                "pe": m.get("pe", {}).get("value"), "ev_ebit": m.get("ev_ebit", {}).get("value"),
                "ev_sales": m.get("ev_sales", {}).get("value"), "fcf_yield": m.get("fcf_yield_pct", {}).get("value"),
                "growth": cm.get("revenue_growth_pct", {}).get("value"),
                "quality": cm.get("operating_margin_pct", {}).get("value"),
                "leverage": (nd / ebit) if nd is not None and ebit and ebit > 0 else None,
            }
    col = lambda k: [r[k] for r in raw.values()]  # noqa: E731
    rows = []
    for t, r in raw.items():
        value_parts = [p for p in (pct_rank(r["pe"], col("pe"), False), pct_rank(r["ev_ebit"], col("ev_ebit"), False))
                       if p is not None]
        value = sum(value_parts) / len(value_parts) if value_parts else None
        note = None
        if value is None and r["ev_sales"] is not None:
            value, note = pct_rank(r["ev_sales"], col("ev_sales"), False), "value_from_ev_sales"
        sig = {"value": value, "fcf_yield": pct_rank(r["fcf_yield"], col("fcf_yield"), True),
               "growth": pct_rank(r["growth"], col("growth"), True), "quality": pct_rank(r["quality"], col("quality"), True),
               "leverage": pct_rank(r["leverage"], col("leverage"), False)}
        avail = {k: x for k, x in sig.items() if x is not None}
        if len(avail) < MIN_SIGNALS:
            excluded.append((t, f"only {len(avail)} measurable signals (need {MIN_SIGNALS})"))
            continue
        wsum = sum(WEIGHTS[k] for k in avail)
        score = 100 * sum(WEIGHTS[k] * x for k, x in avail.items()) / wsum
        rows.append({"ticker": t, "score": round(score, 1), "signals": sig, "signals_used": len(avail),
                     "notes": [note] if note else [], **{k: r[k] for k in ("company", "price", "flags", "pe", "ev_ebit",
                                                                       "ev_sales", "fcf_yield", "growth", "quality", "leverage")}})
    rows.sort(key=lambda x: -x["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"rows": rows, "excluded": excluded, "small_universe": len(rows) < SMALL_UNIVERSE}


# ------------------------------------------------------------------ stage 2: the full agent on the shortlist
def summarize(note: ResearchNote | None, trace: dict, md_path=None, pdf_path=None, error=None) -> dict:
    from .graders import recommendation_consistent

    out = {"ticker": (note.ticker if note else trace.get("ticker")), "error": error or trace.get("error"),
           "trace_url": trace.get("trace_url"), "run_id": trace.get("run_id"), "md": str(md_path) if md_path else None,
           "pdf": str(pdf_path) if pdf_path else None, "rating": None, "consistent": None}
    rec = note.recommendation if note else None
    if rec:
        out.update(rating=rec.rating, price_target=rec.price_target, current_price=rec.current_price,
                   expected_total_return_pct=rec.expected_total_return_pct, conviction=rec.conviction, method=rec.method,
                   flags=rec.flags, consistent=recommendation_consistent(note, trace))
    return out


def find_recent_run(ticker: str, max_age_hours: float, runs_dir: str | Path = "runs") -> dict | None:
    """Newest saved run for this ticker (with a note) younger than max_age_hours, or None."""
    import json

    cutoff = time.time() - max_age_hours * 3600
    for p in sorted(Path(runs_dir).glob(f"*-{ticker.upper()}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.stat().st_mtime < cutoff:
            break
        t = json.loads(p.read_text())
        if t.get("note"):
            return t
    return None


def run_one(ticker: str, *, model: str | None = None, engine: str | None = None, universe: str = "custom",
            reuse_hours: float = 0.0) -> dict:
    """Run (or reuse) the full agent for one ticker, write its markdown + PDF, and log the rating."""
    from . import agent
    from .cli import DEFAULT_TASK, render
    from .pdf import build_pdf

    trace = find_recent_run(ticker, reuse_hours) if reuse_hours > 0 else None
    error = None
    if trace is None:
        res = agent.run(DEFAULT_TASK.format(ticker=ticker), ticker=ticker, model=model, engine=engine,
                        meta={"screen": universe})
        trace, error = res.trace, res.error
    note = ResearchNote.model_validate(trace["note"]) if trace.get("note") else None
    md_path = pdf_path = None
    if note:
        stem = f"{ticker}-{date.today().isoformat()}"
        md_path = Path("reports") / f"{stem}.md"
        md_path.parent.mkdir(exist_ok=True)
        md_path.write_text(render(note) + "\n")
        try:
            pdf_path = build_pdf(note, trace, Path("reports") / f"{stem}.pdf")
        except Exception as e:  # a PDF failure must not lose the rating
            error = (error or "") + f" PDF failed: {type(e).__name__}: {e}"
        try:
            history.log_rating(note, trace)
        except Exception:
            pass
    return summarize(note, trace, md_path, pdf_path, error)


def stage2(tickers: list[str], *, run_fn, workers: int = 3, progress=lambda s: None) -> list[dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(run_fn, t): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                results[t] = fut.result()
            except Exception as e:  # one failed run must not sink the screen
                results[t] = {"ticker": t, "rating": None, "error": f"{type(e).__name__}: {e}", "consistent": None}
            r = results[t]
            progress(f"  {t:6} {r.get('rating') or 'no rating':9} {r.get('conviction') or '':7} "
                     f"{('PT $%.2f' % r['price_target']) if r.get('price_target') else ''}")
    return [results[t] for t in tickers]  # keep the stage-1 rank order


# ------------------------------------------------------------------ report
def _f(x, nd=1, suffix=""):
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def buy_lists(results: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """(normal-conviction Buys, low-conviction Buys, integrity failures, implausible outputs).
    A Buy whose note differs from the computed result is an integrity failure, never a Buy. A Buy promising more than
    +/-PLAUSIBLE_RETURN_PCT is routed to 'implausible': such numbers usually come from bad inputs, not opportunity."""
    ok, low, bad, odd = [], [], [], []
    for r in results:
        if r.get("rating") != "Buy":
            continue
        if r.get("consistent") != 1.0:
            bad.append(r)
        elif abs(r.get("expected_total_return_pct") or 0) > PLAUSIBLE_RETURN_PCT:
            odd.append(r)
        elif r.get("conviction") == "Low":
            low.append(r)
        else:
            ok.append(r)
    key = lambda r: -(r.get("expected_total_return_pct") or 0)  # noqa: E731
    return sorted(ok, key=key), sorted(low, key=key), bad, odd


def render_report(name: str, s1: dict, results: list[dict] | None, today: date, top: int, hide_low: bool = False) -> str:
    L = [f"# Buy screen: {name} ({today.isoformat()})", "",
         "> **Research leads, not advice.** Stage 1 ranks companies against each other by code; stage 2 is the full "
         "agent. A Buy is a fixed-rule rating on model-proposed scenarios. Keeping only Buys from many names partly "
         "selects for the model's optimism and for noise, so verify each one against its filings.", "",
         "## Stage 1: quantitative screen", "",
         "Weights: " + ", ".join(f"{k} {w:.0%}" for k, w in WEIGHTS.items()) + ". Each signal is a percentile within "
         "this universe (1.0 = best).", ""]
    if s1["small_universe"]:
        L += [f"> Only {len(s1['rows'])} companies were screened, so percentiles are coarse.", ""]
    L += ["| rank | ticker | score | value | FCF yld | growth | quality | leverage | P/E | EV/EBIT | FCF yield % | rev growth % | op margin % | flags |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s1["rows"]:
        sg = r["signals"]
        L.append(f"| {r['rank']} | {r['ticker']} | {r['score']:.1f} | " + " | ".join(_f(sg[k], 2) for k in WEIGHTS) +
                 f" | {_f(r['pe'])} | {_f(r['ev_ebit'])} | {_f(r['fcf_yield'])} | {_f(r['growth'])} | {_f(r['quality'])} | "
                 + (", ".join(r["flags"][:3] + r["notes"]) or "-") + " |")
    if s1["excluded"]:
        L += ["", "Excluded (no verified data, not guessed at): " + "; ".join(f"**{t}**: {why}" for t, why in s1["excluded"])]
    if results is not None:
        L += ["", f"## Stage 2: full analysis of the top {top}", "",
              "| ticker | rating | price target | price | expected total return | conviction | method | rating matches computed result | report |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            L.append(f"| {r['ticker']} | {r.get('rating') or 'none'} | {_f(r.get('price_target'), 2)} | {_f(r.get('current_price'), 2)} | "
                     f"{_f(r.get('expected_total_return_pct'), 1, '%')} | {r.get('conviction') or '-'} | {r.get('method') or '-'} | "
                     f"{'yes' if r.get('consistent') == 1.0 else ('NO' if r.get('consistent') == 0.0 else '-')} | "
                     f"{Path(r['pdf']).name if r.get('pdf') else (r.get('error') or '-')[:60]} |")
        ok, low, bad, odd = buy_lists(results)
        L += ["", "## Buys"]
        L += [f"- **{r['ticker']}**: target ${r['price_target']:.2f} vs ${r['current_price']:.2f}, expected total return "
              f"{r['expected_total_return_pct']:+.1f}% ({r['method']}). Flags: {', '.join(r.get('flags') or []) or 'none'}" for r in ok] \
            or ["- none with normal conviction"]
        if low and not hide_low:
            L += ["", "## Buys with LOW conviction (treat with extra caution)"]
            L += [f"- **{r['ticker']}**: target ${r['price_target']:.2f} vs ${r['current_price']:.2f}, expected total return "
                  f"{r['expected_total_return_pct']:+.1f}% ({r['method']}). Flags: {', '.join(r.get('flags') or []) or 'none'}" for r in low]
        elif low:
            L += ["", f"({len(low)} low-conviction Buy(s) hidden by --hide-low-conviction.)"]
        if odd:
            L += ["", f"## Implausible output (expected return beyond +/-{PLAUSIBLE_RETURN_PCT:.0f}%): review by hand, not listed as Buys"]
            L += [f"- **{r['ticker']}**: target ${r['price_target']:.2f} vs ${r['current_price']:.2f} ({r['expected_total_return_pct']:+.1f}%, "
                  f"{r['method']}, {r.get('conviction')} conviction). Flags: {', '.join(r.get('flags') or []) or 'none'}" for r in odd]
        if bad:
            L += ["", "## Integrity failures (note differs from the computed rating; excluded)"] + [f"- {r['ticker']}" for r in bad]
        errs = [r for r in results if r.get("error") and not r.get("rating")]
        if errs:
            L += ["", "## No rating produced"] + [f"- {r['ticker']}: {(r.get('error') or 'no recommendation in note')[:140]}" for r in errs]
    return "\n".join(L) + "\n"


def write_outputs(name: str, s1: dict, results, today: date, top: int, hide_low: bool, out_dir: str | Path = "reports"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    md = out / f"screen-{name}-{today.isoformat()}.md"
    md.write_text(render_report(name, s1, results, today, top, hide_low))
    csv_path = out / f"screen-{name}-{today.isoformat()}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "ticker", "company", "score", *WEIGHTS, "pe", "ev_ebit", "fcf_yield_pct", "rev_growth_pct",
                    "op_margin_pct", "price", "flags"])
        for r in s1["rows"]:
            w.writerow([r["rank"], r["ticker"], r["company"], r["score"], *[r["signals"][k] for k in WEIGHTS], r["pe"],
                        r["ev_ebit"], r["fcf_yield"], r["growth"], r["quality"], r["price"], ";".join(r["flags"])])
    return md, csv_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Screen a universe, run the full agent on the top names, list the Buys")
    ap.add_argument("universe", nargs="+", help=f"a sector name ({', '.join(sorted(UNIVERSES))}) or tickers")
    ap.add_argument("--top", type=int, default=5, help="how many top-ranked names get a full agent run (default 5)")
    ap.add_argument("--no-agent", action="store_true", help="stage 1 only: no agent runs")
    ap.add_argument("--workers", type=int, default=3, help="parallel agent runs (default 3)")
    ap.add_argument("--exclude", nargs="*", default=[], help="tickers to leave out")
    ap.add_argument("--include-suspect", action="store_true", help="keep rows with outlier-flagged multiples in the screen")
    ap.add_argument("--hide-low-conviction", action="store_true", help="do not list low-conviction Buys")
    ap.add_argument("--reuse-hours", type=float, default=0.0, help="reuse a saved run younger than this instead of re-running")
    ap.add_argument("--model")
    ap.add_argument("--engine", choices=["sdk", "loop"])
    args = ap.parse_args(argv)

    today = date.today()
    name, tickers = resolve_universe(args.universe, args.exclude)
    if not tickers:
        sys.exit("no tickers to screen")
    print(f"Stage 1: screening {len(tickers)} companies ({name}) ...", flush=True)
    s1 = stage1(tickers, today, include_suspect=args.include_suspect)
    print(f"  ranked {len(s1['rows'])}, excluded {len(s1['excluded'])}", flush=True)
    results = None
    if not args.no_agent and s1["rows"]:
        shortlist = [r["ticker"] for r in s1["rows"][: args.top]]
        print(f"Stage 2: full agent on {', '.join(shortlist)} ({args.workers} at a time; ~1.5 min each) ...", flush=True)
        results = stage2(shortlist, workers=args.workers, progress=lambda s: print(s, flush=True),
                         run_fn=lambda t: run_one(t, model=args.model, engine=args.engine, universe=name,
                                                  reuse_hours=args.reuse_hours))
    md, csv_path = write_outputs(name, s1, results, today, args.top, args.hide_low_conviction)
    print("\n" + md.read_text())
    print(f"saved: {md}\n       {csv_path}")


if __name__ == "__main__":
    main()
