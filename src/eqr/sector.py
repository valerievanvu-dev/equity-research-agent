"""Sector tracker and sector analysis, computed in code from SEC financials and verified prices (no model call).

`snapshot` measures a universe today: median multiples, growth and margins with their spread, and every company's
screen score. Snapshots are appended to data/sector_snapshots.jsonl, so `compare` can say what changed since the last
one (the "tracker"). `findings` turns one snapshot (plus the previous) into plain statements a person can check:
who is cheap or rich versus the sector median, who is growing faster *and* trading cheaper, and so on.

Sector statistics use only rows that passed the screen's data checks (verified price, fresh financials, no outlier
multiples). Everyone else is listed as excluded with the reason, never averaged in.
"""
from __future__ import annotations

import statistics as st
from datetime import date

from . import comps, screen, store, valuation
from .comps import UNIVERSES

STATS = ("pe", "ev_ebit", "ev_sales", "fcf_yield", "growth", "quality", "leverage")
LABELS = {"pe": "P/E", "ev_ebit": "EV/EBIT", "ev_sales": "EV/Sales", "fcf_yield": "FCF yield %", "growth": "revenue growth %",
          "quality": "operating margin %", "leverage": "net debt / EBIT"}
CHEAP_PCT = -15.0     # at least this far below the sector median multiple counts as a discount (percent)
RICH_PCT = 15.0
LEVERAGE_FLAG = 3.0


# ------------------------------------------------------------------ universes
def all_universes() -> dict[str, list[dict]]:
    """Built-in sectors plus the ones you added (data/universes.json: {name: [tickers]})."""
    extra = store.read_json(store.data_dir() / "universes.json", {})
    return {**UNIVERSES, **{k: [{"ticker": t, "name": t, "segment": ""} for t in v] for k, v in extra.items()}}


def add_universe(name: str, tickers: list[str]) -> list[str]:
    name = name.lower()
    if name in UNIVERSES:
        raise ValueError(f"'{name}' is a built-in sector; pick another name")
    path = store.data_dir() / "universes.json"
    data = store.read_json(path, {})
    data[name] = list(dict.fromkeys(t.upper() for t in tickers))
    store.write_json(path, data)
    return data[name]


# ------------------------------------------------------------------ snapshot
def _spread(values: list[float]) -> dict:
    vals = sorted(values)
    q = st.quantiles(vals, n=4, method="inclusive") if len(vals) >= 2 else [vals[0]] * 3
    return {"median": round(st.median(vals), 2), "p25": round(q[0], 2), "p75": round(q[2], 2), "n": len(vals)}


def snapshot(name: str, tickers: list[str], today: date | None = None, *, valuation_fn=valuation.get_valuation,
             comps_fn=comps.get_comps) -> dict:
    today = today or date.today()
    s1 = screen.stage1(tickers, today, valuation_fn=valuation_fn, comps_fn=comps_fn)
    rows = s1["rows"]
    stats = {}
    for k in STATS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            stats[k] = _spread(vals)
    names = {c["ticker"]: c["name"] for c in all_universes().get(name, [])}
    return {
        "date": today.isoformat(), "sector": name, "universe": len(tickers), "n": len(rows), "stats": stats,
        "profitable_share": round(sum(1 for r in rows if r["pe"] is not None) / len(rows), 2) if rows else None,
        "growing_share": round(sum(1 for r in rows if (r["growth"] or 0) > 0) / len(rows), 2) if rows else None,
        "companies": [{"ticker": r["ticker"], "company": names.get(r["ticker"]) or r["company"], "rank": r["rank"],
                       "score": r["score"], **{k: r.get(k) for k in STATS}, "flags": r["flags"]} for r in rows],
        "excluded": s1["excluded"], "small": len(rows) < screen.SMALL_UNIVERSE,
    }


def snapshots_path():
    return store.data_dir() / "sector_snapshots.jsonl"


def save(snap: dict) -> None:
    store.append_jsonl(snapshots_path(), snap)


def previous(name: str, before: str) -> dict | None:
    """Latest saved snapshot for the sector dated strictly before `before` (YYYY-MM-DD)."""
    rows = [r for r in store.read_jsonl(snapshots_path()) if r["sector"] == name and r["date"] < before]
    return max(rows, key=lambda r: r["date"]) if rows else None


# ------------------------------------------------------------------ analysis
def compare(cur: dict, prev: dict | None) -> dict:
    """Change in each sector statistic's median, and each company's rank, since the previous snapshot."""
    if not prev:
        return {"since": None, "stat_changes": {}, "rank_moves": [], "entered": [], "left": []}
    changes = {}
    for k, s in cur["stats"].items():
        p = prev["stats"].get(k)
        if p and p["median"]:
            changes[k] = {"from": p["median"], "to": s["median"], "change_pct": round((s["median"] / p["median"] - 1) * 100, 1)
                          if p["median"] > 0 else None, "change_abs": round(s["median"] - p["median"], 2)}
    prank = {c["ticker"]: c["rank"] for c in prev["companies"]}
    moves = sorted(({"ticker": c["ticker"], "from": prank[c["ticker"]], "to": c["rank"]} for c in cur["companies"]
                    if c["ticker"] in prank and prank[c["ticker"]] != c["rank"]), key=lambda m: -abs(m["from"] - m["to"]))
    now, was = {c["ticker"] for c in cur["companies"]}, set(prank)
    return {"since": prev["date"], "stat_changes": changes, "rank_moves": moves,
            "entered": sorted(now - was), "left": sorted(was - now)}


def _prem(x, med):
    return None if x is None or not med or med <= 0 else round((x / med - 1) * 100, 1)


def findings(cur: dict, prev: dict | None = None) -> list[str]:
    """Plain, checkable statements about the sector. Every number is in the snapshot."""
    out: list[str] = []
    cs, st_ = cur["companies"], cur["stats"]
    if not cs:
        return ["No company passed the data checks, so there is nothing to say about this sector."]
    if cur["small"]:
        out.append(f"Only {cur['n']} companies passed the checks, so medians and percentiles are coarse.")
    key = "ev_ebit" if "ev_ebit" in st_ else "pe" if "pe" in st_ else None
    if key:
        s = st_[key]
        out.append(f"Sector median {LABELS[key]} is {s['median']:.1f}x (middle half {s['p25']:.1f}x to {s['p75']:.1f}x, n={s['n']}).")
        prem = sorted(((_prem(c[key], s["median"]), c["ticker"]) for c in cs if _prem(c[key], s["median"]) is not None))
        cheap = [f"{t} ({p:+.0f}%)" for p, t in prem if p <= CHEAP_PCT]
        rich = [f"{t} ({p:+.0f}%)" for p, t in reversed(prem) if p >= RICH_PCT]
        if cheap:
            out.append(f"Cheapest vs the sector median on {LABELS[key]}: " + ", ".join(cheap) + ".")
        if rich:
            out.append(f"Richest vs the sector median on {LABELS[key]}: " + ", ".join(rich) + ".")
        gm = st_.get("growth", {}).get("median")
        if gm is not None:
            disc = [c["ticker"] for c in cs if c.get("growth") is not None and c["growth"] > gm and (_prem(c[key], s["median"]) or 0) <= 0
                    and c[key] is not None]
            lag = [c["ticker"] for c in cs if c.get("growth") is not None and c["growth"] < gm and (_prem(c[key], s["median"]) or 0) >= RICH_PCT
                   and c[key] is not None]
            if disc:
                out.append("Growing faster than the median AND at or below the median multiple (candidates to look at): " + ", ".join(disc) + ".")
            if lag:
                out.append("Growing slower than the median yet trading at a premium (expectations may be too high): " + ", ".join(lag) + ".")
    if "quality" in st_:
        top = max((c for c in cs if c.get("quality") is not None), key=lambda c: c["quality"], default=None)
        if top:
            out.append(f"Margin leader: {top['ticker']} at {top['quality']:.1f}% operating margin (sector median {st_['quality']['median']:.1f}%).")
    lev = [f"{c['ticker']} ({c['leverage']:.1f}x)" for c in cs if c.get("leverage") is not None and c["leverage"] > LEVERAGE_FLAG]
    if lev:
        out.append(f"Leverage above {LEVERAGE_FLAG:.0f}x net debt / EBIT: " + ", ".join(lev) + ".")
    if cur["profitable_share"] is not None and cur["profitable_share"] < 1:
        out.append(f"Only {cur['profitable_share']:.0%} of the sector has positive EPS; P/E excludes the rest.")
    if prev:
        ch = compare(cur, prev)["stat_changes"]
        for k in ("ev_ebit", "pe", "growth", "quality"):
            if k in ch and ch[k]["change_abs"] != 0:
                unit = "pts" if k in ("growth", "quality") else "turns"
                out.append(f"Since {prev['date']}: median {LABELS[k]} moved {ch[k]['from']:.1f} -> {ch[k]['to']:.1f} ({ch[k]['change_abs']:+.1f} {unit}).")
    return out


def render(cur: dict, prev: dict | None = None) -> str:
    f = lambda x, nd=1: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
    L = [f"# Sector: {cur['sector']} ({cur['date']})", "",
         "> Research leads, not advice. Computed in code from SEC financials and verified prices; no model judgement.", "",
         "## What stands out", ""] + [f"- {x}" for x in findings(cur, prev)]
    L += ["", "## Sector statistics (screened companies only)", "", "| statistic | median | middle half | n |", "|---|---|---|---|"]
    L += [f"| {LABELS[k]} | {f(s['median'])} | {f(s['p25'])} to {f(s['p75'])} | {s['n']} |" for k, s in cur["stats"].items()]
    L += ["", "## Companies (ranked by screen score)", "",
          "| rank | ticker | score | P/E | EV/EBIT | FCF yld % | growth % | op margin % | net debt/EBIT | vs median EV/EBIT |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    med = cur["stats"].get("ev_ebit", {}).get("median")
    for c in cur["companies"]:
        p = _prem(c["ev_ebit"], med)
        L.append(f"| {c['rank']} | {c['ticker']} | {c['score']:.1f} | {f(c['pe'])} | {f(c['ev_ebit'])} | {f(c['fcf_yield'])} | "
                 f"{f(c['growth'])} | {f(c['quality'])} | {f(c['leverage'])} | {'n/a' if p is None else f'{p:+.0f}%'} |")
    if cur["excluded"]:
        L += ["", "Excluded (no verified data, not guessed at): " + "; ".join(f"**{t}**: {why}" for t, why in cur["excluded"])]
    cmp_ = compare(cur, prev)
    if cmp_["since"]:
        L += ["", f"## Change since {cmp_['since']}", ""]
        L += [f"- median {LABELS[k]}: {v['from']:.1f} -> {v['to']:.1f}" + (f" ({v['change_pct']:+.1f}%)" if v["change_pct"] is not None else "")
              for k, v in cmp_["stat_changes"].items()]
        L += [f"- rank change {m['ticker']}: #{m['from']} -> #{m['to']}" for m in cmp_["rank_moves"][:5]]
        if cmp_["entered"] or cmp_["left"]:
            L += [f"- entered the screen: {', '.join(cmp_['entered']) or 'none'}; left: {', '.join(cmp_['left']) or 'none'}"]
    else:
        L += ["", "_First snapshot for this sector: run again later to see what changed._"]
    return "\n".join(L) + "\n"
