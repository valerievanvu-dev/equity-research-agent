"""Offline tests for the tracking features: thesis, sector, monitor, ideas, recommendations."""
from datetime import date

import pytest

from eqr.schemas import ResearchNote

TODAY = date(2026, 11, 20)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQR_DATA", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data"


def _note(price=100.0, pt=130.0, growth=20.0, margin=15.0, ticker="AAA"):
    return ResearchNote(
        ticker=ticker, company=ticker + " Co", summary="s", thesis="Margins expand and the stock is cheap.", metrics=[],
        bull_case=[{"text": "bull", "sources": ["x"]}], bear_case=[{"text": "bear", "sources": ["x"]}], key_risks=[],
        comps=[{"ticker": ticker, "period_end": "2026-03-31", "sources": ["c"],
                "metrics": {"revenue_growth_pct": growth, "operating_margin_pct": margin}}],
        recommendation={"rating": "Buy", "price_target": pt, "current_price": price, "expected_total_return_pct": 30.0, "method": "pe",
                        "conviction": "Normal", "rationale": "r", "sources": ["s"], "scenarios": []})


def _snap(price=105.0, growth=18.0, margin=14.0, **val):
    return {"ticker": "AAA", "price": price, "flags": [], "comps": {"revenue_growth_pct": growth, "operating_margin_pct": margin},
            "valuation": val}


# ------------------------------------------------------------------ thesis
def test_parse_pillar():
    from eqr.thesis import parse_pillar

    p = parse_pillar("operating_margin_pct >= 17.5")
    assert (p["metric"], p["op"], p["value"], p["on_fail"]) == ("operating_margin_pct", ">=", 17.5, "broken")
    for bad in ("margin >= 3", "pe >> 3", "pe >= x", "pe"):
        with pytest.raises(ValueError):
            parse_pillar(bad)


def test_seed_pillars_come_from_the_notes_own_numbers():
    from eqr.thesis import seed_pillars

    by = {p["id"]: p for p in seed_pillars(_note(growth=20.0, margin=15.0, price=100.0))}
    assert by["seed:growth"]["value"] == 10.0                   # allowed to halve
    assert by["seed:margin"]["value"] == 12.0                   # 300 bps cushion
    assert by["seed:drawdown"]["value"] == 80.0 and by["seed:drawdown"]["on_fail"] == "broken"
    assert by["seed:growth"]["on_fail"] == "watch"


def test_thesis_status_rules():
    from eqr import thesis

    th = thesis.create(_note(), None, [thesis.parse_pillar("pe <= 25")], today=TODAY)
    ok = thesis.evaluate(th, _snap(pe=20), TODAY)
    assert ok["status"] == "Intact" and ok["progress_to_target_pct"] == pytest.approx(16.7, abs=0.1)
    # a soft (seed) pillar failing is Watch
    assert thesis.evaluate(th, _snap(pe=20, growth=5.0), TODAY)["status"] == "Watch"
    # the user's kill criterion failing is Broken
    broken = thesis.evaluate(th, _snap(pe=40), TODAY)
    assert broken["status"] == "Broken" and "pe <= 25" in broken["reasons"][0]
    # the -20% drawdown pillar is also a kill criterion
    assert thesis.evaluate(th, _snap(price=79.0, pe=20), TODAY)["status"] == "Broken"
    # target reached: re-underwrite (Watch), not silently fine
    reached = thesis.evaluate(th, _snap(price=131.0, pe=20), TODAY)
    assert reached["status"] == "Watch" and "reached" in reached["reasons"][0]
    # missing data is unknown, never counted as passing or failing
    unk = thesis.evaluate(th, _snap(), TODAY)
    assert unk["status"] == "Intact" and unk["unknown"] == ["pe <= 25"]
    # no verified price: say so; price pillar unknown
    nop = thesis.evaluate(th, {"price": None, "flags": ["price_conflict"], "comps": {}, "valuation": {}}, TODAY)
    assert any("no verified price" in r for r in nop["reasons"])
    # a stale note becomes Watch
    assert thesis.evaluate(th, _snap(pe=20), date(2027, 4, 1))["status"] == "Watch"


def test_thesis_refresh_keeps_user_pillars_and_check_history():
    from eqr import thesis

    th = thesis.create(_note(), None, [thesis.parse_pillar("pe <= 25")], today=TODAY)
    thesis.record_check(th, thesis.evaluate(th, _snap(pe=20), TODAY))
    again = thesis.create(_note(price=110.0, growth=10.0), None, today=date(2026, 12, 20))
    ids = [p["id"] for p in again["pillars"]]
    assert "user:pe<=25" in ids and ids.count("seed:growth") == 1
    assert again["created"] == "2026-11-20" and again["updated"] == "2026-12-20" and len(again["checks"]) == 1
    assert next(p for p in again["pillars"] if p["id"] == "seed:growth")["value"] == 5.0   # reseeded from the new note
    assert "AAA: thesis OK" in thesis.render(again, thesis.evaluate(again, _snap(price=112.0, pe=20, growth=12.0), TODAY))


# ------------------------------------------------------------------ recommendations
def _rating(rating="Buy", price=100.0, pt=130.0, er=33.0, conviction="Normal", d="2026-11-15", t="AAA"):
    return {"date": d, "ticker": t, "company": t + " Co", "rating": rating, "price_target": pt, "current_price": price,
            "expected_total_return_pct": er, "conviction": conviction, "method": "pe", "flags": []}


def test_reprice_uses_the_same_rule_and_keeps_the_dividend_yield():
    from eqr.recommend import dividend_yield_pct, reprice

    r = _rating(price=100, pt=130, er=33.0)                      # 30% price return + 3% dividend yield
    assert dividend_yield_pct(r) == pytest.approx(3.0)
    assert reprice(r, 100)["expected_total_return_pct"] == 33.0
    assert reprice(r, 110)["rating"] == "Buy"                     # 130/110 -> +18.2% + 3% = 21.2%
    out = reprice(r, 125)                                         # +4% + 3% = 7%: no longer a Buy
    assert out["rating"] == "Hold" and out["expected_total_return_pct"] == 7.0
    below = out["buy_below"]
    assert reprice(r, below)["rating"] == "Buy" and reprice(r, below + 0.5)["rating"] != "Buy"   # the buy-below price is exact


def test_build_buckets():
    from eqr.recommend import build

    ratings = {
        "NOW": _rating(t="NOW"),
        "LOW": _rating(t="LOW", conviction="Low", er=40, price=100, pt=137),
        "RISEN": _rating(t="RISEN"),
        "OLD": _rating(t="OLD", d="2026-08-01"),
        "ODD": _rating(t="ODD", pt=400, er=303),
        "HOLD": _rating(t="HOLD", rating="Hold", er=5),
        "BRK": _rating(t="BRK"),
    }
    snaps = {"RISEN": {"price": 126.0}, "BRK": {"price": 100.0}}
    out = build(TODAY, ratings=ratings, snaps=snaps, theses={"BRK": {"status": "Broken"}})
    names = {k: [r["ticker"] for r in v] for k, v in out.items()}
    assert (names["buy"], names["buy_low"], names["wait"], names["refresh"]) == (["NOW"], ["LOW"], ["RISEN"], ["OLD"])
    assert sorted(names["review"]) == ["BRK", "ODD"]                          # implausible return, and a broken thesis
    stale_price = build(TODAY, ratings={"NOW": ratings["NOW"]}, snaps={})
    assert stale_price["buy"][0]["price_note"] and not stale_price["buy"][0]["price_is_live"]   # no live price: says so
    no_refresh = build(TODAY, ratings={"NOW": ratings["NOW"]})
    assert no_refresh["buy"][0]["price_note"] == ""                             # not asked to refresh: no complaint


def test_latest_rating_supersedes_an_earlier_buy(data_dir):
    from eqr import history, store

    store.append_jsonl(history.default_path(), _rating(d="2026-11-01"))
    store.append_jsonl(history.default_path(), _rating(rating="Hold", d="2026-11-10", er=5))
    assert history.latest_by_ticker()["AAA"]["rating"] == "Hold"
    from eqr.recommend import run
    assert all(not v for v in run(TODAY).values())                              # a Hold is not a recommendation


def test_buys_report_renders_with_disclaimer():
    from eqr.recommend import build, render

    text = render(build(TODAY, ratings={"NOW": _rating(t="NOW")}), TODAY)
    assert "not personal advice" in text and "| NOW |" in text and "## Wait for a dip" in text and "nothing is sized" in text


# ------------------------------------------------------------------ monitor
def _filing(acc, form="8-K", filed="2026-11-18"):
    return {"accession": acc, "form": form, "filed": filed, "period": "2026-09-30", "source": f"edgar:filing:AAA:{acc}"}


def _kinds(alerts):
    return [a["kind"] for a in alerts]


def test_monitor_first_check_is_a_baseline_then_reports_only_new_filings():
    from eqr.monitor import evaluate

    base = [_filing("a1"), _filing("a2", "10-Q")]
    alerts, st = evaluate("AAA", _snap(), entry={}, rating=None, filings=base, state={}, today=TODAY)
    assert "baseline" in _kinds(alerts) and "new_filing" not in _kinds(alerts)
    assert st["seen_filings"] == ["a1", "a2"]
    alerts, st = evaluate("AAA", _snap(), entry={}, rating=None, filings=base + [_filing("a3", "10-K"), _filing("a4")],
                          state=st, today=TODAY)
    new = [a for a in alerts if a["kind"] == "new_filing"]
    assert len(new) == 2 and {a["severity"] for a in new} == {"high", "medium"}      # 10-K high, 8-K medium


def test_monitor_price_alerts():
    from eqr.monitor import evaluate

    rating = _rating(price=100, pt=130, er=33.0)
    state = {"last_check": "2026-11-19", "seen_filings": [], "last_price": 100.0}

    def kinds(price, entry=None, rating=rating, state=state):
        alerts, _ = evaluate("AAA", _snap(price=price), entry=entry or {}, rating=rating, filings=[], state=state, today=TODAY)
        return _kinds(alerts)

    assert "target_reached" in kinds(131.0)
    assert "drawdown" in kinds(84.0) and "price_move" in kinds(84.0)
    assert "run_up" in kinds(126.0) and "buy_lapsed" in kinds(126.0)
    assert "price_below" in kinds(95.0, entry={"price_below": 96.0})
    assert "price_above" in kinds(105.0, entry={"price_above": 105.0})
    assert kinds(101.0) == []                                                   # nothing happened: no alerts
    assert "rating_stale" in kinds(101.0, rating=_rating(d="2026-06-01"))
    assert "no_rating" in kinds(101.0, rating=None)
    alerts, st = evaluate("AAA", {"price": None, "flags": ["price_conflict"], "comps": {}, "valuation": {}}, entry={},
                          rating=rating, filings=[], state=state, today=TODAY)
    assert _kinds(alerts) == ["no_price"] and st["last_price"] == 100.0         # keeps the previous price


def test_monitor_surfaces_the_thesis_status():
    from eqr.monitor import evaluate

    alerts, _ = evaluate("AAA", _snap(), entry={}, rating=None, filings=[], state={"last_check": "x"}, today=TODAY,
                         thesis_result={"status": "Broken", "reasons": ["pe <= 25: now 40.00"]})
    th = next(a for a in alerts if a["kind"] == "thesis")
    assert th["severity"] == "high" and "BROKEN" in th["text"]


def test_watchlist_and_run_persist_state(data_dir):
    from eqr import monitor, store

    monitor.add("aaa", price_below=90, today=TODAY)
    monitor.add("AAA", price_above=150)                       # updating keeps the earlier trigger
    assert monitor.load_watchlist()["AAA"]["price_below"] == 90 and monitor.load_watchlist()["AAA"]["price_above"] == 150
    snap = lambda tickers, today: {"AAA": _snap(price=80.0)}
    out = monitor.run(TODAY, snapshot_fn=snap, filings_fn=lambda t: [_filing("a1")])
    assert "price_below" in _kinds(out["AAA"]["alerts"])
    assert store.read_json(monitor.state_path(), {})["AAA"]["seen_filings"] == ["a1"]
    out = monitor.run(TODAY, snapshot_fn=snap, filings_fn=lambda t: [_filing("a1"), _filing("a2", "10-K")])
    assert "new_filing" in _kinds(out["AAA"]["alerts"])
    assert "AAA: $80.00" in monitor.render(out, TODAY)

    def boom(t):
        raise RuntimeError("sec down")
    out = monitor.run(TODAY, snapshot_fn=snap, filings_fn=boom)       # a failed filings fetch keeps price alerts and the baseline
    assert "filings_error" in _kinds(out["AAA"]["alerts"]) and "price_below" in _kinds(out["AAA"]["alerts"])
    assert store.read_json(monitor.state_path(), {})["AAA"]["seen_filings"] == ["a1", "a2"]
    assert monitor.remove("AAA") and not monitor.remove("AAA")


# ------------------------------------------------------------------ sector + ideas (shared fakes)
def _vr(t, pe, ebit, fcf, nd=100, oi=100):
    return {"ticker": t, "company": t + " Co", "price": 10.0, "withheld": False, "flags": [],
            "metrics": {"pe": {"value": pe}, "ev_ebit": {"value": ebit}, "fcf_yield_pct": {"value": fcf}},
            "fundamentals": {"net_debt_proxy": nd, "operating_income": oi}}


def _cr(t, growth, margin):
    return {"ticker": t, "stale": False, "period_end": "2026-03-31",
            "metrics": {"revenue_growth_pct": {"value": growth}, "operating_margin_pct": {"value": margin}}}


def _fakes(mult=1.0):
    v = [_vr("CHEAP", 8 * mult, 6 * mult, 12, nd=50), _vr("MID", 15 * mult, 12 * mult, 6), _vr("RICH", 30 * mult, 24 * mult, 2, nd=500),
         _vr("GROW", 12 * mult, 9 * mult, 8, nd=100), _vr("LAG", 28 * mult, 22 * mult, 3, nd=450)]
    c = [_cr("CHEAP", 4, 22), _cr("MID", 6, 15), _cr("RICH", 12, 12), _cr("GROW", 25, 18), _cr("LAG", -2, 8)]
    return (lambda t, d: {"rows": v, "errors": {}}), (lambda t, d: {"rows": c, "errors": {}})


def test_sector_snapshot_statistics_and_findings():
    from eqr import sector

    vf, cf = _fakes()
    snap = sector.snapshot("beauty", ["CHEAP", "MID", "RICH", "GROW", "LAG", "GONE"], TODAY, valuation_fn=vf, comps_fn=cf)
    assert snap["n"] == 5 and snap["stats"]["ev_ebit"]["median"] == 12.0 and snap["stats"]["ev_ebit"]["p25"] == 9.0
    assert snap["growing_share"] == 0.8 and snap["profitable_share"] == 1.0
    text = "\n".join(sector.findings(snap))
    assert "CHEAP (-50%)" in text and "RICH (+100%)" in text
    growth_line = next(x for x in sector.findings(snap) if "AND at or below" in x)
    assert "GROW" in growth_line and "RICH" not in growth_line and "LAG" not in growth_line
    lag_line = next(x for x in sector.findings(snap) if "premium (expectations" in x)
    assert "LAG" in lag_line and "GROW" not in lag_line
    assert "Leverage above 3x" in text and "RICH (5.0x)" in text


def test_sector_tracker_compares_with_the_previous_snapshot(data_dir):
    from eqr import sector

    vf, cf = _fakes()
    old = sector.snapshot("beauty", ["CHEAP", "MID", "RICH", "GROW", "LAG"], date(2026, 10, 1), valuation_fn=vf, comps_fn=cf)
    sector.save(old)
    vf2, cf2 = _fakes(mult=1.25)                                              # the whole sector re-rated 25% higher
    new = sector.snapshot("beauty", ["CHEAP", "MID", "RICH", "GROW", "LAG"], TODAY, valuation_fn=vf2, comps_fn=cf2)
    assert sector.previous("beauty", TODAY.isoformat())["date"] == "2026-10-01"
    assert sector.previous("beauty", "2026-10-01") is None and sector.previous("tobacco", TODAY.isoformat()) is None
    ch = sector.compare(new, old)
    assert ch["since"] == "2026-10-01" and ch["stat_changes"]["ev_ebit"]["change_pct"] == 25.0
    text = sector.render(new, old)
    assert "Since 2026-10-01: median EV/EBIT moved 12.0 -> 15.0" in text and "## Change since 2026-10-01" in text
    assert "First snapshot" in sector.render(old, None)


def test_user_defined_universes(data_dir):
    from eqr import sector

    assert sector.add_universe("Cannabis", ["tlry", "CGC", "tlry"]) == ["TLRY", "CGC"]
    assert [c["ticker"] for c in sector.all_universes()["cannabis"]] == ["TLRY", "CGC"] and "beauty" in sector.all_universes()
    with pytest.raises(ValueError):
        sector.add_universe("beauty", ["EL"])                                  # built-ins are not overwritten


def test_ideas_are_themed_explained_and_deduplicated(data_dir):
    from eqr import ideas

    vf, cf = _fakes()
    tick = ["CHEAP", "MID", "RICH", "GROW", "LAG"]
    res = ideas.generate(tick, "beauty", TODAY, valuation_fn=vf, comps_fn=cf, ratings={}, watchlist={}, existing=[])
    by = {i["ticker"]: i for i in res["ideas"]}
    assert "value" in by["CHEAP"]["themes"] and "cheap vs peers" in by["CHEAP"]["why"][0]
    assert "growth_at_discount" in by["GROW"]["themes"] and "RICH" not in by and "LAG" not in by
    # covered, watchlisted and already-open names are skipped, with the reason
    res2 = ideas.generate(tick, "beauty", TODAY, valuation_fn=vf, comps_fn=cf,
                          ratings={"CHEAP": _rating(rating="Hold", d="2026-11-10", t="CHEAP")}, watchlist={"GROW": {}},
                          existing=[])
    why = dict(res2["skipped"])
    assert "already rated Hold" in why["CHEAP"] and "watchlist" in why["GROW"]
    ideas.save(res["ideas"])
    res3 = ideas.generate(tick, "beauty", TODAY, valuation_fn=vf, comps_fn=cf, ratings={}, watchlist={})
    assert not res3["ideas"] and all(w == "already an open idea" for _, w in res3["skipped"])
    assert ideas.set_status("cheap", "dismissed") == 1
    assert [i["status"] for i in ideas.load() if i["ticker"] == "CHEAP"] == ["dismissed"]     # latest line wins
    assert "Research leads, not recommendations" in ideas.render(res)


def test_cli_wiring(data_dir, capsys):
    from eqr import track

    track.main(["watch", "add", "aaa", "--below", "90"])
    track.main(["watch", "list"])
    assert "AAA" in capsys.readouterr().out
    track.main(["sector", "list"])
    assert "beauty" in capsys.readouterr().out
    track.main(["buys"])                                                       # empty ratings log: renders, no crash
    assert "Buy recommendations" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        track.main(["thesis", "add", "ZZZ"])                                   # no saved run for it: clear exit, not a traceback
