"""Offline tests: no network, no API key."""
import pytest

from eqr.edgar import extract_section, select_annual
from eqr.graders import grade
from eqr.schemas import ResearchNote
from eqr.tools import calculate, collect_sources


def test_select_annual_prefers_current_period_over_comparatives():
    # FY2024 10-K reports FY2024 AND FY2023 numbers, both tagged fy=2024
    entries = [
        {"start": "2022-10-02", "end": "2023-09-30", "val": 383, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01", "accn": "a"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01", "accn": "a"},
        {"start": "2024-04-01", "end": "2024-06-29", "val": 85, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01", "accn": "b"},
    ]
    assert select_annual(entries, "duration")[2024]["val"] == 391


def test_select_annual_skips_non_annual_durations():
    entries = [{"start": "2024-01-01", "end": "2024-03-31", "val": 1, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "x", "accn": "a"}]
    assert select_annual(entries, "duration") == {}


def test_extract_section_skips_toc_and_cross_references():
    text = "\n".join([
        "Item 1. Business", "Item 1A. Risk Factors", "Item 1B. Unresolved Staff Comments",  # TOC
        "Item 1. Business", "We make widgets.", "Item 1A. Risk Factors",
        "Supply chains may break. " * 20, "Item 1B. Unresolved Staff Comments", "None.",
    ])
    risk = extract_section(text, "risk_factors")
    assert risk and "Supply chains" in risk and "widgets" not in risk
    assert "Supply chains" not in extract_section(text, "business")


def test_calculator():
    assert calculate("(391035-383285)/383285*100")["result"] == pytest.approx(2.022, abs=1e-3)
    with pytest.raises(ValueError):
        calculate("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        calculate("9**9999")


def _note(value, sources):
    return ResearchNote(
        ticker="AAPL", company="Apple", summary="s", thesis="t", bull_case=[], bear_case=[], key_risks=[],
        metrics=[{"name": "revenue", "period": "FY2024", "value": value, "unit": "USD", "sources": sources}],
    )


CASE = {"expected": [{"name": "revenue", "period": "FY2024", "value": 391035000000}]}


def test_grader_pass():
    g = grade(CASE, _note(391035000000, ["s1"]), {"sources_seen": ["s1"]})
    assert g["passed"] and g["reward"] == 1.0


def test_grader_catches_unit_error_and_fabricated_citation():
    g = grade(CASE, _note(391035, ["made-up"]), {"sources_seen": ["s1"]})
    assert not g["passed"] and g["metric_accuracy"] == 0 and g["citation_grounded"] == 0


def test_grader_no_note():
    assert grade(CASE, None, {})["reward"] == 0.0


def test_web_search_request_shape_and_sources(monkeypatch, tmp_path):
    from eqr import web

    monkeypatch.setattr(web, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    seen = {}

    class R:
        status_code = 200
        def json(self):
            return {"results": [{"url": "https://x.com/a", "title": "T", "publish_date": None, "excerpts": ["hi"]}]}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(web.httpx, "post", fake_post)
    out = web.web_search("Apple news", ["apple news"], max_results=3, after_date="2026-01-01")
    assert seen["headers"]["x-api-key"] == "test-key"
    assert "max_results" not in seen["body"]  # must be nested, top level 422s
    assert seen["body"]["advanced_settings"]["max_results"] == 3
    assert seen["body"]["advanced_settings"]["source_policy"] == {"after_date": "2026-01-01"}
    assert collect_sources(out) == {"web:https://x.com/a"}
    web.web_search("Apple news", ["apple news"], max_results=3, after_date="2026-01-01")  # served from cache


def test_web_search_requires_key(monkeypatch, tmp_path):
    from eqr import web

    monkeypatch.setattr(web, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PARALLEL_API_KEY"):
        web.web_search("q", ["q"])


def test_responses_translation_roundtrip():
    import json as _json
    from types import SimpleNamespace as NS
    from eqr.llm import from_responses_output, to_responses_input, to_responses_tools
    from eqr.tools import TOOLS

    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                          {"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "1+1"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}", "is_error": True}]},
    ]
    items = to_responses_input(msgs)
    assert [i.get("type") or i["role"] for i in items] == ["user", "assistant", "function_call", "function_call_output"]
    assert _json.loads(items[2]["arguments"]) == {"expression": "1+1"} and items[2]["call_id"] == "c1"
    assert items[3]["output"].startswith("ERROR: ")
    assert {t["name"] for t in to_responses_tools(TOOLS)} == {t["name"] for t in TOOLS}

    out = [NS(type="reasoning"),
           NS(type="message", content=[NS(type="output_text", text="hi")]),
           NS(type="function_call", call_id="c2", name="calculator", arguments='{"expression":"2*2"}'),
           NS(type="function_call", call_id="c3", name="x", arguments="{oops")]
    blocks = from_responses_output(out)
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1] == {"type": "tool_use", "id": "c2", "name": "calculator", "input": {"expression": "2*2"}}
    assert "_unparseable_arguments" in blocks[2]["input"]


def test_provider_routing():
    from eqr.llm import provider_for
    assert provider_for("claude-sonnet-5-5") == "anthropic" and provider_for("gpt-5.5") == "openai"


def test_judge_evidence_and_parsing():
    import json as _json
    from eqr.judge import evidence_from_trace, judge

    trace = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "is_error": False,
             "content": _json.dumps({"results": [{"source": "web:https://x.com", "excerpts": ["iPhone sales fell"]}]})}]}]}
    assert "iPhone sales fell" in evidence_from_trace(trace)["web:https://x.com"]

    note = ResearchNote(ticker="A", company="A", summary="s", thesis="t", metrics=[], key_risks=[], bear_case=[],
                        bull_case=[{"text": "sales fell", "sources": ["web:https://x.com"]}])
    seen = {}
    def fake(model, system, user):
        seen["user"] = user
        return 'sure:\n```json\n{"groundedness":4,"insight":3,"balance":5,"clarity":4,"unsupported_claims":[],"rationale":"ok"}\n```'
    r = judge(note, trace, model="fake", complete=fake)
    assert "iPhone sales fell" in seen["user"] and r.groundedness == 4 and r.overall == pytest.approx(0.75)


def test_collect_sources_nested():
    assert collect_sources({"a": [{"source": "x"}], "b": {"source": "y"}}) == {"x", "y"}


def _fin(rows_by_metric, company="Co"):
    return {"company": company, "not_found": [], "metrics": {
        m: [{"fiscal_year": fy, "period_end": end, "value": v, "unit": "USD", "source": f"s:{m}:{fy}"}
            for fy, end, v in rows] for m, rows in rows_by_metric.items()}}


def test_comps_math_staleness_and_errors(monkeypatch):
    from datetime import date
    from eqr import comps

    good = _fin({"revenue": [(2025, "2025-12-31", 1100), (2024, "2024-12-31", 1000)],
                 "gross_profit": [(2025, "2025-12-31", 660)], "operating_income": [(2025, "2025-12-31", 220)],
                 "net_income": [(2025, "2025-12-31", 110)], "operating_cash_flow": [(2025, "2025-12-31", 200)],
                 "capex": [(2025, "2025-12-31", 50)], "cash": [(2025, "2025-12-31", 100)],
                 "long_term_debt": [(2025, "2025-12-31", 400)]})
    stale = _fin({"revenue": [(2023, "2023-12-31", 500), (2022, "2022-12-31", 400)]})
    gap = _fin({"revenue": [(2025, "2025-12-31", 900), (2023, "2023-12-31", 600)]})  # FY2024 missing
    data = {"GOOD": good, "OLD": stale, "GAP": gap}

    def fake(ticker, metrics=None, years=5):
        if ticker not in data:
            raise ValueError("Unknown ticker")
        return data[ticker]

    monkeypatch.setattr(comps.edgar, "get_financials", fake)
    out = comps.get_comps(["GOOD", "OLD", "GAP", "NOPE"], today=date(2026, 3, 1))
    g = next(r for r in out["rows"] if r["ticker"] == "GOOD")["metrics"]
    assert g["revenue_growth_pct"]["value"] == 10.0 and g["gross_margin_pct"]["value"] == 60.0
    assert g["fcf"]["value"] == 150 and g["fcf_margin_pct"]["value"] == pytest.approx(13.64, abs=0.01)
    assert g["net_debt_proxy"]["value"] == 300
    assert next(r for r in out["rows"] if r["ticker"] == "OLD")["stale"]
    assert "revenue_growth_pct" not in next(r for r in out["rows"] if r["ticker"] == "GAP")["metrics"]  # no multi-year growth
    assert "NOPE" in out["errors"]
    assert out["medians"]["gross_margin_pct"]["n"] == 1  # stale and incomplete rows excluded
    seen = collect_sources(out)
    assert "s:revenue:2025" in seen and "edgar:comps:GOOD:2025-12-31" in seen


def test_peer_universe():
    from eqr.comps import peer_universe
    assert {"ELF", "EL", "ULTA"} <= {c["ticker"] for c in peer_universe("beauty")["companies"]}
    with pytest.raises(ValueError):
        peer_universe("unknown")


def _prices_search(excerpt, url="https://f.com/a"):
    return lambda **kw: {"results": [{"url": url, "title": "t", "publish_date": None, "excerpts": [excerpt]}]}


def test_price_guard_blocks_hallucinated_and_unquoted_prices():
    import json as _json
    from datetime import date
    from eqr import prices

    def ext(cands):
        return lambda m, s, u: _json.dumps({"candidates": cands})

    search = _prices_search("ELF Beauty Inc. (ELF) stock price 84.35 USD as of Nov 18")
    ok = prices.get_price("ELF", "e.l.f.", date(2026, 11, 19), search=search, extract=ext(
        [{"price": 84.35, "as_of": "2026-11-18", "market_cap_usd": 4.7e9, "url": "https://f.com/a", "quote": "stock price 84.35 USD"}]))
    assert ok["usable"] and ok["price"] == 84.35 and "single_source" in ok["flags"]
    # quote not in the page, wrong url, and quote lacking the number must all be rejected
    for bad in ({"price": 99.0, "quote": "stock price 99.00 USD", "url": "https://f.com/a"},
                {"price": 84.35, "quote": "stock price 84.35 USD", "url": "https://evil.com/x"},
                {"price": 84.35, "quote": "ELF Beauty Inc.", "url": "https://f.com/a"}):
        r = prices.get_price("ELF", "e.l.f.", date(2026, 11, 19), search=search, extract=ext([{**bad, "as_of": "2026-11-18"}]))
        assert not r["usable"] and "no_verified_price" in r["flags"]


def test_price_conflict_and_staleness():
    import json as _json
    from datetime import date
    from eqr import prices

    search = lambda **kw: {"results": [
        {"url": "https://a.com", "title": "", "publish_date": None, "excerpts": ["price 80.00"]},
        {"url": "https://b.com", "title": "", "publish_date": None, "excerpts": ["price 90.00"]}]}
    def ext(d1, d2):
        return lambda m, s, u: _json.dumps({"candidates": [
            {"price": 80.0, "as_of": d1, "url": "https://a.com", "quote": "price 80.00"},
            {"price": 90.0, "as_of": d2, "url": "https://b.com", "quote": "price 90.00"}]})
    r = prices.get_price("X", "X", date(2026, 11, 19), search=search, extract=ext("2026-11-18", "2026-11-18"))
    assert "price_conflict" in r["flags"] and not r["usable"]
    agree = lambda **kw: {"results": [{"url": "https://a.com", "title": "", "publish_date": None, "excerpts": ["price 80.00"]},
                                      {"url": "https://b.com", "title": "", "publish_date": None, "excerpts": ["price 80.50"]}]}
    ext2 = lambda m, s, u: _json.dumps({"candidates": [
        {"price": 80.0, "as_of": "2026-11-01", "url": "https://a.com", "quote": "price 80.00"},
        {"price": 80.5, "as_of": "2026-11-01", "url": "https://b.com", "quote": "price 80.50"}]})
    assert "stale_price" in prices.get_price("X", "X", date(2026, 11, 19), search=agree, extract=ext2)["flags"]


def test_valuation_math_and_withholding(monkeypatch):
    from datetime import date
    from eqr import valuation as v

    def metric(val, unit="USD", src="s"):
        return {"value": val, "unit": unit, "sources": [f"edgar:{src}"]}
    base = {"revenue": metric(1000, src="rev"), "operating_income": metric(100, src="oi"),
            "eps_diluted": metric(2.0, "USD/shares", "eps"), "fcf": metric(50, src="fcf"),
            "net_debt_proxy": metric(200, src="nd")}
    comps_fn = lambda t, today: {"company": t, "metrics": {**base} if t != "LOSS" else {**base, "eps_diluted": metric(-1, "USD/shares", "eps")},
                                 "period_end": "2025-12-31", "stale": False}
    shares_fn = lambda t: {"shares": 10.0, "as_of": "2026-02-01", "multi_class": False, "source": f"edgar:shares:{t}"}
    good = {"price": 30.0, "as_of": "2026-11-18", "market_cap_reported": 300.0, "sources": ["web:https://a"], "flags": [], "usable": True}
    prices_by = {"GOOD": good, "LOSS": good, "BADCAP": {**good, "market_cap_reported": 900.0},
                 "CONF": {**good, "flags": ["price_conflict"], "usable": False}}
    monkeypatch.setattr(v.edgar, "lookup", lambda t: (1, t))
    out = v.get_valuation(list(prices_by), date(2026, 11, 19), price_fn=lambda t, n, d: prices_by[t],
                          shares_fn=shares_fn, comps_fn=comps_fn)
    rows = {r["ticker"]: r for r in out["rows"]}
    m = rows["GOOD"]["metrics"]
    assert m["market_cap"]["value"] == 300 and m["enterprise_value"]["value"] == 500
    assert m["ev_sales"]["value"] == 0.5 and m["ev_ebit"]["value"] == 5.0 and m["pe"]["value"] == 15.0
    assert m["fcf_yield_pct"]["value"] == pytest.approx(16.67, abs=0.01)
    assert "pe" not in rows["LOSS"]["metrics"] and "ev_sales" in rows["LOSS"]["metrics"]  # negative EPS -> n/m
    assert rows["BADCAP"]["withheld"] and not rows["BADCAP"]["metrics"]
    assert rows["CONF"]["withheld"]
    assert out["medians"]["ev_sales"]["n"] == 2  # only GOOD and LOSS contribute
    assert {"web:https://a", "edgar:rev", "edgar:shares:GOOD"} <= collect_sources(out)


def test_older_quotes_are_ignored_not_conflicts():
    import json as _json
    from datetime import date
    from eqr import prices

    search = lambda **kw: {"results": [
        {"url": "https://a.com", "title": "", "publish_date": None, "excerpts": ["price 4.21"]},
        {"url": "https://b.com", "title": "", "publish_date": None, "excerpts": ["price 4.10"]}]}
    ext = lambda m, s, u: _json.dumps({"candidates": [
        {"price": 4.21, "as_of": "2026-11-18", "url": "https://a.com", "quote": "price 4.21"},
        {"price": 4.10, "as_of": "2026-11-04", "url": "https://b.com", "quote": "price 4.10"}]})
    r = prices.get_price("X", "X", date(2026, 11, 19), search=search, extract=ext)
    assert r["usable"] and r["price"] == 4.21 and r["as_of"] == "2026-11-18"
    assert any(f.startswith("older_or_undated_quotes_ignored") for f in r["flags"])


def test_comps_assumes_zero_debt_only_when_flagged(monkeypatch):
    from datetime import date
    from eqr import comps

    fin = _fin({"revenue": [(2025, "2025-12-31", 1000), (2024, "2024-12-31", 900)], "cash": [(2025, "2025-12-31", 120)]})
    fin["not_found"] = ["long_term_debt"]
    monkeypatch.setattr(comps.edgar, "get_financials", lambda *a, **k: fin)
    row = comps._company("X", date(2026, 3, 1))
    assert row["metrics"]["net_debt_proxy"]["value"] == -120 and "debt_assumed_zero" in row["flags"]
    fin["metrics"]["cash"] = []  # without cash we must not invent a net debt figure
    assert "net_debt_proxy" not in comps._company("X", date(2026, 3, 1))["metrics"]


def test_majority_cluster_drops_a_stray_quote_but_two_way_split_conflicts():
    import json as _json
    from datetime import date
    from eqr import prices

    def run(vals):
        urls = [f"https://s{i}.com" for i in range(len(vals))]
        search = lambda **kw: {"results": [{"url": u, "title": "", "publish_date": None, "excerpts": [f"price {v:.2f}"]}
                                           for u, v in zip(urls, vals)]}
        ext = lambda m, s, u: _json.dumps({"candidates": [
            {"price": v, "as_of": "2026-11-19", "url": u, "quote": f"price {v:.2f}"} for u, v in zip(urls, vals)]})
        return prices.get_price("X", "X", date(2026, 11, 19), search=search, extract=ext)
    r = run([101.77, 101.77, 104.04])
    assert r["usable"] and r["price"] == 101.77 and "outlier_quote_dropped:1" in r["flags"]
    r2 = run([100.0, 104.0])
    assert not r2["usable"] and "price_conflict" in r2["flags"]
