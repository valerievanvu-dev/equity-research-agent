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
    r2 = run([100.0, 110.0])            # 10% apart with no majority: a real conflict
    assert not r2["usable"] and "price_conflict" in r2["flags"]
    r3 = run([199.20, 205.72])          # 3.2% of the midpoint apart, same day (delayed vs live quote): usable, midpoint, flagged
    assert r3["usable"] and r3["price"] == 202.46 and any(f.startswith("price_uncertain:3.2") for f in r3["flags"])
    assert "price_conflict" not in r3["flags"]


def test_note_prose_is_html_unescaped_but_source_ids_are_not():
    url_source = "web:https://x.com/a?x=1&amp;y=2"
    n = ResearchNote(
        ticker="X", company="X", summary="S&amp;P exposure", thesis="SG&amp;A rose; &lt;5% growth", metrics=[],
        bull_case=[{"text": "R&amp;D up", "sources": [url_source]}], bear_case=[], key_risks=[],
        peer_commentary="P&amp;G is bigger", open_questions=["A&amp;B unknown"])
    assert n.summary == "S&P exposure" and n.thesis == "SG&A rose; <5% growth"
    assert n.bull_case[0].text == "R&D up" and n.peer_commentary == "P&G is bigger"
    assert n.open_questions == ["A&B unknown"]
    assert n.bull_case[0].sources == [url_source]  # untouched, so it still matches the tool result


# ------------------------------------------------------------------ PDF, citations, email
def _fake_trace():
    import json as _json

    def use(i, name, inp=None):
        return {"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": name, "input": inp or {}}]}

    def res(i, obj, err=False):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": _json.dumps(obj), "is_error": err}]}

    fin_row = lambda fy, val, m: {"fiscal_year": fy, "period_end": f"{fy}-12-31", "value": val, "unit": "USD",
                                  "source": f"edgar:facts:AAA:{m}:FY{fy}:0000000001-{fy % 100}-000001"}
    fin = {"company": "Alpha Co", "not_found": [], "metrics": {
        "revenue": [fin_row(2024, 1_000_000_000, "Revenues"), fin_row(2023, 800_000_000, "Revenues")],
        "net_income": [fin_row(2024, 100_000_000, "NetIncomeLoss"), fin_row(2023, 80_000_000, "NetIncomeLoss")]}}
    comps_row = {"ticker": "AAA", "period_end": "2024-12-31", "stale": False, "source": "edgar:comps:AAA:2024-12-31",
                 "metrics": {"revenue": {"value": 1e9, "unit": "USD", "sources": [fin["metrics"]["revenue"][0]["source"]]},
                             "revenue_growth_pct": {"value": 25.0, "unit": "percent", "sources": [fin["metrics"]["revenue"][0]["source"], fin["metrics"]["revenue"][1]["source"]]}}}
    val_row = {"ticker": "AAA", "price": 50.0, "price_date": "2026-11-19", "flags": ["single_source"], "withheld": False,
               "source": "calc:valuation:AAA:2026-11-19", "sources": ["web:https://quotes.example/aaa", "edgar:facts:AAA:dei:EntityCommonStockSharesOutstanding:0000000001-24-000001"],
               "metrics": {"market_cap": {"value": 5e9, "unit": "USD"}, "pe": {"value": 20.0, "unit": "x"}}}
    return {"model": "test-model", "messages": [
        use("1", "get_financials"), res("1", fin), use("2", "get_comps"),
        res("2", {"rows": [comps_row], "medians": {}, "errors": {}, "notes": ["Fiscal years are not calendar aligned."]}),
        use("3", "get_valuation"), res("3", {"rows": [val_row], "medians": {}, "errors": {}, "notes": []}),
        use("4", "web_search"), res("4", {"results": [{"url": "https://quotes.example/aaa", "title": "AAA quote page", "excerpts": []}]}),
        use("5", "get_comps"), res("5", {"x": 1}, err=True)]}


def test_refs_expand_derived_ids_and_dedupe():
    from eqr.citations import Refs, artifacts_from_trace

    a = artifacts_from_trace(_fake_trace())
    assert a.company == "Alpha Co" and set(a.financials["revenue"]) == {2023, 2024}
    refs = Refs(a.inputs_of, a.titles, filing_info=lambda t, acc: (f"{t} filing {acc}", f"https://sec.example/{acc}"))
    nums = refs.cite(["edgar:comps:AAA:2024-12-31"])
    assert len(nums) == 2  # two filings behind the comps row (FY2024 + FY2023 for growth)
    assert refs.cite(["edgar:comps:AAA:2024-12-31"]) == nums  # stable numbering
    v = refs.cite(["calc:valuation:AAA:2026-11-19"])
    assert any(i["url"] == "https://quotes.example/aaa" and i["label"] == "AAA quote page" for i in refs.items)
    assert len(v) == 2


def test_pdf_tables_come_from_tool_data_and_citations_resolve(tmp_path):
    from pypdf import PdfReader
    from eqr.pdf import build_pdf

    note = ResearchNote(ticker="AAA", company="Alpha Co", summary="Alpha grew.", thesis="Neutral. R&D rises.",
                        metrics=[], bull_case=[{"text": "Growth 25% (edgar:comps:AAA:2024-12-31).", "sources": ["edgar:comps:AAA:2024-12-31"]}],
                        bear_case=[], key_risks=[], open_questions=["No EPS."])
    out = build_pdf(note, _fake_trace(), tmp_path / "x.pdf", filing_info=lambda t, acc: (f"{t} 10-K {acc}", f"https://sec.example/{acc}"))
    text = "\n".join(p.extract_text() for p in PdfReader(str(out)).pages)
    assert out.read_bytes()[:4] == b"%PDF"
    assert "1,000.0" in text and "800.0" in text      # revenue from tool data, in $M
    assert "25.0%" in text and "10.0%" in text         # derived growth / net margin computed in code
    assert "5,000.0" in text and "20.0x" in text       # valuation row
    assert "single_source" in text and "Fiscal years are not calendar aligned" in text
    assert "R&D rises" in text                          # prose escaped correctly
    assert "edgar:comps" not in text                    # raw ids replaced by [n] markers
    assert "References" in text and "https://sec.example/" in text and "quotes.example" in text


def test_mailer_sends_pdf_and_requires_config(tmp_path):
    from eqr import mailer

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port): sent["host"], sent["port"] = host, port
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, u, p): sent["login"] = (u, p)
        def send_message(self, m): sent["msg"] = m

    env = {"EQR_EMAIL_TO": "me@example.com", "EQR_SMTP_USER": "u", "EQR_SMTP_PASSWORD": "p"}
    assert mailer.send_report(pdf, "subj", "body", env=env, smtp_factory=FakeSMTP) == "me@example.com"
    msg = sent["msg"]
    assert msg["To"] == "me@example.com" and msg["Subject"] == "subj" and sent["port"] == 587
    assert [a.get_filename() for a in msg.iter_attachments()] == ["r.pdf"]
    with pytest.raises(mailer.NotConfigured, match="EQR_EMAIL_TO"):
        mailer.send_report(pdf, "s", "b", env={}, smtp_factory=FakeSMTP)


def test_macos_mail_method_passes_args_safely_and_reports_failures(tmp_path):
    from types import SimpleNamespace as NS
    from eqr import mailer

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"], calls["kw"] = cmd, kw
        return NS(returncode=0, stderr="")

    env = {"EQR_EMAIL_METHOD": "macos-mail", "EQR_EMAIL_TO": "me@example.com"}
    tricky = 'Subject with "quotes" & a \\ backslash'
    assert mailer.send_report(pdf, tricky, "body", env=env, run=fake_run, platform="darwin") == "me@example.com"
    cmd = calls["cmd"]
    assert cmd[:2] == ["osascript", "-"] and cmd[2] == tricky and cmd[4] == "me@example.com"  # args via argv, not interpolated
    assert cmd[5] == str(pdf.resolve()) and cmd[6] == ""
    assert 'tell application "Mail"' in calls["kw"]["input"] and tricky not in calls["kw"]["input"]

    bad = lambda cmd, **kw: NS(returncode=1, stderr="Not authorized to send Apple events to Mail.")
    with pytest.raises(RuntimeError, match="Not authorized"):
        mailer.send_report(pdf, "s", "b", env=env, run=bad, platform="darwin")
    with pytest.raises(mailer.NotConfigured, match="macOS"):
        mailer.send_report(pdf, "s", "b", env=env, run=fake_run, platform="linux")
    with pytest.raises(mailer.NotConfigured, match="EQR_EMAIL_TO"):
        mailer.send_report(pdf, "s", "b", env={"EQR_EMAIL_METHOD": "macos-mail"}, run=fake_run, platform="darwin")


def test_resend_method_payload_headers_and_errors(tmp_path):
    import base64 as _b64
    from types import SimpleNamespace as NS
    from eqr import mailer

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4 resend test")
    seen = {}

    def ok(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return NS(status_code=200, json=lambda: {"id": "abc"}, text="")

    env = {"EQR_EMAIL_METHOD": "resend", "EQR_EMAIL_TO": "me@example.com", "RESEND_API_KEY": "re_secret123"}
    assert mailer.send_report(pdf, "subj", "body", env=env, post=ok) == "me@example.com"
    assert seen["url"] == "https://api.resend.com/emails"
    assert seen["headers"]["Authorization"] == "Bearer re_secret123"
    b = seen["body"]
    assert b["to"] == ["me@example.com"] and b["subject"] == "subj" and "onboarding@resend.dev" in b["from"]
    att = b["attachments"][0]
    assert att["filename"] == "r.pdf" and _b64.b64decode(att["content"]) == pdf.read_bytes()

    forbidden = lambda url, headers, json, timeout: NS(
        status_code=403, text="", json=lambda: {"name": "validation_error", "message": "You can only send testing emails to your own email address."})
    with pytest.raises(RuntimeError, match="403.*own email address") as e:
        mailer.send_report(pdf, "s", "b", env=env, post=forbidden)
    assert "re_secret123" not in str(e.value)  # the key must never leak into errors

    junk = lambda url, headers, json, timeout: NS(status_code=500, text="boom", json=lambda: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(RuntimeError, match="500.*boom"):
        mailer.send_report(pdf, "s", "b", env=env, post=junk)
    for missing in ({"EQR_EMAIL_METHOD": "resend", "EQR_EMAIL_TO": "me@example.com"},
                    {"EQR_EMAIL_METHOD": "resend", "RESEND_API_KEY": "k"}):
        with pytest.raises(mailer.NotConfigured):
            mailer.send_report(pdf, "s", "b", env=missing, post=ok)


# ------------------------------------------------------------------ price target and rating
def _val(price=40.0, eps=4.0, dps=2.0, oi=1000.0, nd=2000.0, shares=100.0, pe=10.0, ev_ebit=9.0, withheld=False,
         med_pe=11.0, med_n=3, flags=None, rev=1000.0, ev_sales=2.0):
    row = {"ticker": "AAA", "price": price, "price_date": "2026-11-19", "withheld": withheld, "flags": flags or [],
           "source": "calc:valuation:AAA:2026-11-19", "sources": ["web:https://q.example/a", "edgar:facts:AAA:EPS:FY2025:0001-26-000001"],
           "metrics": {"pe": {"value": pe}, "ev_ebit": {"value": ev_ebit}, "ev_sales": {"value": ev_sales}},
           "fundamentals": {"eps_diluted": eps, "operating_income": oi, "net_debt_proxy": nd, "dps": dps, "shares": shares, "revenue": rev}}
    medians = {"pe": {"value": med_pe, "n": med_n}, "ev_ebit": {"value": 9.5, "n": med_n}, "ev_sales": {"value": 3.0, "n": med_n}}
    return lambda tickers, today=None: {"rows": [row], "medians": medians, "errors": {}}


SCEN = {"bear": {"growth_pct": -5, "multiple": 9, "rationale": "bear"},
        "base": {"growth_pct": 3, "multiple": 11, "rationale": "base"},
        "bull": {"growth_pct": 8, "multiple": 13, "rationale": "bull"}}


def test_rating_thresholds():
    from eqr.target import rating_for
    assert [rating_for(x) for x in (15.0, 14.9, 0, -9.9, -10.0, -30)] == ["Buy", "Hold", "Hold", "Hold", "Sell", "Sell"]


def test_price_target_pe_math_and_rating():
    from datetime import date
    from eqr.target import get_price_target

    r = get_price_target("aaa", ["PEER"], "pe", SCEN, today=date(2026, 11, 19), valuation_fn=_val())
    # 4*.95*9=34.2, 4*1.03*11=45.32, 4*1.08*13=56.16 -> .25/.5/.25 weighted = 45.25
    assert [s["price"] for s in r["scenarios"]] == [34.2, 45.32, 56.16] and r["price_target"] == 45.25
    assert r["dividend_yield_pct"] == 5.0 and r["expected_price_return_pct"] == 13.1
    assert r["expected_total_return_pct"] == 18.1 and r["rating"] == "Buy" and r["conviction"] == "Normal"
    assert r["flags"] == [] and r["source"] == "calc:target:AAA:2026-11-19"
    assert "web:https://q.example/a" in r["sources"] and not any(x.startswith("calc:valuation") for x in r["sources"])
    mid = r["sensitivity"]["rows"][2]["prices"][2]
    assert len(r["sensitivity"]["rows"]) == 5 and mid == r["scenarios"][1]["price"]  # grid centre = base case


def test_price_target_ev_ebit_path_and_dividend_effect():
    from eqr.target import get_price_target

    sc = {k: {**v, "growth_pct": 0, "multiple": m} for (k, v), m in zip(SCEN.items(), (8, 10, 12))}
    r = get_price_target("AAA", [], "ev_ebit", sc, valuation_fn=_val(price=80.0, dps=None))
    # (1000*10 - 2000)/100 = 80 base; bear 60; bull 100 -> target 80; no dividend known
    assert r["price_target"] == 80.0 and r["expected_total_return_pct"] == 0.0 and r["rating"] == "Hold"
    assert "dividend_unknown" in r["flags"] and r["conviction"] == "Low"


def test_price_target_guardrails():
    from eqr.target import get_price_target
    import copy

    def run(sc=None, method="pe", **kw):
        return get_price_target("AAA", ["P"], method, sc or SCEN, valuation_fn=_val(**kw))
    bad = copy.deepcopy(SCEN); bad["bear"]["multiple"] = 14        # bear above base
    with pytest.raises(ValueError, match="ordered"):
        run(bad)
    wild = copy.deepcopy(SCEN); wild["bull"]["multiple"] = 90
    with pytest.raises(ValueError, match="multiple must be between"):
        run(wild)
    wild2 = copy.deepcopy(SCEN); wild2["base"]["growth_pct"] = 400
    with pytest.raises(ValueError, match="growth_pct"):
        run(wild2)
    with pytest.raises(ValueError, match="exactly"):
        run({"base": SCEN["base"]})
    w = copy.deepcopy(SCEN)
    for k, x in zip(w, (0.1, 0.1, 0.1)):
        w[k]["weight"] = x
    with pytest.raises(ValueError, match="weights"):
        run(w)
    with pytest.raises(ValueError, match="not positive"):
        run(eps=-1.0)
    with pytest.raises(ValueError, match="price unavailable"):
        run(withheld=True, flags=["price_conflict"])
    with pytest.raises(ValueError, match="net debt"):
        run(method="ev_ebit", nd=None)


def test_unanchored_multiple_and_thin_peers_lower_conviction():
    from eqr.target import get_price_target

    sc = {"bear": {"growth_pct": -5, "multiple": 18, "rationale": "x"}, "base": {"growth_pct": 3, "multiple": 20, "rationale": "x"},
          "bull": {"growth_pct": 8, "multiple": 22, "rationale": "x"}}
    r = get_price_target("AAA", ["P"], "pe", sc, valuation_fn=_val(pe=10.0, med_pe=11.0, med_n=1))
    assert {"multiple_not_anchored", "few_peers"} <= set(r["flags"]) and r["conviction"] == "Low"


def _rec_note(rating="Buy", pt=45.25):
    return ResearchNote(ticker="AAA", company="A", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[], key_risks=[],
                        recommendation={"rating": rating, "price_target": pt, "current_price": 40.0, "expected_total_return_pct": 18.1,
                                        "method": "pe", "conviction": "Normal", "rationale": "r", "sources": ["calc:target:AAA:2026-11-19"],
                                        "scenarios": [{"name": "base", "weight": 0.5, "growth_pct": 3, "multiple": 11, "price": 45.32, "rationale": "b"}]})


def _target_trace(rating="Buy", pt=45.25):
    import json as _json
    res = {"rating": rating, "price_target": pt, "current_price": 40.0, "expected_total_return_pct": 18.1}
    return {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "get_price_target", "input": {}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": _json.dumps(res), "is_error": False}]}]}


def test_recommendation_must_match_computed_result():
    from eqr.graders import recommendation_consistent

    assert recommendation_consistent(_rec_note(), _target_trace()) == 1.0
    assert recommendation_consistent(_rec_note(rating="Sell"), _target_trace()) == 0.0   # note changed the rating
    assert recommendation_consistent(_rec_note(pt=60.0), _target_trace()) == 0.0         # note changed the target
    assert recommendation_consistent(_rec_note(), {"messages": []}) == 0.0               # rating with no computed basis
    plain = ResearchNote(ticker="A", company="A", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[], key_risks=[])
    assert recommendation_consistent(plain, {"messages": []}) is None
    assert recommendation_consistent(plain, _target_trace()) == 0.0                      # computed a target but dropped it


def test_pdf_shows_rating_box_and_derivation(tmp_path):
    import json as _json
    from datetime import date
    from pypdf import PdfReader
    from eqr.pdf import build_pdf
    from eqr.target import get_price_target

    res = get_price_target("AAA", ["P"], "pe", SCEN, today=date(2026, 11, 19), valuation_fn=_val())
    trace = _fake_trace()
    trace["messages"] += [{"role": "assistant", "content": [{"type": "tool_use", "id": "pt", "name": "get_price_target", "input": {}}]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "pt", "content": _json.dumps(res), "is_error": False}]}]
    note = ResearchNote(ticker="AAA", company="Alpha Co", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[], key_risks=[],
                        recommendation={"rating": res["rating"], "price_target": res["price_target"], "current_price": res["current_price"],
                                        "price_date": res["price_date"], "expected_total_return_pct": res["expected_total_return_pct"],
                                        "method": "pe", "conviction": res["conviction"], "scenarios": res["scenarios"],
                                        "rationale": "Dividend plus modest growth. R&amp;D steady.", "sources": res["sources"]})
    out = build_pdf(note, trace, tmp_path / "rec.pdf", filing_info=lambda t, acc: (f"{t} 10-K {acc}", f"https://sec.example/{acc}"))
    text = "\n".join(p.extract_text() for p in PdfReader(str(out)).pages)
    assert "BUY" in text and "$45.25" in text and "18.1%" in text and "12-MONTH PRICE TARGET" in text
    assert "Rating rule (fixed in code)" in text and "Probability-weighted target" in text
    assert "Sensitivity" in text and "$45.32" in text            # base-case centre of the grid
    assert "Not investment advice" in text or "not a recommendation to buy or sell" in text
    assert "R&D steady" in text and "calc:target" not in text      # entity decoded; ids replaced by markers


def test_unresolvable_derived_ids_do_not_become_references():
    from eqr.citations import Refs

    refs = Refs({"calc:valuation:X:1": {"web:https://p.example/x"}}, {}, filing_info=lambda t, a: (f"{t} {a}", None))
    assert refs.cite(["calc:valuation:X:1"]) == [1]
    assert refs.cite(["calc:valuation:UNKNOWN:9", "edgar:comps:UNKNOWN:9"]) == []   # silently dropped
    assert [i["label"] for i in refs.items] == ["p.example"]


# ------------------------------------------------------------------ OpenAI Agents SDK engine
def _valid_note_json():
    return ResearchNote(ticker="AAA", company="A", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[],
                        key_risks=[]).model_dump_json()


def test_sdk_tools_mirror_ours_and_report_errors_to_the_model():
    import asyncio
    import json as _json
    from eqr.sdk_engine import RunState, make_tools
    from eqr.tools import TOOLS

    state = RunState()
    tools = {t.name: t for t in make_tools(state)}
    assert set(tools) == {t["name"] for t in TOOLS}                                   # one definition of each tool
    assert tools["get_comps"].params_json_schema == next(t for t in TOOLS if t["name"] == "get_comps")["input_schema"]
    ok = asyncio.run(tools["calculator"].on_invoke_tool(None, '{"expression": "6*7"}'))
    assert _json.loads(ok)["result"] == 42
    bad = asyncio.run(tools["calculator"].on_invoke_tool(None, '{"expression": "__import__(1)"}'))
    assert bad.startswith("ERROR: ")                                                    # errors are observations, not crashes
    assert [c["is_error"] for c in state.tool_log] == [False, True]


def test_sdk_submit_note_validates_and_finish_rule_waits_for_acceptance():
    import asyncio
    from eqr.sdk_engine import RunState, finish_when_note_accepted, make_tools

    state = RunState()
    submit = {t.name: t for t in make_tools(state)}["submit_note"]
    behavior = finish_when_note_accepted(state)
    msg = asyncio.run(submit.on_invoke_tool(None, '{"ticker": "AAA"}'))                # invalid: missing fields
    assert msg.startswith("ERROR: Invalid note") and state.note is None
    assert behavior(None, []).is_final_output is False                                  # model gets to repair
    assert asyncio.run(submit.on_invoke_tool(None, _valid_note_json())) == "Note accepted."
    assert state.note.ticker == "AAA" and behavior(None, []).is_final_output is True


def test_sdk_items_convert_to_canonical_trace_the_rest_of_the_project_reads():
    import json as _json
    from types import SimpleNamespace as NS
    from eqr.citations import tool_calls
    from eqr.sdk_engine import items_to_messages

    items = [
        NS(type="reasoning_item", raw_item=NS()),                                       # ignored
        NS(type="message_output_item", raw_item=NS(content=[NS(type="output_text", text="Checking.")])),
        NS(type="tool_call_item", raw_item=NS(call_id="c1", name="calculator", arguments='{"expression": "1+1"}')),
        NS(type="tool_call_output_item", raw_item={"call_id": "c1", "output": _json.dumps({"result": 2})}),
        NS(type="tool_call_item", raw_item=NS(call_id="c2", name="get_comps", arguments="{oops")),
        NS(type="tool_call_output_item", raw_item={"call_id": "c2", "output": "ERROR: ValueError: bad"}),
    ]
    msgs = items_to_messages("task", items)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "text", "text": "Checking."}
    assert msgs[1]["content"][1]["input"] == {"expression": "1+1"}
    assert "_unparseable_arguments" in msgs[3]["content"][0]["input"]
    assert [b["is_error"] for b in msgs[2]["content"] + msgs[4]["content"]] == [False, True]
    assert [(n, r) for n, _i, r in tool_calls({"messages": msgs})] == [("calculator", {"result": 2})]   # errors skipped


def test_engine_dispatch_sdk_for_openai_models_loop_for_claude(monkeypatch):
    import eqr.agent as agent
    import eqr.sdk_engine as sdk

    called = {}
    monkeypatch.setattr(sdk, "run_sdk", lambda task, **kw: called.setdefault("sdk", kw["model"]) or "sdk-result")
    assert agent.run("t", ticker="X", model="gpt-5.5", trace_dir=None) == "gpt-5.5"      # default engine is the SDK
    assert called["sdk"] == "gpt-5.5"

    class Boom(Exception):
        pass

    def fake_backend(model):
        raise Boom(model)
    monkeypatch.setattr(agent, "get_backend", fake_backend)
    with pytest.raises(Boom):                                                            # claude models skip the SDK
        agent.run("t", ticker="X", model="claude-sonnet-5-5", trace_dir=None)
    with pytest.raises(Boom):                                                            # and --engine loop forces the loop
        agent.run("t", ticker="X", model="gpt-5.5", engine="loop", trace_dir=None)


# ------------------------------------------------------------------ ev_sales for loss-making companies
LOSS = dict(price=50.0, eps=-2.0, oi=-100.0, nd=-500.0, shares=100.0, dps=None, rev=1000.0, ev_sales=2.0)
EVS = {"bear": {"growth_pct": 0, "multiple": 2, "rationale": "stalls"}, "base": {"growth_pct": 20, "multiple": 4, "rationale": "grows"},
       "bull": {"growth_pct": 40, "multiple": 6, "rationale": "scales"}}


def test_ev_sales_target_for_a_loss_maker_is_computed_and_low_conviction():
    from eqr.target import get_price_target

    r = get_price_target("AAA", ["P"], "ev_sales", EVS, valuation_fn=_val(**LOSS))
    # (revenue*(1+g)*multiple - net debt)/shares with net cash of 500: 25.0 / 53.0 / 89.0 -> weighted 55.0
    assert [s["price"] for s in r["scenarios"]] == [25.0, 53.0, 89.0] and r["price_target"] == 55.0
    assert r["growth_basis"] == "revenue" and r["rating"] == "Hold" and r["expected_total_return_pct"] == 10.0
    assert "loss_making_revenue_multiple" in r["flags"] and r["conviction"] == "Low"
    assert "dividend_unknown" not in r["flags"]                     # a loss-maker with no dps data is not "unknown"
    assert any("cash burn" in n for n in r["notes"])


def test_ev_sales_guardrails_and_profitable_companies_are_not_forced_low_conviction():
    import copy
    from eqr.target import get_price_target

    bad = copy.deepcopy(EVS); bad["bear"]["multiple"] = 0.1
    with pytest.raises(ValueError, match="between 0.3"):
        get_price_target("AAA", [], "ev_sales", bad, valuation_fn=_val(**LOSS))
    fast = copy.deepcopy(EVS); fast["bull"]["growth_pct"] = 120      # allowed for revenue growth, not for EPS growth
    assert get_price_target("AAA", [], "ev_sales", fast, valuation_fn=_val(**LOSS))["scenarios"][2]["growth_pct"] == 120
    with pytest.raises(ValueError, match="growth_pct must be between"):
        get_price_target("AAA", [], "pe", {**SCEN, "bull": {**SCEN["bull"], "growth_pct": 120}}, valuation_fn=_val())
    with pytest.raises(ValueError, match="revenue is missing or not positive"):
        get_price_target("AAA", [], "ev_sales", EVS, valuation_fn=_val(**{**LOSS, "rev": None}))
    with pytest.raises(ValueError, match="use method 'ev_sales'"):                        # P/E on a loss-maker points to the fix
        get_price_target("AAA", [], "pe", SCEN, valuation_fn=_val(**LOSS))
    ok = get_price_target("AAA", [], "ev_sales", EVS, valuation_fn=_val(price=50.0, eps=4.0, oi=500.0, nd=-500.0, dps=1.0))
    assert "loss_making_revenue_multiple" not in ok["flags"]       # profitable: the flag is for loss-makers only


def test_pdf_labels_follow_the_valuation_method(tmp_path):
    import json as _json
    from datetime import date
    from pypdf import PdfReader
    from eqr.pdf import build_pdf
    from eqr.target import get_price_target

    res = get_price_target("AAA", ["P"], "ev_sales", EVS, today=date(2026, 11, 19), valuation_fn=_val(**LOSS))
    trace = _fake_trace()
    trace["messages"] += [{"role": "assistant", "content": [{"type": "tool_use", "id": "pt", "name": "get_price_target", "input": {}}]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "pt", "content": _json.dumps(res), "is_error": False}]}]
    note = ResearchNote(ticker="AAA", company="Alpha Co", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[], key_risks=[])
    out = build_pdf(note, trace, tmp_path / "evs.pdf", filing_info=lambda t, a: (f"{t} {a}", None))
    text = "\n".join(p.extract_text() for p in PdfReader(str(out)).pages)
    assert "Revenue growth" in text and "Target EV/Sales" in text and "EV/SALES" in text and "Low" in text
    assert "ignore cash burn" in text


def test_price_guard_previous_day_corroborates_but_never_conflicts():
    import json as _json
    from datetime import date
    from eqr import prices

    def run(p_new, p_old):
        search = lambda **kw: {"results": [
            {"url": "https://a.com", "title": "", "publish_date": None, "excerpts": [f"price {p_new:.2f}"]},
            {"url": "https://b.com", "title": "", "publish_date": None, "excerpts": [f"price {p_old:.2f}"]}]}
        ext = lambda m, s, u: _json.dumps({"candidates": [
            {"price": p_new, "as_of": "2026-11-18", "url": "https://a.com", "quote": f"price {p_new:.2f}"},
            {"price": p_old, "as_of": "2026-11-17", "url": "https://b.com", "quote": f"price {p_old:.2f}"}]})
        return prices.get_price("X", "X", date(2026, 11, 19), search=search, extract=ext)
    volatile = run(196.79, 204.58)      # a 4% overnight move: yesterday's close is not a conflicting quote
    assert volatile["usable"] and volatile["price"] == 196.79 and "single_source" in volatile["flags"]
    assert "price_conflict" not in volatile["flags"]
    agree = run(100.0, 100.8)           # roughly agrees: kept as corroboration, so not a single source
    assert agree["usable"] and "single_source" not in agree["flags"] and len(agree["sources"]) == 2


def test_uncertain_price_forces_low_conviction_in_the_target():
    from eqr.target import get_price_target

    r = get_price_target("AAA", ["P"], "pe", SCEN, valuation_fn=_val(flags=["price_uncertain:3.3%"]))
    assert r["conviction"] == "Low" and "price_uncertain:3.3%" in r["data_flags"]


# ------------------------------------------------------------------ foreign (IFRS, 20-F) filers and currency
def _fact(fy, val, form="20-F"):
    return {"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": val, "fy": fy, "fp": "FY", "form": form,
            "filed": f"{fy + 1}-02-01", "accn": f"0001-{fy % 100}-000001"}


def _ifrs_facts():
    return {"dei": {}, "ifrs-full": {
        "Revenue": {"units": {"EUR": [_fact(2025, 1000), _fact(2024, 900)]}},
        "ProfitLossFromOperatingActivities": {"units": {"EUR": [_fact(2025, 200), _fact(2024, 180)]}},
        "DilutedEarningsLossPerShare": {"units": {"EUR/shares": [_fact(2025, 2.0), _fact(2024, 1.5)]}},
        "DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities": {"units": {"EUR": [_fact(2025, -200), _fact(2024, -150)]}},
        "WeightedAverageShares": {"units": {"shares": [_fact(2025, 100), _fact(2024, 100)]}},
        "CashAndCashEquivalents": {"units": {"EUR": [{**_fact(2025, 300), "start": None}]}},
    }}


def test_ifrs_filer_is_read_in_its_reporting_currency_with_derived_dividend(monkeypatch):
    from eqr import edgar

    facts = _ifrs_facts()
    monkeypatch.setattr(edgar, "lookup", lambda t: (123, "Foreign Co"))
    monkeypatch.setattr(edgar, "_fetch", lambda *a, **k: {"facts": facts})
    r = edgar.get_financials("FOR", ["revenue", "operating_income", "eps_diluted", "dps"], years=5)
    assert r["taxonomy"] == "ifrs-full" and r["currency"] == "EUR"
    assert [x["fiscal_year"] for x in r["metrics"]["revenue"]] == [2025, 2024] and r["metrics"]["revenue"][0]["unit"] == "EUR"
    assert r["metrics"]["eps_diluted"][0]["unit"] == "EUR/shares"
    dps = r["metrics"]["dps"][0]                                   # 200 cash dividends / 100 weighted shares, sign ignored
    assert dps["value"] == 2.0 and dps["concept"].startswith("derived:") and dps["unit"] == "EUR/shares"
    assert any("dps is derived" in n for n in r["notes"]) and any("not USD" in n for n in r["notes"])
    # US-GAAP wins when the company reports it
    facts["us-gaap"] = {"Revenues": {"units": {"USD": [_fact(2025, 5, form="10-K")]}}}
    assert edgar.get_financials("FOR", ["revenue"])["taxonomy"] == "us-gaap"


def test_shares_fall_back_to_ifrs_weighted_average(monkeypatch):
    from eqr import edgar

    monkeypatch.setattr(edgar, "lookup", lambda t: (123, "Foreign Co"))
    monkeypatch.setattr(edgar, "_fetch", lambda *a, **k: {"facts": _ifrs_facts()})   # no dei cover-page count
    sh = edgar.get_shares_outstanding("FOR")
    assert sh["shares"] == 100.0 and sh["basis"] == "weighted_avg_diluted"


def test_fx_rate_parsing_caching_and_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from eqr import fx

    monkeypatch.setattr(fx, "CACHE_DIR", tmp_path)
    calls = []

    def ok(url, timeout):
        calls.append(url)
        return NS(status_code=200, json=lambda: {"base": "EUR", "date": "2026-11-19", "rates": {"USD": 1.0992}})
    r = fx.get_rate("eur", "usd", get=ok)
    assert r["rate"] == 1.0992 and r["source"] == "fx:ecb:EURUSD:2026-11-19" and "base=EUR&symbols=USD" in r["url"]
    fx.get_rate("EUR", "USD", get=ok)
    assert len(calls) == 1                                          # second call served from cache
    assert fx.get_rate("USD", "USD", get=ok)["rate"] == 1.0
    with pytest.raises(RuntimeError, match="HTTP 404"):
        fx.get_rate("XXX", "USD", get=lambda url, timeout: NS(status_code=404, json=lambda: {}))
    with pytest.raises(RuntimeError, match="unusable"):
        fx.get_rate("GBP", "USD", get=lambda url, timeout: NS(status_code=200, json=lambda: {"date": "2026-11-19", "rates": {"USD": 0}}))


def _eur_comps(t, today=None):
    srcs = lambda x: {"sources": [f"edgar:facts:F:{x}:FY2025:0001-25-000001"]}
    return {"company": "Foreign Co", "period_end": "2025-12-31", "stale": False, "currency": "EUR", "flags": [], "metrics": {
        "revenue": {"value": 1000.0, "unit": "EUR", **srcs("Revenue")},
        "operating_income": {"value": 200.0, "unit": "EUR", **srcs("Op")},
        "eps_diluted": {"value": 2.0, "unit": "EUR/shares", **srcs("Eps")},
        "dps": {"value": 1.0, "unit": "EUR/shares", **srcs("Dps")},
        "net_debt_proxy": {"value": 300.0, "unit": "EUR", **srcs("Debt")},
        "fcf": {"value": 100.0, "unit": "EUR", **srcs("Fcf")},
        "revenue_growth_pct": {"value": 5.0, "unit": "percent", **srcs("Rev")}}}


def _fx_ok(base, quote, on=None):
    return {"pair": "EURUSD", "rate": 1.2, "date": "2026-11-19", "source": "fx:ecb:EURUSD:2026-11-19", "url": "u"}


def _no_mcap(*a, **k):
    return {"market_cap_usd": None, "sources": [], "flags": ["mcap_not_verified"]}


def _usd_price(mcap=2400.0, usable=True):
    return {"price": 24.0, "as_of": "2026-11-19", "market_cap_reported": mcap, "sources": ["web:https://q.example/f"],
            "flags": [], "usable": usable}


def test_valuation_converts_foreign_financials_to_usd_at_spot(monkeypatch):
    from datetime import date
    from eqr import valuation as v

    monkeypatch.setattr(v.edgar, "lookup", lambda t: (1, t))
    shares = lambda t: {"shares": 100.0, "as_of": "2025-12-31", "multi_class": False, "basis": "cover_page", "source": "edgar:shares:F"}
    out = v.get_valuation(["F"], date(2026, 11, 19), price_fn=lambda t, n, d: _usd_price(), shares_fn=shares,
                          comps_fn=_eur_comps, fx_fn=_fx_ok, mcap_fn=_no_mcap)
    row = out["rows"][0]
    m = row["metrics"]
    # EPS 2.0 EUR -> 2.4 USD, so P/E = 24 / 2.4 = 10.0 (a naive 24 / 2.0 = 12.0 would be wrong by the exchange rate)
    assert m["pe"]["value"] == 10.0 and m["market_cap"]["value"] == 2400.0
    assert m["enterprise_value"]["value"] == 2760.0                              # 2400 + 300 EUR * 1.2
    assert m["ev_ebit"]["value"] == 11.5 and m["ev_sales"]["value"] == 2.3       # 2760 / 240, 2760 / 1200
    assert any(f.startswith("converted_at_spot:EURUSD=1.2000") for f in row["flags"]) and not row["withheld"]
    assert "fx:ecb:EURUSD:2026-11-19" in row["sources"] and row["reporting_currency"] == "EUR"
    assert row["fundamentals"]["eps_diluted"] == 2.4 and row["fundamentals"]["dps"] == 1.2   # fundamentals feed the target in USD


def test_foreign_valuation_is_withheld_when_it_cannot_be_verified(monkeypatch):
    from datetime import date
    from eqr import valuation as v

    monkeypatch.setattr(v.edgar, "lookup", lambda t: (1, t))
    shares = lambda t: {"shares": 100.0, "as_of": "x", "multi_class": False, "basis": "cover_page", "source": "edgar:shares:F"}

    def run(price, fx_fn=_fx_ok, mcap_fn=_no_mcap):
        return v.get_valuation(["F"], date(2026, 11, 19), price_fn=lambda t, n, d: price, shares_fn=shares,
                               comps_fn=_eur_comps, fx_fn=fx_fn, mcap_fn=mcap_fn)["rows"][0]

    no_cap = run(_usd_price(mcap=None))                                         # cannot verify the ADR-to-share ratio
    assert no_cap["withheld"] and "adr_ratio_unverified" in no_cap["flags"] and not no_cap["metrics"]
    adr2 = run(_usd_price(mcap=4800.0))                                         # page says 2x: one ADR = 0.5 shares, etc.
    assert adr2["withheld"] and any(f.startswith("market_cap_mismatch") for f in adr2["flags"])

    def boom(*a, **k):
        raise RuntimeError("no FX rate")
    nofx = run(_usd_price(), fx_fn=boom)
    assert nofx["withheld"] and any(f.startswith("fx_unavailable:EUR") for f in nofx["flags"])


def test_comps_keeps_reporting_currency_and_citations_resolve_fx(monkeypatch):
    from datetime import date
    from eqr import comps
    from eqr.citations import Refs

    fin = {"company": "Foreign Co", "currency": "EUR", "not_found": [], "metrics": {
        "revenue": [{"fiscal_year": 2025, "period_end": "2025-12-31", "value": 1000, "unit": "EUR", "source": "s:rev25"},
                    {"fiscal_year": 2024, "period_end": "2024-12-31", "value": 900, "unit": "EUR", "source": "s:rev24"}],
        "eps_diluted": [{"fiscal_year": 2025, "period_end": "2025-12-31", "value": 2.0, "unit": "EUR/shares", "source": "s:eps"}]}}
    monkeypatch.setattr(comps.edgar, "get_financials", lambda *a, **k: fin)
    row = comps._company("F", date(2026, 3, 1))
    assert row["currency"] == "EUR" and row["metrics"]["revenue"]["unit"] == "EUR" and row["metrics"]["eps_diluted"]["unit"] == "EUR/shares"
    refs = Refs({}, {}, filing_info=lambda t, a: (f"{t} {a}", None))
    refs.cite(["fx:ecb:EURUSD:2026-11-19"])
    assert refs.items[0]["label"] == "ECB reference rate EUR/USD on 2026-11-19 (via Frankfurter)"
    assert refs.items[0]["url"] == "https://api.frankfurter.dev/v1/2026-11-19?base=EUR&symbols=USD"


def test_pdf_labels_the_reporting_currency_for_foreign_filers(tmp_path):
    import json as _json
    from pypdf import PdfReader
    from eqr.pdf import build_pdf

    tr = _fake_trace()
    fin = _json.loads(tr["messages"][1]["content"][0]["content"])
    fin["currency"], fin["taxonomy"] = "EUR", "ifrs-full"
    tr["messages"][1]["content"][0]["content"] = _json.dumps(fin)
    comps_res = _json.loads(tr["messages"][3]["content"][0]["content"])
    comps_res["rows"][0]["currency"] = "EUR"
    tr["messages"][3]["content"][0]["content"] = _json.dumps(comps_res)
    note = ResearchNote(ticker="AAA", company="Alpha Co", summary="s", thesis="t", metrics=[], bull_case=[], bear_case=[], key_risks=[])
    out = build_pdf(note, tr, tmp_path / "eur.pdf", filing_info=lambda t, a: (f"{t} {a}", None))
    text = "\n".join(p.extract_text() for p in PdfReader(str(out)).pages)
    assert "€M" in text and "EUR, fiscal years" in text and "Ccy" in text
    assert "not converted here" in text and "not comparable across currencies" in text


def test_dedicated_market_cap_lets_a_foreign_row_through_and_catches_wrong_ratios(monkeypatch):
    from datetime import date
    from eqr import valuation as v

    monkeypatch.setattr(v.edgar, "lookup", lambda t: (1, t))
    shares = lambda t: {"shares": 100.0, "as_of": "x", "multi_class": False, "basis": "cover_page", "source": "edgar:shares:F"}
    ok = lambda *a: {"market_cap_usd": 2400.0, "sources": ["web:https://mc.example/f"], "flags": []}
    wrong = lambda *a: {"market_cap_usd": 12000.0, "sources": ["web:https://mc.example/f"], "flags": []}   # 5 shares per ADS

    def run(mcap_fn):
        return v.get_valuation(["F"], date(2026, 11, 19), price_fn=lambda t, n, d: _usd_price(mcap=None), shares_fn=shares,
                               comps_fn=_eur_comps, fx_fn=_fx_ok, mcap_fn=mcap_fn)["rows"][0]
    good = run(ok)
    assert not good["withheld"] and good["metrics"]["pe"]["value"] == 10.0 and "web:https://mc.example/f" in good["sources"]
    bad = run(wrong)
    assert bad["withheld"] and any(f.startswith("market_cap_mismatch") for f in bad["flags"])


def test_market_cap_extraction_is_verified_against_the_quote():
    import json as _json
    from datetime import date
    from eqr import prices

    page = "Unilever (UL) Market Cap 138.2B Revenue 51.0B"
    search = lambda **kw: {"results": [{"url": "https://f.com/ul", "title": "", "publish_date": None, "excerpts": [page]},
                                       {"url": "https://g.com/ul", "title": "", "publish_date": None, "excerpts": ["Market capitalization: $138,500,000,000"]}]}

    def ext(cands):
        return lambda m, s, u: _json.dumps({"candidates": cands})
    two = prices.get_market_cap("UL", "Unilever", date(2026, 11, 19), search=search, extract=ext([
        {"market_cap_usd": 138.2e9, "url": "https://f.com/ul", "quote": "Market Cap 138.2B"},
        {"market_cap_usd": 138.5e9, "url": "https://g.com/ul", "quote": "Market capitalization: $138,500,000,000"}]))
    assert two["market_cap_usd"] == pytest.approx(138.35e9) and len(two["sources"]) == 2 and two["flags"] == []
    # invented figure, number not in the quote, quote not on the page, and unknown url are all rejected
    for bad in ({"market_cap_usd": 200e9, "url": "https://f.com/ul", "quote": "Market Cap 138.2B"},
                {"market_cap_usd": 51.0e9, "url": "https://f.com/ul", "quote": "Unilever (UL)"},
                {"market_cap_usd": 99e9, "url": "https://f.com/ul", "quote": "Market Cap 99B"},
                {"market_cap_usd": 138.2e9, "url": "https://evil.com/x", "quote": "Market Cap 138.2B"}):
        r = prices.get_market_cap("UL", "Unilever", date(2026, 11, 19), search=search, extract=ext([bad]))
        assert r["market_cap_usd"] is None and "mcap_not_verified" in r["flags"]
    # two verified quotes that disagree by more than 5%: a real conflict, so no market cap is returned
    pages = lambda **kw: {"results": [{"url": "https://f.com/ul", "title": "", "publish_date": None, "excerpts": ["Market Cap 138.2B"]},
                                      {"url": "https://g.com/ul", "title": "", "publish_date": None, "excerpts": ["Market Cap 200.0B"]}]}
    split = prices.get_market_cap("UL", "Unilever", date(2026, 11, 19), search=pages, extract=ext([
        {"market_cap_usd": 138.2e9, "url": "https://f.com/ul", "quote": "Market Cap 138.2B"},
        {"market_cap_usd": 200.0e9, "url": "https://g.com/ul", "quote": "Market Cap 200.0B"}]))
    assert split["market_cap_usd"] is None and "mcap_conflict" in split["flags"]
