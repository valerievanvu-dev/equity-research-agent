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
         med_pe=11.0, med_n=3, flags=None):
    row = {"ticker": "AAA", "price": price, "price_date": "2026-11-19", "withheld": withheld, "flags": flags or [],
           "source": "calc:valuation:AAA:2026-11-19", "sources": ["web:https://q.example/a", "edgar:facts:AAA:EPS:FY2025:0001-26-000001"],
           "metrics": {"pe": {"value": pe}, "ev_ebit": {"value": ev_ebit}},
           "fundamentals": {"eps_diluted": eps, "operating_income": oi, "net_debt_proxy": nd, "dps": dps, "shares": shares}}
    medians = {"pe": {"value": med_pe, "n": med_n}, "ev_ebit": {"value": 9.5, "n": med_n}}
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
