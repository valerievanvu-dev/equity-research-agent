"""Render the research note as a PDF with full data tables and numbered, linked citations.

Tables come from the saved tool outputs (see citations.artifacts_from_trace), not from the LLM's copy of them.
Prose (thesis, bull/bear, risks) is the model's, with its source ids resolved to numbered references.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .citations import ID_RE, Artifacts, Refs, artifacts_from_trace
from .schemas import ResearchNote

NAVY, GREY, LIGHT = colors.HexColor("#1F3A5F"), colors.HexColor("#555555"), colors.HexColor("#F3F6FA")
_REPL = {"→": "->", "≥": ">=", "≤": "<=", "≈": "~", "−": "-", "‑": "-", " ": " ", "•": "-", "×": "x"}

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=12.5, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=NAVY,
                    spaceBefore=12, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BASE, fontSize=7.5, leading=10, textColor=GREY)
BULLET = ParagraphStyle("bullet", parent=BASE, leftIndent=12, bulletIndent=2)
CELL_L = ParagraphStyle("cl", parent=BASE, fontSize=7, leading=8.5, spaceAfter=0)
CELL_R = ParagraphStyle("cr", parent=CELL_L, alignment=TA_RIGHT)
CELL_S = ParagraphStyle("cs", parent=CELL_L, fontSize=6, leading=7)
HDR_L = ParagraphStyle("hl", parent=CELL_L, fontName="Helvetica-Bold", textColor=colors.white)
HDR_R = ParagraphStyle("hr", parent=HDR_L, alignment=TA_RIGHT)


def safe(s: object) -> str:
    """Escape for reportlab markup and drop characters the built-in Helvetica font cannot draw."""
    s = str(s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    return escape(s.encode("cp1252", "replace").decode("cp1252"))


def rich(text: str, refs: Refs) -> str:
    """Escape prose and replace any source ids the model wrote inline with [n] reference markers."""
    parts, ids = ID_RE.split(text), ID_RE.findall(text)
    out = []
    for i, part in enumerate(parts):
        out.append(safe(part))
        if i < len(ids):
            raw = ids[i].rstrip(".")
            out.append(marker(refs.cite([raw])) + safe(ids[i][len(raw):]))
    return "".join(out)


def marker(nums: list[int]) -> str:
    return f'<font size="7" color="#1F3A5F">[{",".join(map(str, nums))}]</font>' if nums else ""


CCY_SYMBOL = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5"}


def sym(ccy: str) -> str:
    return CCY_SYMBOL.get(ccy, f"{ccy} ")


# ----------------------------------------------------------------------------- number formats
def fm(v):  # USD -> $ millions
    return "n/a" if v is None else f"{v / 1e6:,.1f}"


def fp(v):
    return "n/a" if v is None else f"{v:.1f}%"


def fx(v):
    return "n/a" if v is None else f"{v:.1f}x"


def fe(v):
    return "n/a" if v is None else f"{v:,.2f}"


def mv(row: dict, key: str):
    return row["metrics"].get(key, {}).get("value")


# ----------------------------------------------------------------------------- tables
def make_table(header: list[str], rows: list[list[str]], widths: list[float], bold_rows: set[int] = frozenset(),
               small_cols: set[int] = frozenset()) -> Table:
    data = [[Paragraph(h, HDR_L if i == 0 else HDR_R) for i, h in enumerate(header)]]
    for ri, r in enumerate(rows):
        row = []
        for ci, c in enumerate(r):
            style = CELL_S if ci in small_cols else (CELL_L if ci == 0 else CELL_R)
            text = f"<b>{c}</b>" if ri in bold_rows else c
            row.append(Paragraph(text, style))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DBE3")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


FIN_ROWS = [("revenue", "Revenue ($M)", fm), ("gross_profit", "Gross profit ($M)", fm),
            ("operating_income", "Operating income ($M)", fm), ("net_income", "Net income ($M)", fm),
            ("eps_diluted", "Diluted EPS ($)", fe), ("dps", "Dividends per share ($)", fe), ("operating_cash_flow", "Operating cash flow ($M)", fm),
            ("capex", "Capex ($M)", fm), ("cash", "Cash ($M)", fm), ("long_term_debt", "Long-term debt ($M)", fm),
            ("total_assets", "Total assets ($M)", fm), ("total_liabilities", "Total liabilities ($M)", fm),
            ("shareholders_equity", "Shareholders' equity ($M)", fm)]


def financial_table(a: Artifacts, refs: Refs, years: int = 5) -> list:
    fys = sorted({fy for rows in a.financials.values() for fy in rows})[-years:]
    if not fys:
        return [Paragraph("Financial data was not retrieved in this run.", SMALL)]
    val = lambda m, fy: a.financials.get(m, {}).get(fy, {}).get("value")  # noqa: E731

    def end(fy):
        for rows in a.financials.values():
            if fy in rows:
                return rows[fy]["period_end"]
        return ""

    header = [f"{a.currency}, fiscal years"]
    for fy in fys:
        srcs = [r[fy]["source"] for r in a.financials.values() if fy in r]
        header.append(f"FY{fy}<br/>{safe(end(fy)[:7])} {marker(refs.cite(srcs))}")
    rows: list[list[str]] = []
    for key, label, fmt in FIN_ROWS:
        if key in a.financials:
            rows.append([safe(label.replace("$", sym(a.currency)))] + [fmt(val(key, fy)) for fy in fys])
    derived = []

    def ratio(num, den, fy):
        n, d = val(num, fy), val(den, fy)
        return None if n is None or not d else n / d * 100

    derived.append(["Revenue growth, YoY"] + [fp(None if val("revenue", fy - 1) in (None, 0) or val("revenue", fy) is None
                                                  else (val("revenue", fy) / val("revenue", fy - 1) - 1) * 100)
                                              for fy in fys])
    derived.append(["Gross margin"] + [fp(ratio("gross_profit", "revenue", fy)) for fy in fys])
    derived.append(["Operating margin"] + [fp(ratio("operating_income", "revenue", fy)) for fy in fys])
    derived.append(["Net margin"] + [fp(ratio("net_income", "revenue", fy)) for fy in fys])
    fcf = {fy: (None if val("operating_cash_flow", fy) is None or val("capex", fy) is None
                else val("operating_cash_flow", fy) - val("capex", fy)) for fy in fys}
    derived.append([f"Free cash flow ({sym(a.currency)}M)"] + [fm(fcf[fy]) for fy in fys])
    derived.append(["FCF margin"] + [fp(None if fcf[fy] is None or not val("revenue", fy)
                                          else fcf[fy] / val("revenue", fy) * 100) for fy in fys])
    derived = [[safe(d[0])] + d[1:] for d in derived]
    first = 160
    widths = [first] + [(525 - first) / len(fys)] * len(fys)
    n_raw = len(rows)
    t = make_table(header, rows + derived, widths, bold_rows=set(range(n_raw, n_raw + len(derived))) - set())
    foreign = a.currency != "USD" or a.taxonomy != "us-gaap"
    note = Paragraph("Bold rows are derived in code from the reported figures above. Source: SEC XBRL company facts "
                     "from each fiscal year's annual report; [n] points to the filing."
                     + (f" Figures are in {a.currency} as reported (IFRS, foreign filer) and are not converted here; "
                        "the valuation tables convert to USD at the ECB spot rate." if foreign else ""), SMALL)
    return [t, Spacer(1, 3), note]


def comps_table(a: Artifacts, refs: Refs, subject: str) -> list:
    if not a.comps:
        return [Paragraph("Peer comps were not retrieved in this run.", SMALL)]
    header = ["Ticker", "Period end", "Ccy", "Revenue (M)", "Growth", "Gross", "Operating", "Net", "FCF", "Net debt (M)", "Refs"]
    rows, bold = [], set()
    for i, r in enumerate(a.comps):
        if r["ticker"] == subject:
            bold.add(i)
        flag = " (stale)" if r.get("stale") else ""
        rows.append([safe(r["ticker"]), safe(r["period_end"]) + safe(flag), safe(r.get("currency", "USD")),
                     fm(mv(r, "revenue")), fp(mv(r, "revenue_growth_pct")), fp(mv(r, "gross_margin_pct")),
                     fp(mv(r, "operating_margin_pct")), fp(mv(r, "net_margin_pct")), fp(mv(r, "fcf_margin_pct")),
                     fm(mv(r, "net_debt_proxy")), marker(refs.cite([r["source"]]))])
    if a.comps_calls == 1 and a.comps_medians:
        md = lambda k: (a.comps_medians.get(k) or {}).get("value")  # noqa: E731
        rows.append(["Median", "", "", "", fp(md("revenue_growth_pct")), fp(md("gross_margin_pct")),
                     fp(md("operating_margin_pct")), fp(md("net_margin_pct")), fp(md("fcf_margin_pct")), "", ""])
        bold.add(len(rows) - 1)
    widths = [40, 62, 30, 58, 42, 42, 48, 36, 36, 58, 73]
    mixed = len({r.get("currency", "USD") for r in a.comps}) > 1 or any(r.get("currency", "USD") != "USD" for r in a.comps)
    return [make_table(header, rows, widths, bold_rows=bold), Spacer(1, 3),
            Paragraph("Percentages are of revenue and are currency-neutral. Revenue and net debt are in each company's "
                      "reporting currency (Ccy), in millions" + ("; they are not converted and not comparable across currencies"
                      if mixed else "") + ". Net debt = long-term debt less cash (see notes). Fiscal year-ends differ; "
                      "compare by period end. n/a = not reported in a standard XBRL tag, never zero. Subject company in bold.",
                      SMALL)]


def valuation_table(a: Artifacts, refs: Refs, subject: str) -> list:
    if not a.valuation:
        return [Paragraph("Valuation multiples were not retrieved in this run.", SMALL)]
    header = ["Ticker", "Price ($)", "Price date", "Mkt cap ($M)", "EV ($M)", "EV/Sales", "EV/EBIT", "P/E",
              "FCF yield", "Flags", "Refs"]
    rows, bold = [], set()
    for i, r in enumerate(a.valuation):
        if r["ticker"] == subject:
            bold.add(i)
        flags = ", ".join(r.get("flags", [])) + (" WITHHELD" if r.get("withheld") else "")
        rows.append([safe(r["ticker"]), fe(r.get("price")), safe(r.get("price_date") or "n/a"), fm(mv(r, "market_cap")),
                     fm(mv(r, "enterprise_value")), fx(mv(r, "ev_sales")), fx(mv(r, "ev_ebit")), fx(mv(r, "pe")),
                     fp(mv(r, "fcf_yield_pct")), safe(flags) or "-", marker(refs.cite([r["source"]]))])
    if a.valuation_calls == 1 and a.valuation_medians:
        md = lambda k: (a.valuation_medians.get(k) or {}).get("value")  # noqa: E731
        rows.append(["Median", "", "", "", "", fx(md("ev_sales")), fx(md("ev_ebit")), fx(md("pe")),
                     fp(md("fcf_yield_pct")), "", ""])
        bold.add(len(rows) - 1)
    widths = [36, 40, 48, 52, 52, 36, 38, 32, 38, 106, 47]
    return [make_table(header, rows, widths, bold_rows=bold, small_cols={9}), Spacer(1, 3),
            Paragraph("Multiples use the latest fiscal year, not last-twelve-months. Prices come from web pages "
                      "(see references) and are verified in code, but treat flagged rows with caution.", SMALL)]


METHOD_LABEL = {"pe": "P/E", "ev_ebit": "EV/EBIT", "ev_sales": "EV/SALES"}
RATING_COLORS = {"Buy": colors.HexColor("#1B7F3B"), "Hold": colors.HexColor("#B7791F"), "Sell": colors.HexColor("#B42318")}


def _rec_data(a: Artifacts, note: ResearchNote) -> dict | None:
    """Prefer the tool's computed result; fall back to the model's copy of it."""
    if a.price_target:
        return a.price_target
    return note.recommendation.model_dump() if note.recommendation else None


def recommendation_box(rec: dict, refs: Refs) -> Table:
    big = ParagraphStyle("rating", parent=BASE, fontName="Helvetica-Bold", fontSize=22, leading=26,
                         textColor=colors.white, alignment=1, spaceAfter=0)
    lab = ParagraphStyle("lab", parent=SMALL, fontSize=7, leading=9, spaceAfter=0)
    val = ParagraphStyle("val", parent=BASE, fontName="Helvetica-Bold", fontSize=12, leading=14, spaceAfter=0)
    horizon = rec.get("horizon_months", 12)
    pr, dy = rec.get("expected_price_return_pct"), rec.get("dividend_yield_pct")
    split = f" (price {pr:+.1f}% + dividend {dy:.1f}%)" if pr is not None and dy is not None else ""
    cells = [
        [Paragraph(safe(rec["rating"]).upper(), big),
         [Paragraph(f"{horizon}-MONTH PRICE TARGET", lab), Paragraph(f"${rec['price_target']:,.2f}", val)],
         [Paragraph("CURRENT PRICE", lab), Paragraph(f"${rec['current_price']:,.2f}", val),
          Paragraph(safe(rec.get("price_date") or ""), lab)],
         [Paragraph("EXPECTED TOTAL RETURN", lab), Paragraph(f"{rec['expected_total_return_pct']:+.1f}%", val),
          Paragraph(safe(split), lab)],
         [Paragraph("METHOD / CONVICTION", lab), Paragraph(f"{METHOD_LABEL.get(rec['method'], safe(rec['method']).upper())} / {safe(rec.get('conviction', ''))}", val)]]]
    t = Table(cells, colWidths=[84, 105, 100, 135, 101])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), RATING_COLORS.get(rec["rating"], NAVY)),
                           ("BACKGROUND", (1, 0), (-1, 0), LIGHT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("BOX", (0, 0), (-1, -1), 0.6, NAVY), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def target_section(a: Artifacts, note: ResearchNote, refs: Refs) -> list:
    rec = _rec_data(a, note)
    if not rec:
        return []
    out: list = [Paragraph("Price target and rating derivation", H2),
                 Paragraph(f"<b>Rating rule (fixed in code):</b> {safe(rec.get('rating_rule', 'see methodology'))}", BASE)]
    unit = {"pe": "EPS growth", "ev_ebit": "EBIT growth", "ev_sales": "Revenue growth"}.get(rec["method"], "Growth")
    mult = {"pe": "Target P/E", "ev_ebit": "Target EV/EBIT", "ev_sales": "Target EV/Sales"}.get(rec["method"], "Target multiple")
    rows = [[safe(s["name"]).title(), f"{s['weight'] * 100:.0f}%", f"{s['growth_pct']:+.1f}%", f"{s['multiple']:.1f}x",
             f"${s['price']:,.2f}", f"{(s['price'] / rec['current_price'] - 1) * 100:+.1f}%"] for s in rec["scenarios"]]
    rows.append(["Probability-weighted target", "100%", "", "", f"${rec['price_target']:,.2f}",
                 f"{rec.get('expected_price_return_pct', 0):+.1f}%"])
    out.append(make_table(["Scenario", "Probability", f"{unit} (12m)", mult, "Implied price", "vs current"], rows,
                          [150, 60, 85, 80, 80, 70], bold_rows={len(rows) - 1}))
    anchors = rec.get("anchors") or {}
    if anchors:
        cur, med = anchors.get("current_multiple"), anchors.get("peer_median_multiple")
        out.append(Paragraph(f"Anchors: current {safe(rec['method'])} multiple {fx(cur)}; peer median {fx(med)} "
                             f"(n={anchors.get('peers_in_median', 0)}). Dividend yield uses the last declared dividend "
                             f"per share.", SMALL))
    out.append(Paragraph("Scenario assumptions (the model's judgement, with its evidence)", H2))
    for s in rec["scenarios"]:
        out.append(Paragraph(f"<b>{safe(s['name']).title()}:</b> {rich(s.get('rationale', ''), refs)}", BULLET, bulletText="-"))
    sens = rec.get("sensitivity")
    if sens:
        hdr = [f"{unit} \\ {mult}"] + [f"{m:.1f}x" for m in sens["multiples"]]
        grid = [[f"{r['growth_pct']:+.1f}%"] + [("n/a" if p is None else f"${p:,.2f}") for p in r["prices"]]
                for r in sens["rows"]]
        out += [Paragraph("Sensitivity: implied price, base-case assumptions varied", H2),
                make_table(hdr, grid, [105] + [84] * 5, bold_rows={2})]
    if note.recommendation and note.recommendation.rationale:
        out += [Paragraph("Why this rating", H2), Paragraph(rich(note.recommendation.rationale, refs) + " "
                + marker(refs.cite(note.recommendation.sources)), BASE)]
    flags = (rec.get("flags") or []) + (rec.get("data_flags") or [])
    if flags:
        out.append(Paragraph("Flags: " + safe(", ".join(flags)), SMALL))
    for n in rec.get("notes", []):
        out.append(Paragraph(safe(n), BULLET, bulletText="-"))
    return out


# ----------------------------------------------------------------------------- document
def build_pdf(note: ResearchNote, trace: dict, path: str | Path, today: date | None = None,
              filing_info=None) -> Path:
    today = today or date.today()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = artifacts_from_trace(trace)
    refs = Refs(a.inputs_of, a.titles, **({"filing_info": filing_info} if filing_info else {}))
    story: list = []

    story += [Paragraph(f"{safe(note.company)} ({safe(note.ticker)})", H1),
              Paragraph(f"Equity research note &middot; generated {today.isoformat()} &middot; model "
                        f"{safe(trace.get('model', 'n/a'))}", SMALL), Spacer(1, 4)]
    box = Table([[Paragraph("<b>Automated research output. Not investment advice.</b> Generated by an AI agent from "
                            "SEC filings and web sources. Figures were not independently audited; verify against the "
                            "cited filings before relying on them.", SMALL)]], colWidths=[525])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6E0")),
                             ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0B84C")),
                             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    rec = _rec_data(a, note)
    if rec:
        story += [box, Spacer(1, 8), recommendation_box(rec, refs),
                  Paragraph("Rating and target are the output of a fixed rule applied to model-proposed scenarios "
                            "(see derivation). Illustration only, not a recommendation to buy or sell any security. "
                            + marker(refs.cite(rec.get("sources", []))), SMALL)]
    else:
        story += [box]
    story += [Paragraph("Summary", H2), Paragraph(rich(note.summary, refs), BASE),
              Paragraph("Thesis", H2), Paragraph(rich(note.thesis, refs), BASE)]

    story += [Paragraph("Financial summary", H2)] + financial_table(a, refs)
    story += [Paragraph("Peer comparison: operating", H2)] + comps_table(a, refs, note.ticker)
    if note.peer_commentary:
        story += [Spacer(1, 4), Paragraph(rich(note.peer_commentary, refs), BASE)]
    story += [Paragraph("Peer comparison: valuation", H2)] + valuation_table(a, refs, note.ticker)
    story += target_section(a, note, refs)

    for title, claims in (("Bull case", note.bull_case), ("Bear case", note.bear_case), ("Key risks", note.key_risks)):
        block = [Paragraph(title, H2)]
        for c in claims:
            block.append(Paragraph(rich(c.text, refs) + " " + marker(refs.cite(c.sources)), BULLET, bulletText="-"))
        story += block
    if note.open_questions:
        story.append(Paragraph("Open questions and data gaps", H2))
        story += [Paragraph(rich(q, refs), BULLET, bulletText="-") for q in note.open_questions]
    if a.notes:
        story.append(Paragraph("Methodology and data limitations", H2))
        story += [Paragraph(safe(n), BULLET, bulletText="-") for n in a.notes]

    ref_block = [Paragraph("References", H2)]
    for i, item in enumerate(refs.items, 1):
        link = (f' <a href="{escape(item["url"], {chr(34): "&quot;"})}" color="#1F3A5F">{safe(item["url"])}</a>'
                if item["url"] else "")
        ref_block.append(Paragraph(f"[{i}] {safe(item['label'])}.{link}", SMALL))
    story += ref_block

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(0.6 * inch, 0.4 * inch, "Automated research output. Not investment advice. Verify against cited filings.")
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                            title=f"{note.company} research note", author="Equity research agent")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path
