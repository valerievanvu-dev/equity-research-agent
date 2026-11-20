from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .agent import run
from .schemas import ResearchNote

DEFAULT_TASK = (
    "Write a research note on {ticker}. Pull the last 5 fiscal years of financials (at minimum revenue, "
    "gross_profit, operating_income, net_income, operating_cash_flow, capex), read the latest 10-K's "
    "risk_factors and mdna, search the web for recent news and catalysts, build a peer comps table (peer_universe, get_comps) and valuation multiples (get_valuation), compute margins and revenue growth with the "
    "calculator, then submit the note."
)


def render(n: ResearchNote) -> str:
    def fmt(v: float, unit: str) -> str:
        if unit == "percent":
            return f"{v:.1f}%"
        if unit == "USD" and abs(v) >= 1e6:
            return f"${v / 1e9:,.2f}B"
        return f"{v:,.2f} {unit}"

    lines = [f"# {n.company} ({n.ticker})", "", n.summary, "", "## Thesis", n.thesis, "", "## Metrics"]
    lines += [f"- {m.name} {m.period}: {fmt(m.value, m.unit)}  `{m.sources[0]}`" for m in n.metrics]
    if n.comps:
        cols = sorted({k for c in n.comps for k in c.metrics})
        lines += ["", "## Peer comps (operating)", "| ticker | period end | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 2)]
        lines += [f"| {c.ticker} | {c.period_end} | " + " | ".join(
            f"{c.metrics[k]:,.1f}" if k in c.metrics else "n/a" for k in cols) + " |" for c in n.comps]
        if n.peer_commentary:
            lines += ["", n.peer_commentary]
    if n.valuation:
        cols = sorted({k for v in n.valuation for k in v.metrics})
        lines += ["", "## Valuation (latest fiscal year, not LTM)", "| ticker | price date | " + " | ".join(cols) + " | flags |",
                  "|" + "---|" * (len(cols) + 3)]
        lines += [f"| {v.ticker} | {v.price_date or 'n/a'} | " + " | ".join(
            f"{v.metrics[k]:,.1f}" if k in v.metrics else "n/a" for k in cols) + f" | {', '.join(v.flags) or '-'} |"
            for v in n.valuation]
    for title, claims in (("Bull case", n.bull_case), ("Bear case", n.bear_case), ("Key risks", n.key_risks)):
        lines += ["", f"## {title}"] + [f"- {c.text}  `{'; '.join(c.sources)}`" for c in claims]
    if n.open_questions:
        lines += ["", "## Open questions"] + [f"- {q}" for q in n.open_questions]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the equity research agent on a ticker")
    ap.add_argument("ticker")
    ap.add_argument("--task", help="Override the default task prompt ({ticker} is substituted)")
    ap.add_argument("--model")
    ap.add_argument("--max-turns", type=int, default=15)
    args = ap.parse_args()

    res = run((args.task or DEFAULT_TASK).format(ticker=args.ticker.upper()), ticker=args.ticker.upper(),
              model=args.model, max_turns=args.max_turns)
    if res.note:
        md = render(res.note)
        print(md)
        report = Path("reports") / f"{args.ticker.upper()}-{date.today().isoformat()}.md"
        report.parent.mkdir(exist_ok=True)
        report.write_text(md + "\n")
        print(f"\nreport saved: {report}")
    else:
        print(f"FAILED: {res.error}")
    t = res.trace
    print(f"\n[{t['turns']} turns, {t['usage']['input_tokens']} in / {t['usage']['output_tokens']} out tokens, "
          f"{t['wall_s']}s] trace: {res.trace_path}")
