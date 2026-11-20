# Equity research agent (learning project)

A tool-use agent that writes a sourced research note from SEC filings, built so you can practice
**agent engineering, evals, and post-training** on one small codebase.

```
src/eqr/
  edgar.py     data layer: XBRL facts, filing list, 10-K text + section splitter (disk-cached)
  tools.py     tool schemas + dispatcher + safe calculator
  schemas.py   ResearchNote: the output contract (structured, cited)
  agent.py     the loop (~100 lines, no framework) + full trace logging
  graders.py   deterministic graders; also usable as RL / rejection-sampling rewards
  comps.py     peer universes (beauty) + operating comps table computed in code from XBRL
  cli.py       `uv run eqr ELF`
evals/         cases.jsonl + run.py (parallel, --repeat k)
scripts/       export_sft.py (traces -> SFT), show_comps.py (comps table, no LLM), check_universe.py, find_concepts.py (debug XBRL tags)
tests/         offline unit tests (no key, no network)
```

## Sample output

[examples/ELF-2026-11-19.md](examples/ELF-2026-11-19.md) is a real run on e.l.f. Beauty: financials, peer comps, valuation with
flags, bull/bear cases, and open questions. It is a demo of the format, **not investment advice**, and it was not audited.
Every run saves its note to `reports/` (git-ignored) and its full trace to `runs/`.

## PDF report and email

Every run also writes `reports/<TICKER>-<date>.pdf` and, if email is configured, sends it to you. The PDF has the
thesis, a 5-year financial table (all metrics, with margins and growth derived in code), every peer in the comps
and valuation tables, bull/bear/risks, and **numbered references that link to the SEC filing or web page**.

- The tables are rendered from the saved tool outputs, not from numbers the model re-typed, so a transcription
  slip by the model cannot reach the PDF.
- Re-render a PDF from any saved trace for free: `uv run python scripts/pdf_from_trace.py runs/<run>.json`.
- Email uses SMTP, configured only through environment variables (nothing personal lives in the repo):

```bash
export EQR_EMAIL_TO="you@example.com"
export EQR_SMTP_USER="you@example.com"        # SMTP login
export EQR_SMTP_PASSWORD="app-password"       # for Gmail: an App Password, not your account password
# optional: EQR_SMTP_HOST (default smtp.gmail.com), EQR_SMTP_PORT (default 587), EQR_EMAIL_FROM
```

**Easiest: Resend** (HTTPS API, no Mail app, no SMTP login). Create a free account at resend.com, make an API key, then:

```bash
export EQR_EMAIL_METHOD="resend"
export RESEND_API_KEY="re_..."
export EQR_EMAIL_TO="the-email-you-signed-up-to-resend-with@example.com"
```

Resend's default test sender (`onboarding@resend.dev`) can only deliver to the address your Resend account was
created with. To send to other recipients, verify a domain at resend.com/domains and set `EQR_EMAIL_FROM` to an
address on it. Keep the key in your shell profile, never in the repo.

**If your provider blocks App Passwords** (common on managed Google Workspace accounts), send through the Mail app
on your Mac instead. No password is stored anywhere; the first run asks macOS to let Terminal control Mail:

```bash
export EQR_EMAIL_TO="you@example.com"
export EQR_EMAIL_METHOD="macos-mail"
# optional: EQR_EMAIL_FROM="address of the Mail account to send from"
```

If email is not configured the run still succeeds and prints `email skipped`. Use `--no-email` to build the PDF without sending.

## Setup

```bash
cp .env.example .env     # fill in OPENAI_API_KEY, PARALLEL_API_KEY, SEC_USER_AGENT, then:
set -a; source .env; set +a
uv run pytest            # offline sanity check
uv run eqr AAPL          # one run; trace saved to runs/
uv run python evals/run.py --repeat 3
```

`SEC_USER_AGENT` must be "Your Name you@example.com": the SEC blocks anonymous scrapers.

## How it works

1. The model gets 9 tools: `get_financials`, `list_filings`, `read_filing_section`, `peer_universe` + `get_comps`
   (operating comps computed in code, see `comps.py`), `get_valuation` (multiples, see below), `web_search` (Parallel, needs
   `PARALLEL_API_KEY`), `calculator`, `submit_note`. Web results are flagged untrusted and cached on disk; reported
   figures must still come from filings. Good eval cases to add: a web page that tries to inject instructions, and
   a web figure that contradicts the filing.
2. Every tool result carries a `source` id. The note must cite those exact ids.
3. `submit_note` validates against the pydantic schema. If invalid, the error goes back to the model to self-repair.
4. Every run saves a full trace (messages, tool calls, tokens, latency) to `runs/`.
5. Graders check metric accuracy against golden values, and that every citation really appeared in a tool result.

## Comparing models and judging quality

```bash
uv run python evals/run.py --repeat 3 --judge                                    # defaults: gpt-6.2-astra agent, gpt-5.5 judge
uv run python evals/run.py --repeat 3 --judge claude-sonnet-5-5                  # cross-family judge (needs ANTHROPIC_API_KEY)
```

- `--model` picks the provider by name (`claude*` -> Anthropic, anything else -> OpenAI). Traces use one canonical
  format either way (`llm.py` translates), so graders and `export_sft.py` work on both.
- Everything defaults to OpenAI (`gpt-6.2-astra` agent, `gpt-5.5` judge), so the judge is the same vendor as the agent. Add an
  Anthropic key and use `--judge claude-sonnet-5-5` for a cross-family judge.
- **Use a judge from a different family than the agent** to avoid self-preference bias.
- Judge scores are reported *separately* from `reward`/`passed`. Deterministic graders and the judge catch different
  failures: a citation can point to a real source id (grader passes) that doesn't support the claim (judge flags it).
- Before trusting the judge: hand-label ~20 notes, measure agreement, and check it isn't just rewarding length.

## Comps: what is and isn't covered

- `get_comps`: operating comps (growth, gross/operating/net/FCF margin, net-debt proxy), all from SEC filings.
- `get_valuation`: EV/Sales, EV/EBIT, P/E, FCF yield. Price = Parallel web search + a small extraction model
  (`EQR_EXTRACT_MODEL`, default gpt-5.4-mini), shares = SEC cover page. Multiples use the latest fiscal year, **not LTM**.
- Price guards (`prices.py`, each is an eval opportunity): verbatim-quote check, majority cluster of same-day quotes,
  staleness limit, market-cap cross-check vs the page. Failing rows are `withheld`, never silently used.
- Debug tools: `scripts/show_comps.py`, `show_valuation.py`, `debug_price.py TICKER`, `debug_netdebt.py`, `find_concepts.py`.
- US SEC filers only: L'Oreal, Shiseido, Beiersdorf, Kao, Amorepacific are not in the universe.
- XBRL tag gaps exist (e.g. Estee Lauder net income, Coty capex): missing values are reported as missing, never zero.
- Fiscal years are not calendar aligned; compare by `period_end`.

## Caveats you should know

- **Goldens are from public 10-K figures I recalled; verify them against the filings** before trusting a fail.
  A failing case can be a bad golden, a bad tool, or a bad agent. Diagnosing which is the skill.
- The fiscal-year picker (`select_annual`) and the section splitter are heuristics. They are covered by unit tests but will break on odd filers.
- Cached data never refreshes. Delete `.cache/` to re-pull.
- v1 scores *facts and grounding*, not whether the thesis is any good.

## Learning path

**Agent engineering**
1. Run one ticker, then read `runs/*.json` end to end. Where did turns go? Which tool calls were wasted?
2. Break things on purpose: remove the calculator, shorten `max_turns`, drop the "cite exactly" rule. Watch which graders move.
3. Add a tool: `search_filing(accession, query)` (keyword snippets) so the agent stops paging whole sections.
4. Add a `peers` tool + comps table, then a subagent per section.

**Evals**
1. Grow `cases.jsonl` to ~30: more tickers, restated years, 52/53-week filers, a ticker with no data (correct behaviour = say so).
2. Add an **LLM judge** for thesis quality with a rubric. Then grade the judge: hand-label 20 notes and measure agreement.
3. Run with `--repeat 5` and report pass@1 vs pass@5; learn how noisy "one run" is.
4. Add negative cases: fabricated-number traps, prompt injection inside a filing's text.

**Post-training**
Claude models are not something you fine-tune directly, so the practice loop is distillation into an open model:
1. Generate many traces with Claude (`--repeat`, many tickers) and grade them.
2. `scripts/export_sft.py` keeps only traces that pass (rejection sampling).
3. SFT a small open model (e.g. Qwen or Llama class, with TRL or Unsloth) on those tool-use traces.
4. Evaluate it on the **same** `evals/` harness (write a small adapter for its API) and compare to the teacher.
5. Stretch: use `grade()["reward"]` as the reward for GRPO-style RL. Hold out tickers so you measure generalization, not memorization.

Keep a held-out eval set that never enters training data.

## License

MIT, see [LICENSE](LICENSE). Research output is generated by AI from public sources and comes with no warranty;
it is not investment advice.
