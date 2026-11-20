"""Run the eval suite:  uv run python evals/run.py [--repeat 3] [--workers 4] [--model ...]

--repeat matters: agent runs are stochastic, so one pass/fail is noise. Look at pass rate over k runs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eqr.agent import run
from eqr.graders import grade
from eqr.judge import judge

ROOT = Path(__file__).parent


def run_case(case: dict, model: str | None, rep: int, judge_model: str | None) -> dict:
    res = run(case["task"], ticker=case["ticker"], model=model, meta={"case_id": case["id"], "rep": rep})
    g = grade(case, res.note, res.trace)
    jr = None
    if judge_model and res.note:
        try:
            jr = judge(res.note, res.trace, model=judge_model)
            g["judge"] = {**jr.model_dump(), "overall": jr.overall, "judge_model": judge_model}
        except Exception as e:  # a judge failure must not sink the run
            g["judge"] = {"error": f"{type(e).__name__}: {e}"}
    # persist the grade next to the trace so export_sft.py can filter on it
    res.trace["grade"] = g
    if res.trace_path:
        res.trace_path.write_text(json.dumps(res.trace, indent=2))
    return {
        "case_id": case["id"], "rep": rep, **{k: g[k] for k in ("schema_ok", "metric_accuracy", "citation_grounded", "reward", "passed")},
        "details": g["details"], "turns": res.trace["turns"], "tokens": sum(res.trace["usage"].values()),
        "error": res.error, "trace": str(res.trace_path),
        "judge": g.get("judge"), "judge_overall": jr.overall if jr else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "cases.jsonl"))
    ap.add_argument("--model")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", help="substring filter on case id")
    ap.add_argument("--judge", nargs="?", const="default", metavar="MODEL",
                    help="also run the LLM judge (default model: EQR_JUDGE_MODEL). Reported separately from pass/fail.")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    from eqr.judge import DEFAULT_JUDGE_MODEL
    judge_model = None if not args.judge else (DEFAULT_JUDGE_MODEL if args.judge == "default" else args.judge)
    jobs = [(c, r) for c in cases for r in range(args.repeat)]

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda j: run_case(j[0], args.model, j[1], judge_model), jobs))

    print(f"\n{'case':34} {'rep':>3} {'acc':>5} {'cite':>5} {'pass':>5} {'turns':>5} {'tokens':>7} {'judge':>6}")
    for r in rows:
        print(f"{r['case_id']:34} {r['rep']:>3} {r['metric_accuracy']:>5.2f} {r['citation_grounded']:>5.2f} "
              f"{'PASS' if r['passed'] else 'FAIL':>5} {r['turns']:>5} {r['tokens']:>7} "
              f"{'-' if r['judge_overall'] is None else format(r['judge_overall'], '.2f'):>6}")
        for d in r["details"]:
            if not d["ok"]:
                print(f"    x {d['name']} {d['period']}: expected {d['expected']}, got {d['got']}")
        if r["error"]:
            print(f"    ! {r['error']}")
    summary = {
        "pass_rate": sum(r["passed"] for r in rows) / len(rows),
        "mean_reward": st.mean(r["reward"] for r in rows),
        "mean_turns": st.mean(r["turns"] for r in rows),
        "mean_tokens": st.mean(r["tokens"] for r in rows),
        "n": len(rows),
    }
    judged = [r["judge_overall"] for r in rows if r["judge_overall"] is not None]
    if judged:
        summary["mean_judge_overall"] = st.mean(judged)
    print("\n" + json.dumps(summary, indent=2))
    out = ROOT / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "judge_model": judge_model, "summary": summary, "rows": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
