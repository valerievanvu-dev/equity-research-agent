"""Turn graded traces into an SFT dataset (rejection sampling / distillation).

    uv run python scripts/export_sft.py --min-reward 0.99 --out data/sft.jsonl

Only traces that eval/run.py graded (it writes `grade` into the trace) and that clear the
reward threshold are kept. Output is one JSON object per line: system, tools, messages, reward.
Convert to your trainer's chat/tool-call format (e.g. TRL) in the next step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eqr.tools import TOOLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--min-reward", type=float, default=0.99)
    ap.add_argument("--out", default="data/sft.jsonl")
    args = ap.parse_args()

    kept = seen = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in sorted(Path(args.runs).glob("*.json")):
            t = json.loads(p.read_text())
            seen += 1
            g = t.get("grade")
            if not g or not g["passed"] or g["reward"] < args.min_reward:
                continue
            f.write(json.dumps({"system": t["system"], "tools": TOOLS, "messages": t["messages"],
                                "reward": g["reward"], "run_id": t["run_id"]}) + "\n")
            kept += 1
    print(f"kept {kept}/{seen} traces -> {args.out}")


if __name__ == "__main__":
    main()
