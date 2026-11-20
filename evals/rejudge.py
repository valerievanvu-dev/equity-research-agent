"""Re-judge saved traces without re-running the agent: compare judge versions/models on identical notes.

    uv run python evals/rejudge.py --judge gpt-5.5 [--limit 15]
"""
import argparse, glob, json, statistics as st
from eqr.judge import judge
from eqr.schemas import ResearchNote

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default=None)
ap.add_argument("--limit", type=int, default=15)
args = ap.parse_args()

paths = sorted(glob.glob("runs/*.json"))
rows = []
for p in paths:
    t = json.load(open(p))
    old = t.get("grade", {}).get("judge", {})
    if not t.get("note") or "groundedness" not in old:
        continue
    new = judge(ResearchNote.model_validate(t["note"]), t, model=args.judge)
    rows.append((old, new))
    if len(rows) >= args.limit:
        break
print(f"{len(rows)} notes re-judged")
for d in ("groundedness", "insight", "balance", "clarity"):
    print(f"  {d:13} old {st.mean(o[d] for o, _ in rows):.2f} -> new {st.mean(getattr(n, d) for _, n in rows):.2f}")
print(f"  unsupported claims/note: old {st.mean(len(o['unsupported_claims']) for o, _ in rows):.1f} -> new {st.mean(len(n.unsupported_claims) for _, n in rows):.1f}")
