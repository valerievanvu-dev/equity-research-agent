"""Re-run ONLY the PDF + email step from a saved run (no LLM calls, no cost) and print the exact result.
Use it to debug email setup without waiting for a full agent run.

    uv run python scripts/deliver_from_trace.py runs/<run-id>.json
    uv run python scripts/deliver_from_trace.py "$(ls -t runs/*.json | head -1)"     # most recent run
"""
import json
import sys
from pathlib import Path

from eqr.cli import deliver
from eqr.schemas import ResearchNote

trace = json.loads(Path(sys.argv[1]).read_text())
note = ResearchNote.model_validate(trace["note"])
print(f"re-delivering {note.ticker} from {sys.argv[1]}")
deliver(note, trace, note.ticker, no_email=False)
