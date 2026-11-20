"""Re-render the PDF from a saved run trace (no LLM calls, no cost). Handy for iterating on layout.

    uv run python scripts/pdf_from_trace.py runs/<run-id>.json [out.pdf]
"""
import json
import sys
from pathlib import Path

from eqr.pdf import build_pdf
from eqr.schemas import ResearchNote

trace = json.loads(Path(sys.argv[1]).read_text())
note = ResearchNote.model_validate(trace["note"])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports") / f"{note.ticker}-from-trace.pdf"
print("wrote", build_pdf(note, trace, out))
