"""Engine-independent result of an agent run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import ResearchNote


@dataclass
class RunResult:
    note: ResearchNote | None
    trace: dict
    trace_path: Path | None = None
    error: str | None = None
