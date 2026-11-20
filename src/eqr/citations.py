"""Turn the agent's internal source ids into numbered, linkable references for the PDF.

Also rebuilds the underlying DATA from the saved trace. The PDF's tables are rendered from the tool outputs
(SEC XBRL, comps, valuation), never from numbers the LLM re-typed, so a transcription slip by the model cannot
reach the PDF. The model's prose is rendered as written, with its citations resolved to links.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import edgar

ID_RE = re.compile(r"(?:edgar|calc):[A-Za-z0-9_.:\-]+|web:https?://[^\s;,)\]`<>]+")


def tool_calls(trace: dict) -> Iterator[tuple[str, dict, dict]]:
    """Yield (tool name, input, parsed result) for every successful tool call in a trace, in order."""
    uses: dict[str, tuple[str, dict]] = {}
    for m in trace.get("messages", []):
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and not b.get("is_error") and b.get("tool_use_id") in uses:
                try:
                    yield (*uses[b["tool_use_id"]], json.loads(b["content"]))
                except (json.JSONDecodeError, TypeError):
                    continue


def _all_sources(obj) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "source" and isinstance(v, str):
                found.add(v)
            elif k == "sources" and isinstance(v, list):
                found.update(x for x in v if isinstance(x, str))
            else:
                found |= _all_sources(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _all_sources(v)
    return found


@dataclass
class Artifacts:
    company: str | None = None
    financials: dict[str, dict[int, dict]] = field(default_factory=dict)  # metric -> fiscal_year -> row
    comps: list[dict] = field(default_factory=list)
    comps_medians: dict = field(default_factory=dict)
    comps_calls: int = 0
    valuation: list[dict] = field(default_factory=list)
    valuation_medians: dict = field(default_factory=dict)
    valuation_calls: int = 0
    notes: list[str] = field(default_factory=list)
    inputs_of: dict[str, set[str]] = field(default_factory=dict)  # derived source id -> underlying source ids
    titles: dict[str, str] = field(default_factory=dict)  # web url -> page title


def _merge_rows(existing: list[dict], new: list[dict]) -> list[dict]:
    by = {r["ticker"]: r for r in existing}
    for r in new:
        by[r["ticker"]] = r  # latest call wins, first-seen order kept
    return list(by.values())


def artifacts_from_trace(trace: dict) -> Artifacts:
    a = Artifacts()
    for name, _inp, res in tool_calls(trace):
        if name == "get_financials":
            a.company = res.get("company") or a.company
            for metric, rows in res.get("metrics", {}).items():
                for r in rows:
                    a.financials.setdefault(metric, {})[r["fiscal_year"]] = r
        elif name == "get_comps":
            a.comps = _merge_rows(a.comps, res.get("rows", []))
            a.comps_medians, a.comps_calls = res.get("medians", {}), a.comps_calls + 1
            a.notes += [n for n in res.get("notes", []) if n not in a.notes]
            for r in res.get("rows", []):
                a.inputs_of[r["source"]] = _all_sources(r["metrics"]) - {r["source"]}
        elif name == "get_valuation":
            a.valuation = _merge_rows(a.valuation, res.get("rows", []))
            a.valuation_medians, a.valuation_calls = res.get("medians", {}), a.valuation_calls + 1
            a.notes += [n for n in res.get("notes", []) if n not in a.notes]
            for r in res.get("rows", []):
                a.inputs_of[r["source"]] = set(r.get("sources", [])) - {r["source"]}
        elif name == "web_search":
            for r in res.get("results", []):
                if r.get("title"):
                    a.titles[r["url"]] = r["title"]
    return a


def default_filing_info(ticker: str, accession: str) -> tuple[str, str | None]:
    """(human label, SEC filing-index URL) for an accession number."""
    try:
        cik, _ = edgar.lookup(ticker)
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"
        label = f"{ticker.upper()} SEC filing {accession}"
        try:
            recent = edgar._submissions(cik)["filings"]["recent"]
            i = recent["accessionNumber"].index(accession)
            label = (f"{ticker.upper()} Form {recent['form'][i]}, period ending {recent['reportDate'][i]}, "
                     f"filed {recent['filingDate'][i]} (accession {accession})")
        except Exception:
            pass
        return label, url
    except Exception:
        return f"SEC filing {accession}", None


class Refs:
    """Numbers references in order of first citation, so numbering follows reading order."""

    def __init__(self, inputs_of: dict[str, set[str]] | None = None, titles: dict[str, str] | None = None,
                 filing_info: Callable[[str, str], tuple[str, str | None]] = default_filing_info):
        self.inputs_of, self.titles, self.filing_info = inputs_of or {}, titles or {}, filing_info
        self.items: list[dict] = []
        self._idx: dict[str, int] = {}

    def _leaves(self, sid: str, depth: int = 0) -> list[str]:
        if sid.startswith(("edgar:comps:", "calc:valuation:")) and depth < 4:
            out: list[str] = []
            for i in sorted(self.inputs_of.get(sid, [])):
                out += self._leaves(i, depth + 1)
            return out or [sid]
        return [sid]

    @staticmethod
    def _key(leaf: str) -> str:
        p = leaf.split(":")
        if leaf.startswith("edgar:facts:"):
            return f"filing:{p[2]}:{p[-1]}"  # accession is always last
        if leaf.startswith("edgar:filing:"):
            return f"filing:{p[2]}:{p[3]}"
        return leaf

    def _describe(self, key: str, leaf: str) -> tuple[str, str | None]:
        if key.startswith("filing:"):
            _, ticker, acc = key.split(":")
            return self.filing_info(ticker, acc)
        if leaf.startswith("web:"):
            url = leaf[4:]
            return self.titles.get(url) or urlparse(url).netloc or url, url
        return leaf, None  # derived id we could not expand

    def cite(self, ids: list[str]) -> list[int]:
        nums: set[int] = set()
        for sid in ids:
            for leaf in self._leaves(sid):
                key = self._key(leaf)
                if key not in self._idx:
                    label, url = self._describe(key, leaf)
                    self.items.append({"label": label, "url": url})
                    self._idx[key] = len(self.items)
                nums.add(self._idx[key])
        return sorted(nums)
