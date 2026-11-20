"""The agent's output contract. Forcing structure is what makes grading possible."""
from __future__ import annotations

import html
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _unescape(v):
    """Models sometimes emit HTML-escaped prose ("SG&amp;A"). Decode it. Applied to prose fields ONLY: source ids
    (URLs can legitimately contain '&amp;') must stay byte-for-byte so citation checks still match."""
    return [html.unescape(x) for x in v] if isinstance(v, list) else html.unescape(v)


class Metric(BaseModel):
    name: str = Field(description="Canonical name, e.g. revenue, net_income, operating_margin")
    period: str = Field(description="Fiscal period like FY2024")
    value: float = Field(description="Raw units: USD not millions; percentages as 29.8 with unit 'percent'")
    unit: str = Field(description="USD, USD/shares, or percent")
    sources: list[str] = Field(min_length=1, description="Exact `source` strings from tool results this value came from")


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(min_length=1, description="Exact `source` strings supporting this claim")

    _clean = field_validator("text")(_unescape)


class CompsRow(BaseModel):
    ticker: str
    period_end: str = Field(description="Latest annual period end, YYYY-MM-DD, exactly as in get_comps")
    metrics: dict[str, float] = Field(description="Copy values from get_comps exactly: revenue_growth_pct, gross_margin_pct, operating_margin_pct, ...")
    sources: list[str] = Field(min_length=1, description="The row's `source` id from get_comps")


class ValuationRow(BaseModel):
    ticker: str
    price_date: str | None = Field(default=None, description="Date of the price used, exactly as in get_valuation")
    metrics: dict[str, float] = Field(description="Copy exactly from get_valuation: ev_sales, ev_ebit, pe, fcf_yield_pct, market_cap, ...")
    flags: list[str] = Field(default_factory=list, description="Copy the row's flags")
    sources: list[str] = Field(min_length=1, description="The row's `source` id and/or `sources` from get_valuation")


class ScenarioOut(BaseModel):
    name: Literal["bear", "base", "bull"]
    weight: float
    growth_pct: float
    multiple: float
    price: float
    rationale: str

    _clean = field_validator("rationale")(_unescape)


class Recommendation(BaseModel):
    rating: Literal["Buy", "Hold", "Sell"] = Field(description="Copy exactly from get_price_target; never change it")
    price_target: float = Field(description="12-month price target in USD, exactly as computed")
    horizon_months: int = 12
    current_price: float
    price_date: str | None = None
    expected_total_return_pct: float
    method: str
    conviction: str = Field(description="Copy from get_price_target: Normal or Low")
    scenarios: list[ScenarioOut]
    rationale: str = Field(description="Why this rating, consistent with the computed result, citing sources inline")
    flags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(min_length=1, description="The `source` / `sources` ids from get_price_target")

    _clean = field_validator("rationale")(_unescape)


class ResearchNote(BaseModel):
    ticker: str
    company: str
    summary: str = Field(description="At most 3 sentences")
    metrics: list[Metric]
    thesis: str
    bull_case: list[Claim]
    bear_case: list[Claim]
    key_risks: list[Claim]
    recommendation: Recommendation | None = Field(default=None, description="Rating and price target, copied exactly from get_price_target")
    comps: list[CompsRow] = Field(default_factory=list, description="Peer table: the subject company plus its peers, from get_comps")
    valuation: list[ValuationRow] = Field(default_factory=list, description="Valuation multiples for subject and peers, from get_valuation")
    peer_commentary: str = Field(default="", description="How the company ranks vs peers and why, citing the comps rows")
    open_questions: list[str] = Field(default_factory=list, description="Data gaps or things you could not verify")

    _clean = field_validator("summary", "thesis", "peer_commentary", "open_questions")(_unescape)
