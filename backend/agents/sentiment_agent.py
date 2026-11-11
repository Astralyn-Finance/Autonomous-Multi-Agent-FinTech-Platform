"""Gemini-powered financial research agent that proposes BL views.

The agent never computes portfolio weights and never applies its own proposed
confidence values automatically. The caller must explicitly collect user
approval (or an override) through ``finalize_confidences`` before passing the
views to the Black-Litterman estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import re
from typing import Any, Optional

import pandas as pd

from .news_retriever import NewsArticle, NewsRetriever


@dataclass(frozen=True)
class SentimentViewProposal:
    ticker: str
    expected_annual_return: float
    recommended_confidence: float
    rationale: str
    evidence_urls: tuple[str, ...] = ()
    additional_note: str = ""


@dataclass
class SentimentResearchResult:
    proposals: dict[str, SentimentViewProposal]
    articles_by_ticker: dict[str, list[NewsArticle]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def views(self) -> dict[str, float]:
        return {ticker: proposal.expected_annual_return for ticker, proposal in self.proposals.items()}

    @property
    def recommended_confidences(self) -> dict[str, float]:
        return {ticker: proposal.recommended_confidence for ticker, proposal in self.proposals.items()}


@dataclass(frozen=True)
class ApprovedSentimentInputs:
    views: dict[str, float]
    confidences: dict[str, float]
    additional_notes: dict[str, str]


class SentimentAgentError(RuntimeError):
    """Raised when Gemini returns unusable or untraceable output."""


class GeminiSentimentAgent:
    """Generate annualized absolute-return views from retrieved research.

    ``current_market_caps`` and ``reference_market_caps`` are passed in by the
    existing market-data pipeline. Reference caps may represent a prior
    snapshot; this class does not invent historical market-cap observations.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.8-flash",
                 news_retriever: Optional[NewsRetriever] = None, client=None):
        # Load a project-local .env when python-dotenv is installed; keeping
        # this optional preserves importability in minimal test environments.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        self.news_retriever = news_retriever or NewsRetriever()
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in the environment or .env file.")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install the Gemini SDK with: pip install google-genai") from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _as_mapping(values) -> dict:
        if values is None:
            return {}
        if isinstance(values, pd.Series):
            return values.to_dict()
        return dict(values)

    @staticmethod
    def _finite_number(value, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number

    def _build_prompt(self, tickers: list[str], base_expected_returns: dict,
                      current_market_caps: dict, reference_market_caps: dict,
                      articles_by_ticker: dict, additional_notes: dict,
                      return_horizon: str) -> str:
        research = {}
        for ticker in tickers:
            research[ticker] = {
                "baseline_expected_annual_return": base_expected_returns.get(ticker),
                "current_market_cap": current_market_caps.get(ticker),
                "reference_market_cap": reference_market_caps.get(ticker),
                "market_cap_units": "Use the supplied units; do not infer missing values",
                "user_or_constraint_notes": additional_notes.get(ticker, ""),
                "news": [a.to_prompt_dict() for a in articles_by_ticker.get(ticker, [])],
            }

        return f"""You are a financial research assistant for an Indian-equity portfolio research system.
Your task is to propose Black-Litterman ABSOLUTE expected-return views, not portfolio weights.

Return horizon: {return_horizon}. Every expected return MUST be an annualized decimal
in the same units and horizon as the supplied baseline expected annual returns (e.g. 0.12 = 12%).
Use only the supplied news and market-cap context. Do not invent facts, prices, market caps,
news, article URLs, or historical observations. Missing market-cap values must remain unknown.
Treat user notes as research context, not permission to violate hard portfolio constraints.
A market-cap change is meaningful only when both current and reference values are supplied
and their dates/meaning are comparable. Do not claim causality from market-cap changes alone.

For each ticker with sufficient relevant evidence, return one view with:
- ticker
- expected_annual_return (decimal)
- recommended_confidence (strictly between 0 and 1)
- rationale (brief, evidence-based)
- evidence_urls (only exact URLs present in supplied news)
- additional_note (relevant risk/constraint context that may inform research)

Confidence represents confidence in the view, NOT the investor's risk tolerance and NOT a
probability that the stock will rise. Be conservative when evidence is sparse, stale, conflicting,
or based on a single source. If evidence is insufficient, omit that ticker. Do not force a view.

Return ONLY valid JSON in this exact shape:
{{"views":[{{"ticker":"TICKER","expected_annual_return":0.10,
"recommended_confidence":0.55,"rationale":"...","evidence_urls":[],
"additional_note":"..."}}]}}

RESEARCH INPUT:
{json.dumps(research, ensure_ascii=False, default=str)}
"""

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text or not text.strip():
            raise SentimentAgentError("Gemini returned an empty response")
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise SentimentAgentError(f"Gemini response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("views"), list):
            raise SentimentAgentError("Gemini JSON must contain a 'views' array")
        return data

    def generate_views(self, tickers: list[str], base_expected_returns,
                       current_market_caps=None, reference_market_caps=None,
                       additional_notes: Optional[dict[str, str]] = None,
                       company_names: Optional[dict[str, str]] = None,
                       return_horizon: str = "1 year",
                       max_articles_per_ticker: int = 8,
                       lookback_days: int = 14) -> SentimentResearchResult:
        tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
        if not tickers:
            raise ValueError("At least one ticker is required")
        base = self._as_mapping(base_expected_returns)
        current_caps = self._as_mapping(current_market_caps)
        reference_caps = self._as_mapping(reference_market_caps)
        notes = additional_notes or {}

        for ticker in tickers:
            if ticker not in base:
                raise ValueError(f"Missing baseline expected return for {ticker}")
            base[ticker] = self._finite_number(base[ticker], f"baseline return for {ticker}")
            for label, caps in (("current market cap", current_caps), ("reference market cap", reference_caps)):
                if caps.get(ticker) is not None:
                    caps[ticker] = self._finite_number(caps[ticker], f"{label} for {ticker}")
                    if caps[ticker] <= 0:
                        raise ValueError(f"{label} for {ticker} must be positive")

        articles = self.news_retriever.retrieve_many(
            tickers, company_names=company_names,
            max_articles_per_ticker=max_articles_per_ticker,
            lookback_days=lookback_days,
        )
        total_articles = sum(len(items) for items in articles.values())
        if total_articles == 0:
            return SentimentResearchResult(
                proposals={}, articles_by_ticker=articles,
                diagnostics={"status": "no_news", "message": "No RSS articles retrieved; no views generated."},
            )

        prompt = self._build_prompt(tickers, base, current_caps, reference_caps,
                                    articles, notes, return_horizon)
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except Exception as exc:
            raise SentimentAgentError(f"Gemini request failed: {exc}") from exc

        payload = self._extract_json(getattr(response, "text", ""))
        article_urls = {ticker: {a.url for a in items} for ticker, items in articles.items()}
        proposals: dict[str, SentimentViewProposal] = {}
        for item in payload["views"]:
            if not isinstance(item, dict):
                raise SentimentAgentError("Each item in views must be a JSON object")
            ticker = str(item.get("ticker", "")).strip().upper()
            if ticker not in tickers:
                raise SentimentAgentError(f"Gemini returned unknown ticker: {ticker!r}")
            if ticker in proposals:
                raise SentimentAgentError(f"Gemini returned duplicate view for {ticker}")
            expected = self._finite_number(item.get("expected_annual_return"), f"view for {ticker}")
            confidence = self._finite_number(item.get("recommended_confidence"), f"confidence for {ticker}")
            if not 0.0 < confidence < 1.0:
                raise SentimentAgentError(f"Confidence for {ticker} must be strictly between 0 and 1")
            rationale = str(item.get("rationale", "")).strip()
            if not rationale:
                raise SentimentAgentError(f"Gemini did not provide a rationale for {ticker}")
            supplied_urls = item.get("evidence_urls", []) or []
            if not isinstance(supplied_urls, list):
                raise SentimentAgentError(f"evidence_urls for {ticker} must be a list")
            # Do not allow model-invented URLs to be presented as retrieved evidence.
            valid_urls = tuple(url for url in supplied_urls if url in article_urls.get(ticker, set()))
            proposals[ticker] = SentimentViewProposal(
                ticker=ticker,
                expected_annual_return=expected,
                recommended_confidence=confidence,
                rationale=rationale,
                evidence_urls=valid_urls,
                additional_note=str(item.get("additional_note", "")).strip(),
            )

        return SentimentResearchResult(
            proposals=proposals,
            articles_by_ticker=articles,
            diagnostics={
                "status": "review_required",
                "model": self.model,
                "articles_retrieved": total_articles,
                "views_proposed": len(proposals),
                "confidence_requires_user_approval": True,
                "return_horizon": return_horizon,
            },
        )


    def finalize_confidences(result: SentimentResearchResult,
                            accepted_recommendations: Optional[set[str]] = None,
                            confidence_overrides: Optional[dict[str, float]] = None) -> ApprovedSentimentInputs:
        """Finalize confidence only after an explicit user decision per proposal.

        Every proposed ticker must appear in exactly one of:
        * ``accepted_recommendations``: use Gemini's recommended confidence;
        * ``confidence_overrides``: use the user's own confidence value.

        Missing decisions raise an error; no default or silent auto-approval exists.
        """
        accepted = {str(t).upper() for t in (accepted_recommendations or set())}
        overrides = {str(t).upper(): v for t, v in (confidence_overrides or {}).items()}
        proposed = set(result.proposals)
        unknown = (accepted | set(overrides)) - proposed
        if unknown:
            raise ValueError(f"Confidence decision supplied for unknown ticker(s): {sorted(unknown)}")
        overlap = accepted & set(overrides)
        if overlap:
            raise ValueError(f"Choose either accept or override for: {sorted(overlap)}")
        unresolved = proposed - accepted - set(overrides)
        if unresolved:
            raise ValueError(f"User decision required for confidence of: {sorted(unresolved)}")

        final_confidences = {}
        for ticker in proposed:
            if ticker in accepted:
                final_confidences[ticker] = result.proposals[ticker].recommended_confidence
            else:
                value = float(overrides[ticker])
                if not math.isfinite(value) or not 0.0 < value < 1.0:
                    raise ValueError(f"User confidence for {ticker} must be strictly between 0 and 1")
                final_confidences[ticker] = value

        return ApprovedSentimentInputs(
            views=result.views,
            confidences=final_confidences,
            additional_notes={t: p.additional_note for t, p in result.proposals.items() if p.additional_note},
        )
