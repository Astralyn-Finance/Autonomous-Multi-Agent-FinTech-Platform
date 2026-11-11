"""backend/agents/constraint_agent.py

Standalone LLM-based constraint extraction agent for conversational frontends.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from backend.schemas.constraint_schema import (
    ConversationExtractionResponse,
    ExtractedConstraintsModel,
)

logger = logging.getLogger(__name__)


# Mandatory parameters required to run optimization
MANDATORY_FIELDS = [
    "investment_amount",
    "target_risk_aversion",
]


class ConstraintExtractionAgent:
    """Conversational agent that iteratively extracts and validates portfolio constraints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client=None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._client = client

        if self._client is None and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini constraint extractor requires google-genai. Install requirements.txt."
                ) from exc

    def process_chat_message(
        self,
        user_message: str,
        current_state: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ConversationExtractionResponse:
        """Main entry point for frontend chat integration.

        Args:
            user_message: Natural language prompt from the user.
            current_state: Dictionary of already collected constraints from previous turns.
            chat_history: Optional list of past chat messages [{"role": "user"|"assistant", "content": "..."}].

        Returns:
            ConversationExtractionResponse: Ready-to-use payload for the chat UI.
        """
        if not user_message or not user_message.strip():
            return ConversationExtractionResponse(
                status="collecting",
                follow_up_question="Please share your investment amount, risk tolerance, or portfolio preferences.",
                missing_fields=MANDATORY_FIELDS,
                extracted_constraints=current_state or {},
            )

        if self._client is None:
            return ConversationExtractionResponse(
                status="error",
                error_message="GEMINI_API_KEY is not configured.",
            )

        accumulated = dict(current_state or {})

        # 1. Ask LLM to extract updates from natural language
        extracted_patch = self._extract_with_llm(user_message, accumulated, chat_history)

        # Merge new findings into accumulated state
        for key, val in extracted_patch.items():
            if val is not None:
                accumulated[key] = val

        # 2. Validate using Pydantic
        try:
            validated_model = ExtractedConstraintsModel(**accumulated)
            accumulated = validated_model.model_dump(exclude_unset=False)
        except ValidationError as val_err:
            first_err = val_err.errors()[0]["msg"]
            return ConversationExtractionResponse(
                status="collecting",
                follow_up_question=f"I noticed a conflicting constraint: {first_err} How would you like to adjust it?",
                extracted_constraints=accumulated,
                error_message=first_err,
            )

        # 3. Check for missing mandatory fields
        missing = [f for f in MANDATORY_FIELDS if accumulated.get(f) is None]

        # 4. If missing mandatory info, generate a context-aware follow-up question
        if missing:
            follow_up = self._generate_follow_up_question(accumulated, missing, user_message)
            return ConversationExtractionResponse(
                status="collecting",
                follow_up_question=follow_up,
                missing_fields=missing,
                extracted_constraints=accumulated,
            )

        # 5. All mandatory fields satisfied: convert to internal UserPortfolioRequirements
        portfolio_reqs = validated_model.to_user_portfolio_requirements()
        return ConversationExtractionResponse(
            status="ready",
            follow_up_question=None,
            missing_fields=[],
            extracted_constraints=accumulated,
            user_portfolio_requirements=portfolio_reqs,
        )

    def _extract_with_llm(
        self,
        user_message: str,
        current_state: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """Prompts Gemini to extract structured fields and numerical values."""
        prompt = f"""You are a quantitative portfolio constraint assistant. Extract investment constraints from the user message.
Current Accumulated State:
{json.dumps(current_state, default=str)}

Recent User Message:
"{user_message}"

Extract and return a valid JSON object matching these optional fields (only include fields mentioned or implied; omit or set null if not mentioned):
- "investment_amount": float (e.g. 500000. Extract from 5 lakhs, 10k, 1 crore, etc. in numeric base units)
- "target_risk_aversion": float (e.g. 1.5 for aggressive/high risk, 2.5 for moderate/balanced, 5.0 for conservative/low risk)
- "exact_k_stocks": int or null (e.g. "exactly 5 stocks")
- "max_k_stocks": int or null (e.g. "no more than 8 stocks")
- "min_weight": float (e.g. 0.05 for 5%)
- "max_weight": float (e.g. 0.25 for 25% max per stock)
- "min_allocation_for_selected": float or null (e.g. "at least 10% in any picked stock")
- "l1_sparsity_gamma": float or null (if user wants soft sparsity)
- "sector_limits": dict mapping sector to float (e.g. {{"Technology": 0.30, "Financials": 0.25}})
- "decompose_by_sector": bool (true if user asks for sector-by-sector consensus/decomposition)
- "include_tickers": list of strings (NSE tickers with .NS like ["RELIANCE.NS", "TCS.NS"])
- "exclude_tickers": list of strings (NSE tickers to avoid)

Return ONLY the raw JSON object."""



        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw_text = getattr(response, "text", "") or "{}"
        try:
            return json.loads(raw_text)
        except Exception as e:
            logger.warning("Failed to parse LLM extraction JSON: %s. Raw: %s", e, raw_text)
            return {}

    def _generate_follow_up_question(
        self, current_state: Dict[str, Any], missing: List[str], last_user_message: str
    ) -> str:
        """Generates an intuitive, polite single follow-up question for missing constraints."""
        missing_descriptions = {
            "investment_amount": "the total amount of capital you want to invest (e.g., ₹5,00,000)",
            "target_risk_aversion": "your risk tolerance (e.g., conservative, moderate, or aggressive)",
        }
        needed = [missing_descriptions.get(f, f) for f in missing]

        prompt = f"""You are a professional investment chat assistant. The user just said: "{last_user_message}".
We currently know: {json.dumps({k: v for k, v in current_state.items() if v is not None})}
We still need: {', '.join(needed)}.

Write a natural, polite, single-sentence question asking the user for this missing information.
Do not mention variable names like target_risk_aversion. Speak plainly."""

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return getattr(response, "text", "").strip() or f"Could you please specify {needed[0]}?"