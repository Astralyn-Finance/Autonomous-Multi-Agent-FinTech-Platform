"""
Example FastAPI endpoint showing how the frontend interacts with ConstraintExtractionAgent.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd

from agents.constraint_agent import ConstraintExtractionAgent
from agents.orchestrator import PortfolioOrchestrator

router = APIRouter()
agent = ConstraintExtractionAgent()
orchestrator = PortfolioOrchestrator()

# Temporary in-memory session store (or use Redis / DB)
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    status: str                         # "collecting" | "ready" | "error"
    reply: str                          # Question to display in UI or success summary
    extracted_constraints: Dict[str, Any]
    optimization_completed: bool = False
    allocation_summary: Optional[List[Dict[str, Any]]] = None


@router.post("/chat/constraints", response_model=ChatResponse)
async def chat_constraint_extraction(payload: ChatRequest):
    session_id = payload.session_id
    current_state = SESSION_STORE.get(session_id, {})

    # 1. Process message through the agent
    res = agent.process_chat_message(
        user_message=payload.message,
        current_state=current_state,
    )

    # Update in-memory state
    SESSION_STORE[session_id] = res.extracted_constraints

    if res.status == "collecting":
        return ChatResponse(
            session_id=session_id,
            status="collecting",
            reply=res.follow_up_question or "Could you clarify your investment amount and risk profile?",
            extracted_constraints=res.extracted_constraints,
            optimization_completed=False,
        )

    elif res.status == "ready":
        # 2. Trigger existing optimizer pipeline directly with the validated requirements
        user_reqs = res.user_portfolio_requirements

        # Mock / Ingest universe market data
        tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS"]
        baseline_returns = pd.Series([0.14, 0.12, 0.11, 0.13, 0.15, 0.10], index=tickers)
        cov_matrix = pd.DataFrame(0.04 * (pd.np.eye(len(tickers)) + 0.1), index=tickers, columns=tickers)
        current_caps = {t: 1_000_000.0 for t in tickers}

        opt_result = orchestrator.run(
            tickers=tickers,
            baseline_returns=baseline_returns,
            cov_matrix=cov_matrix,
            current_market_caps=current_caps,
            constraints=user_reqs,
            auto_confirm_views=True,
        )

        table_records = opt_result["recommendation_report"].to_dict(orient="records")

        return ChatResponse(
            session_id=session_id,
            status="ready",
            reply=f"All requirements gathered! Routed to {opt_result['result'].algorithm} solver successfully.",
            extracted_constraints=res.extracted_constraints,
            optimization_completed=True,
            allocation_summary=table_records,
        )

    else:
        raise HTTPException(status_code=400, detail=res.error_message or "Extraction failed.")