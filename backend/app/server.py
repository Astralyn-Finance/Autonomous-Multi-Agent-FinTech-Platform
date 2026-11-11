"""
FastAPI backend service exposing conversational constraint extraction,
Black-Litterman posterior calculation, and constraint-aware optimizer execution.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd

from agents.constraint_agent import ConstraintExtractionAgent
from data.data_fetcher import load_nifty50_data
from agents.orchestrator import PortfolioOrchestrator
from schemas.constraint_schema import UserPortfolioRequirements

load_dotenv()

app = FastAPI(
    title="Autonomous Quantitative Portfolio Optimization Platform",
    version="2.0.0",
)

# Enable CORS for local React/Vite/frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent singletons
constraint_agent = ConstraintExtractionAgent()
orchestrator = PortfolioOrchestrator()

# In-memory session store for chat state
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class OptimizationDirectRequest(BaseModel):
    investment_amount: float 
    target_risk_aversion: float 
    min_weight: float 
    max_weight: float 
    exact_k_stocks: Optional[int] = None
    max_k_stocks: Optional[int] = None
    min_allocation_for_selected: Optional[float] = None
    decompose_by_sector: bool = False
    universe_limit: int
    auto_confirm_views: bool = True


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend operational readiness."""
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "online",
        "gemini_api_configured": has_api_key,
        "engine": "Black-Litterman + Constraint-Aware Multi-Optimizer",
    }


@app.post("/api/chat/constraints")
def chat_constraint_extraction(payload: ChatRequest):
    """Conversational endpoint: processes user intent, extracts constraints,

    returns follow-up questions or marks state as ready.
    """
    session_id = payload.session_id
    current_state = SESSION_STORE.get(session_id, {})

    try:
        response = constraint_agent.process_chat_message(
            user_message=payload.message,
            current_state=current_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    # Update session memory
    SESSION_STORE[session_id] = response.extracted_constraints

    return {
        "session_id": session_id,
        "status": response.status,
        "reply": response.follow_up_question or "All requirements successfully gathered!",
        "missing_fields": response.missing_fields,
        "extracted_constraints": response.extracted_constraints,
        "is_ready_for_optimization": response.status == "ready",
        "error_message": response.error_message,
    }


@app.post("/api/portfolio/run")
def run_portfolio_pipeline(payload: OptimizationDirectRequest):
    """Executes the complete optimization pipeline using live Nifty 50 data."""
    # 1. Load market inputs via unified data_fetcher
    try:
        baseline_returns, cov_matrix, market_caps, sector_mapping, source = load_nifty50_data(
            years=3, limit=payload.universe_limit
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Market data loading failed: {str(exc)}")

    tickers = list(baseline_returns.index)
    caps_dict = market_caps.to_dict() if market_caps is not None else {t: 1_000_000.0 for t in tickers}

    # 2. Build UserPortfolioRequirements
    req = UserPortfolioRequirements(
        target_risk_aversion=payload.target_risk_aversion,
        min_weight=payload.min_weight,
        max_weight=payload.max_weight,
        exact_k_stocks=payload.exact_k_stocks,
        max_k_stocks=payload.max_k_stocks,
        min_allocation_for_selected=payload.min_allocation_for_selected,
        decompose_by_sector=payload.decompose_by_sector,
        investment_amount=payload.investment_amount,
    )

    # 3. Execute End-to-End Orchestrator Pipeline
    try:
        pipeline_output = orchestrator.run(
            tickers=tickers,
            baseline_returns=baseline_returns,
            cov_matrix=cov_matrix,
            current_market_caps=caps_dict,
            constraints=req,
            sector_mapping=sector_mapping,
            auto_confirm_views=payload.auto_confirm_views,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pipeline execution error: {str(exc)}")

    result = pipeline_output["result"]
    report_df = pipeline_output["recommendation_report"]

    return {
        "status": "success",
        "algorithm_selected": result.algorithm,
        "expected_return": result.expected_return,
        "expected_volatility": result.expected_volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "active_stock_count": result.active_stock_count,
        "solve_time_sec": result.solve_time_sec,
        "allocations": report_df.to_dict(orient="records"),
        "audit_trail": pipeline_output["audit_trail"],
    }


@app.get("/", response_class=HTMLResponse)
def local_web_interface():
    """Minimal standalone localhost verification console."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Autonomous Portfolio Allocation Platform</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #121212; color: #E0E0E0; margin: 0; padding: 25px; }
            h2, h3 { color: #4CAF50; }
            .container { max-width: 900px; margin: 0 auto; background: #1E1E1E; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
            input, button, textarea { width: 100%; padding: 12px; margin-top: 8px; border-radius: 4px; border: 1px solid #333; background: #2A2A2A; color: #FFF; box-sizing: border-box; }
            button { background: #4CAF50; color: white; font-weight: bold; cursor: pointer; border: none; margin-top: 15px; }
            button:hover { background: #45a049; }
            pre { background: #000; padding: 15px; border-radius: 4px; overflow-x: auto; color: #00FF66; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Autonomous Portfolio Allocation Platform</h2>
            <p>Status: <span style="color:#4CAF50; font-weight:bold;">Online on Localhost:8000</span></p>
            
            <h3>1. Conversational Constraint Gathering</h3>
            <textarea id="chatInput" rows="3" placeholder="e.g., I want to invest 5 lakhs with moderate risk across exactly 4 stocks with max 30% each."></textarea>
            <button onclick="sendChatMessage()">Send Message to Agent</button>
            <pre id="chatOutput">Agent responses and extracted parameters will appear here...</pre>

            <h3>2. Direct Pipeline Run</h3>
            <button onclick="runDirectPipeline()">Execute Full Optimization Pipeline (Nifty 50)</button>
            <pre id="pipelineOutput">Allocation results, Sharpe ratio, and audit trail will appear here...</pre>
        </div>

        <script>
            const sessionId = "local_session_" + Math.random().toString(36).substring(7);

            async function sendChatMessage() {
                const msg = document.getElementById('chatInput').value;
                if (!msg) return;
                document.getElementById('chatOutput').innerText = "Processing with Gemini Agent...";
                try {
                    const res = await fetch('/api/chat/constraints', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({session_id: sessionId, message: msg})
                    });
                    const data = await res.json();
                    document.getElementById('chatOutput').innerText = JSON.stringify(data, null, 2);
                } catch(e) {
                    document.getElementById('chatOutput').innerText = "Error: " + e.message;
                }
            }

            async function runDirectPipeline() {
                document.getElementById('pipelineOutput').innerText = "Running data fetcher, sentiment agent, Black-Litterman, and solver routing...";
                try {
                    const res = await fetch('/api/portfolio/run', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            investment_amount: 500000,
                            target_risk_aversion: 2.5,
                            exact_k_stocks: 4,
                            max_weight: 0.35,
                            universe_limit: 8,
                            auto_confirm_views: true
                        })
                    });
                    const data = await res.json();
                    document.getElementById('pipelineOutput').innerText = JSON.stringify(data, null, 2);
                } catch(e) {
                    document.getElementById('pipelineOutput').innerText = "Error: " + e.message;
                }
            }
        </script>
    </body>
    </html>
    """