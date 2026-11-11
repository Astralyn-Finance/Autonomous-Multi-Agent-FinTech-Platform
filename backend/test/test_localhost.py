"""portfolio_optimization/test_localhost.py

Automated local smoke-test verifying market data, BL calculation,
optimizer routing, constraint validation, and server endpoint responses.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

from backend.data.data_fetcher import load_nifty50_data
from backend.agents.orchestrator import PortfolioOrchestrator
from backend.optimization.optimizer_selector import OptimizerRegistry, UserPortfolioRequirements
from backend.app.server import app


def test_complete_stack():
    print("=" * 80)
    print("       PORTFOLIO OPTIMIZATION PLATFORM: LOCALHOST VERIFICATION        ")
    print("=" * 80)

    # 1. Verify Registry Routing
    print("\n[CHECK 1] Testing Optimizer Capability Registry Routing...")
    req_miqp = UserPortfolioRequirements(exact_k_stocks=3, max_weight=0.45)
    solver_miqp, _ = OptimizerRegistry.select_optimizer(req_miqp)
    assert solver_miqp == "MIQP", f"Expected MIQP, got {solver_miqp}"
    print("  -> Exact K=3 successfully routed to MIQP.")

    req_mvo = UserPortfolioRequirements(min_weight=0.05, max_weight=0.30)
    solver_mvo, _ = OptimizerRegistry.select_optimizer(req_mvo)
    assert solver_mvo == "MVO_QP", f"Expected MVO_QP, got {solver_mvo}"
    print("  -> Continuous bounds successfully routed to MVO_QP.")

    # 2. Verify Market Data Fetcher
    print("\n[CHECK 2] Testing Market Data Fetcher...")
    mu, cov, caps, sectors, source = load_nifty50_data(years=3, limit=6)
    assert len(mu) == 6, f"Expected 6 assets, got {len(mu)}"
    assert not cov.empty, "Covariance matrix is empty"
    print(f"  -> Successfully loaded {len(mu)} stocks from source '{source}'.")

    # 3. Verify End-to-End Orchestrator Pipeline
    print("\n[CHECK 3] Testing Full Orchestration Pipeline...")
    orchestrator = PortfolioOrchestrator()
    caps_dict = caps.to_dict() if caps is not None else {t: 1_000_000.0 for t in mu.index}

    pipeline_res = orchestrator.run(
        tickers=list(mu.index),
        baseline_returns=mu,
        cov_matrix=cov,
        current_market_caps=caps_dict,
        constraints=req_miqp,
        sector_mapping=sectors,
        auto_confirm_views=True,
    )

    opt_result = pipeline_res["result"]
    assert opt_result.algorithm == "MIQP", f"Algorithm mismatch: {opt_result.algorithm}"
    assert opt_result.active_stock_count == 3, f"Expected exactly 3 stocks, got {opt_result.active_stock_count}"
    print(f"  -> Pipeline executed: Solver={opt_result.algorithm}, Sharpe={opt_result.sharpe_ratio:.2f}")

    # 4. Verify FastAPI Endpoints via TestClient
    print("\n[CHECK 4] Testing FastAPI Server Endpoints via TestClient...")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, f"Health check failed: {health.text}"
    print("  -> /health endpoint responded HTTP 200 OK.")

    chat_payload = {
        "session_id": "test_session_1",
        "message": "I want to invest 5 lakhs with moderate risk in exactly 3 stocks.",
    }
    chat_resp = client.post("/api/chat/constraints", json=chat_payload)
    assert chat_resp.status_code == 200, f"Chat endpoint failed: {chat_resp.text}"
    print("  -> /api/chat/constraints responded HTTP 200 OK.")

    print("\n" + "=" * 80)
    print("              ALL LOCALHOST VERIFICATION TESTS PASSED!                ")
    print("=" * 80)


if __name__ == "__main__":
    test_complete_stack()