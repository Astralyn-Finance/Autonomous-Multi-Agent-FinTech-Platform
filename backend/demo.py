"""portfolio_optimization/demo.py

End-to-end verification of Constraint-Aware Routing and Single BL Posterior.
"""
import numpy as np
import pandas as pd

from backend.agents.sentiment_agent import GeminiSentimentAgent
from backend.agents.orchestrator import PortfolioOrchestrator
from backend.optimization.optimizer_selector import UserPortfolioRequirements


def main():
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS"]
    np.random.seed(42)

    baseline_returns = pd.Series([0.14, 0.12, 0.11, 0.13, 0.15, 0.10], index=tickers)
    A = np.random.randn(len(tickers), len(tickers)) * 0.05
    cov_matrix = pd.DataFrame(A @ A.T + np.eye(len(tickers)) * 0.02, index=tickers, columns=tickers)

    current_caps = {
        "RELIANCE.NS": 2000000.0,
        "TCS.NS": 1400000.0,
        "HDFCBANK.NS": 1200000.0,
        "INFY.NS": 600000.0,
        "ICICIBANK.NS": 700000.0,
        "ITC.NS": 500000.0,
    }

    sectors = {
        "RELIANCE.NS": "Energy",
        "TCS.NS": "Technology",
        "HDFCBANK.NS": "Financials",
        "INFY.NS": "Technology",
        "ICICIBANK.NS": "Financials",
        "ITC.NS": "Consumer Goods",
    }

    orchestrator = PortfolioOrchestrator()

    print("=" * 80)
    print("TEST 1: Exact K=3 Stocks + Semicontinuous Threshold -> Expected: MIQP")
    print("=" * 80)
    req1 = UserPortfolioRequirements(
        exact_k_stocks=3,
        min_allocation_for_selected=0.15,
        max_weight=0.50,
        investment_amount=500000.0,
    )
    res1 = orchestrator.run(
        tickers=tickers,
        baseline_returns=baseline_returns,
        cov_matrix=cov_matrix,
        current_market_caps=current_caps,
        constraints=req1,
        sector_mapping=sectors,
        auto_confirm_views=True,
    )
    print(f"Selected Solver : {res1['result'].algorithm}")
    print(f"Sharpe Ratio    : {res1['result'].sharpe_ratio:.2f}")
    print(f"Active Holdings : {res1['result'].active_stock_count}")
    print(res1["recommendation_report"][["Ticker", "Weight (%)", "Allocation (INR)", "Risk Constraint Flag"]])

    print("\n" + "=" * 80)
    print("TEST 2: Standard Continuous Bounds -> Expected: MVO_QP")
    print("=" * 80)
    req2 = UserPortfolioRequirements(
        min_weight=0.05,
        max_weight=0.30,
        investment_amount=1000000.0,
    )
    res2 = orchestrator.run(
        tickers=tickers,
        baseline_returns=baseline_returns,
        cov_matrix=cov_matrix,
        current_market_caps=current_caps,
        constraints=req2,
        sector_mapping=sectors,
        auto_confirm_views=True,
    )
    print(f"Selected Solver : {res2['result'].algorithm}")
    print(f"Sharpe Ratio    : {res2['result'].sharpe_ratio:.2f}")
    print(res2["recommendation_report"][["Ticker", "Weight (%)", "Allocation (INR)", "Risk Constraint Flag"]])

    print("\n" + "=" * 80)
    print("AUDIT TRAIL LOGS:")
    print("=" * 80)
    for log in res1["audit_trail"]:
        print(f"  [AUDIT] {log}")


if __name__ == "__main__":
    main()