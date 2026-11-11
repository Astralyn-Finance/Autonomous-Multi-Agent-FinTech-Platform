"""tests/test_constraint_agent.py

Verification script for conversational constraint extraction and optimizer execution.
"""

import os
import pandas as pd
from backend.agents.constraint_agent import ConstraintExtractionAgent
from backend.agents.orchestrator import PortfolioOrchestrator


def run_multi_turn_test():
    agent = ConstraintExtractionAgent()
    orchestrator = PortfolioOrchestrator()

    session_state = {}

    print("=" * 75)
    print("STEP 1: User provides partial input (amount only)")
    print("=" * 75)
    turn1 = agent.process_chat_message(
        user_message="Hi, I want to invest 10 lakhs in the Indian market with at most 25% in any stock.",
        current_state=session_state,
    )
    session_state = turn1.extracted_constraints
    print(f"Status           : {turn1.status}")
    print(f"Missing Fields   : {turn1.missing_fields}")
    print(f"Follow-up Prompt : {turn1.follow_up_question}")
    print(f"Current State    : {session_state}")
    assert turn1.status == "collecting"
    assert "target_risk_aversion" in turn1.missing_fields

    print("\n" + "=" * 75)
    print("STEP 2: User provides invalid/infeasible input (checking validation)")
    print("=" * 75)
    turn2 = agent.process_chat_message(
        user_message="I want exactly 10 stocks, but keep maximum weight at 5% per stock.",
        current_state=session_state,
    )
    print(f"Status           : {turn2.status}")
    print(f"Validation Error : {turn2.error_message}")
    print(f"Follow-up Prompt : {turn2.follow_up_question}")
    assert turn2.status == "collecting"
    assert "Infeasible" in str(turn2.error_message)

    print("\n" + "=" * 75)
    print("STEP 3: User fixes input & provides risk tolerance (completing requirements)")
    print("=" * 75)
    turn3 = agent.process_chat_message(
        user_message="Okay, pick exactly 4 stocks with maximum 30% each. I have a moderate risk appetite.",
        current_state=session_state,
    )
    session_state = turn3.extracted_constraints
    print(f"Status           : {turn3.status}")
    print(f"Missing Fields   : {turn3.missing_fields}")
    print(f"Extracted Config : {turn3.extracted_constraints}")
    assert turn3.status == "ready"
    assert turn3.user_portfolio_requirements is not None

    print("\n" + "=" * 75)
    print("STEP 4: Run Optimizer with Extracted UserPortfolioRequirements")
    print("=" * 75)
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS"]
    baseline_returns = pd.Series([0.14, 0.12, 0.11, 0.13, 0.15, 0.10], index=tickers)
    cov_matrix = pd.DataFrame(
        0.04 * (pd.np.eye(len(tickers)) + 0.05), index=tickers, columns=tickers
    )
    caps = {t: 1_500_000.0 for t in tickers}

    out = orchestrator.run(
        tickers=tickers,
        baseline_returns=baseline_returns,
        cov_matrix=cov_matrix,
        current_market_caps=caps,
        constraints=turn3.user_portfolio_requirements,
        auto_confirm_views=True,
    )

    print(f"Selected Solver : {out['result'].algorithm}")
    print(f"Active Holdings : {out['result'].active_stock_count} (Requested: 4)")
    print("\nRecommended Allocations:")
    print(out["recommendation_report"][["Ticker", "Weight (%)", "Allocation (INR)"]])
    assert out["result"].algorithm == "MIQP"
    assert out["result"].active_stock_count == 4
    print("\nALL MULTI-TURN EXTRACTION AND INTEGRATION TESTS PASSED.")


if __name__ == "__main__":
    run_multi_turn_test()