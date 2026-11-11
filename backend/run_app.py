"""
Full end-to-end interactive terminal application.
Connects Natural Language Chat Extraction -> Constraint Validation -> 
Data Fetcher (Nifty 50) -> Gemini Sentiment News Scrape -> Black-Litterman ->
Constraint-Aware Optimizer Solver -> Final Recommendation Report.
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from agents.constraint_agent import ConstraintExtractionAgent
from data.data_fetcher import load_nifty50_data
from agents.orchestrator import PortfolioOrchestrator


def main():
    print("=" * 80)
    print("         AUTONOMOUS NIFTY 50 QUANTITATIVE PORTFOLIO PLATFORM          ")
    print("=" * 80)
    print("I am your AI quantitative assistant. Tell me about your investment goals,")
    print("capital, risk tolerance, stock preferences, or cardinality rules.")
    print("Type 'exit' to quit.\n")

    agent = ConstraintExtractionAgent()
    orchestrator = PortfolioOrchestrator()
    session_state = {}

    # Stage 1: Conversational Constraint Gathering
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Session terminated.")
            sys.exit(0)

        response = agent.process_chat_message(
            user_message=user_input,
            current_state=session_state,
        )
        session_state = response.extracted_constraints

        if response.status == "collecting":
            print(f"\nAssistant: {response.follow_up_question}\n")
            if response.error_message:
                print(f"[Validation Warning]: {response.error_message}\n")
        elif response.status == "ready":
            print("\nAssistant: All required constraints have been collected and verified!")
            print(f"Extracted Specifications: {session_state}")
            user_requirements = response.user_portfolio_requirements
            break
        elif response.status == "error":
            print(f"\nAssistant Error: {response.error_message}")
            sys.exit(1)

    # Stage 2: Ingest Market Data via unified data_fetcher
    print("\n" + "-" * 80)
    print(">> Step 1/4: Loading Nifty 50 data, covariance, and market caps via data_fetcher...")
    baseline_returns, cov_matrix, market_caps, sector_mapping, price_source = load_nifty50_data(years=3, limit=10)
    universe = list(baseline_returns.index)
    market_caps_dict = market_caps.to_dict() if market_caps is not None else {t: 1_000_000.0 for t in universe}
    print(f">> Loaded {len(universe)} assets from source '{price_source}': {', '.join(universe)}")

    # Stage 3: Execute Optimization Pipeline
    print(">> Step 2/4: Triggering Gemini Sentiment Agent with news & market cap analysis...")
    print(">> Step 3/4: Calculating shared Black-Litterman expected return posterior...")
    print(">> Step 4/4: Routing to constraint-appropriate mathematical solver...")
    print("-" * 80)

    try:
        pipeline_output = orchestrator.run(
            tickers=universe,
            baseline_returns=baseline_returns,
            cov_matrix=cov_matrix,
            current_market_caps=market_caps_dict,
            constraints=user_requirements,
            sector_mapping=sector_mapping,
            auto_confirm_views=False,  # Triggers human-in-the-loop review prompt
        )
    except Exception as exc:
        print(f"\n[Execution Error]: {exc}")
        sys.exit(1)

    # Stage 4: Display Results & Recommendation Report
    result = pipeline_output["result"]
    report_df = pipeline_output["recommendation_report"]

    print("\n" + "=" * 80)
    print("                       PORTFOLIO EXECUTION SUMMARY                    ")
    print("=" * 80)
    print(f"Algorithm Selected  : {result.algorithm}")
    print(f"Optimization Status : {result.status}")
    print(f"Expected Return     : {result.expected_return * 100:.2f}%")
    print(f"Expected Volatility : {result.expected_volatility * 100:.2f}%")
    print(f"Sharpe Ratio        : {result.sharpe_ratio:.2f}")
    print(f"Active Stock Count  : {result.active_stock_count} stocks")
    print(f"Execution Time      : {result.solve_time_sec:.4f} seconds")

    print("\n" + "=" * 80)
    print("                   FINAL ALLOCATION & RECOMMENDATIONS                 ")
    print("=" * 80)
    print(report_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("                        AUDIT LOG VERIFICATION                       ")
    print("=" * 80)
    for log in pipeline_output["audit_trail"]:
        print(f"  [AUDIT] {log}")
    print("=" * 80)


if __name__ == "__main__":
    main()