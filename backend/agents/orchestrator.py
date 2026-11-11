"""Portfolio Orchestrator: Constraint-Aware Pipeline with Single BL Posterior."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from agents.sentiment_pipeline import build_sentiment_views
from optimizers.admm_sparse import ADMMOptimizer
from optimizers.base import OptimizationResult
from optimizers.black_litterman import BlackLittermanOptimizerAgent
from optimizers.miqp import MIQPOptimizer
from optimizers.mvo_qp import MVOQPOptimizer
from optimizers.proximal_gradient import ProximalGradientOptimizer
from optimization.feasibility_checker import FeasibilityChecker
from optimization.optimizer_selector import (
    OptimizerRecommender,
    UserPortfolioRequirements,
)
from optimization.constraint_validator import IndependentPortfolioValidator, ValidationReport

logger = logging.getLogger(__name__)


class PortfolioOrchestrator:
    """Coordinates the constraint-driven multi-agent optimization pipeline."""

    def __init__(self):
        # 1. Shared BL Posterior Engine
        self.bl_agent = BlackLittermanOptimizerAgent()

        # 2. Solver pool mapped to OptimizerRegistry keys
        self.solvers = {
            "MVO_QP": MVOQPOptimizer(),
            "PROXIMAL_GRADIENT": ProximalGradientOptimizer(),
            "ADMM": ADMMOptimizer(),
            "MIQP": MIQPOptimizer(),
        }

    def run(
        self,
        tickers: List[str],
        baseline_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        current_market_caps: Dict[str, float],
        retrieved_market_caps: Optional[Dict[str, float]] = None,
        constraints: Optional[UserPortfolioRequirements] = None,
        company_names: Optional[Dict[str, str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        auto_confirm_views: bool = False,
        risk_free_rate: float = 0.06,
    ) -> Dict[str, Any]:
        """Executes the end-to-end portfolio construction pipeline.

        Returns:
            Dictionary containing final result, validation report,
            tabular recommendation report, audit trail, and BL posteriors.
        """
        
        audit_trail: List[str] = []
        user_constraints = constraints or UserPortfolioRequirements()
        audit_trail.append(f"Initialized pipeline for universe of {len(tickers)} assets.")

        # Step 1: Pre-optimization Feasibility Check
        audit_trail.append("Validating constraint feasibility against universe boundaries.")
        OptimizerRecommender.select_optimizer(user_constraints, universe_size=len(tickers))
        audit_trail.append("Constraint feasibility check passed.")

        # Step 2: Gemini Sentiment Agent + User Calibration
        audit_trail.append("Gathering live news and evaluating market-cap shifts with GeminiSentimentAgent.")
        views, confidences, custom_constraints, diagnostics = build_sentiment_views(
            tickers=tickers,
            baseline_returns=baseline_returns,
            current_market_caps=current_market_caps,
            retrieved_market_caps=retrieved_market_caps,
            company_names=company_names,
            auto_confirm=auto_confirm_views,
        )
        audit_trail.append(
            f"Generated {len(views)} sentiment views, {len(confidences)} confidences, and "
            f"{len(custom_constraints)} LLM constraint limits."
        )

        # Step 3: Compute Unified Black-Litterman Posterior
        audit_trail.append("Invoking BlackLittermanOptimizerAgent.compute_posterior() for shared returns.")
        market_caps_series = pd.Series(current_market_caps).reindex(tickers).dropna()

        posterior_mu, posterior_cov, bl_diagnostics = self.bl_agent.compute_posterior(
            mu=baseline_returns.reindex(tickers),
            cov=cov_matrix.loc[tickers, tickers],
            constraints=user_constraints,
            market_caps=market_caps_series if not market_caps_series.empty else None,
            views=views,
            view_confidences=confidences,
        )
        audit_trail.append(
            f"BL Posterior calculated: {bl_diagnostics.get('n_views', 0)} views applied, "
            f"used_market_prior={bl_diagnostics.get('used_market_prior', False)}."
        )

        # Step 4: Constraint-Aware Optimizer Selection
        selected_opt_name, selection_justification = OptimizerRecommender.select_optimizer(user_constraints)
        audit_trail.append(f"Optimizer Selected: {selected_opt_name}. Reason: {selection_justification}")

        # Step 5: Direct Execution by Selected Optimizer
        solver = self.solvers.get(selected_opt_name)
        if solver is None:
            raise ValueError(f"Optimizer '{selected_opt_name}' registered but not instantiated in solver pool.")

        optimization_result: OptimizationResult = solver.optimize(
            mu_bl=posterior_mu,
            sigma_bl=posterior_cov,
            req=user_constraints,
            risk_free_rate=risk_free_rate,
            sector_mapping=sector_mapping,
            custom_constraints=custom_constraints,
        )
        audit_trail.append(
            f"Solver {selected_opt_name} completed in {optimization_result.solve_time_sec:.4f}s "
            f"with status '{optimization_result.status}'."
        )

        # Step 6: Independent Hard-Constraint Validation
        val_report: ValidationReport = IndependentPortfolioValidator.audit(
            result=optimization_result,
            req=user_constraints,
            sector_mapping=sector_mapping,
        )
        audit_trail.extend(val_report.audit_log)

        if not val_report.is_valid:
            error_details = "; ".join(val_report.violations)
            audit_trail.append(f"Validation FAILED: {error_details}")
            raise RuntimeError(f"Independent validation failed on hard constraints: {error_details}")

        # Step 7: Explainable Recommendation Breakdown
        recommendation_df = self._build_recommendation_report(
            result=optimization_result,
            req=user_constraints,
            diagnostics=diagnostics,
            sector_mapping=sector_mapping,
        )

        return {
            "result": optimization_result,
            "validation_report": val_report,
            "recommendation_report": recommendation_df,
            "audit_trail": audit_trail,
            "posterior_mu": posterior_mu,
            "posterior_cov": posterior_cov,
            "bl_diagnostics": bl_diagnostics,
        }

    def _build_recommendation_report(
        self,
        result: OptimizationResult,
        req: UserPortfolioRequirements,
        diagnostics: Dict[str, Any],
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Constructs an explainable tabular allocation view with rationales."""
        records = []
        total_investment = getattr(req, "investment_amount", 1_000_000.0)

        for ticker, weight in result.weights.items():
            if weight > 1e-4:
                diag = diagnostics.get(ticker, {})
                records.append({
                    "Ticker": ticker,
                    "Sector": sector_mapping.get(ticker, "General") if sector_mapping else "N/A",
                    "Weight (%)": round(weight * 100, 2),
                    "Allocation (INR)": round(weight * total_investment, 2),
                    "Sentiment Rationale": diag.get("rationale", "Baseline Equilibrium Allocation"),
                    "Risk Constraint Flag": diag.get("constraint_note") or "None",
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)
        return df