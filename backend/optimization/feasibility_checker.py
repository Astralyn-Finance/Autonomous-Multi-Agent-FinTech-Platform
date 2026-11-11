"""Optimizer Feasibility Checker."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from backend.optimization.optimizer_selector import UserPortfolioRequirements

class FeasibilityChecker:
    """Audits user constraints for mathematical feasibility prior to optimization."""

    @staticmethod
    def validate(req: UserPortfolioRequirements, universe_tickers: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        n_assets = len(universe_tickers)

        if req.min_weight < 0.0:
            errors.append("min_weight must be non-negative (long-only).")
        if req.max_weight <= 0.0 or req.max_weight > 1.0:
            errors.append("max_weight must be in (0.0, 1.0].")
        if req.min_weight > req.max_weight:
            errors.append(f"Infeasible bounds: min_weight ({req.min_weight}) > max_weight ({req.max_weight}).")

        if n_assets * req.max_weight < 1.0:
            errors.append(
                f"Infeasible diversification: {n_assets} assets with max_weight {req.max_weight} sum to "
                f"{n_assets * req.max_weight:.2f} < 1.0 (cannot satisfy 100% budget)."
            )

        if req.exact_k_stocks is not None:
            if req.exact_k_stocks > n_assets:
                errors.append(f"exact_k_stocks ({req.exact_k_stocks}) exceeds universe size ({n_assets}).")
            if req.exact_k_stocks * req.max_weight < 1.0:
                errors.append(
                    f"Infeasible cardinality: {req.exact_k_stocks} stocks * {req.max_weight} max_weight = "
                    f"{req.exact_k_stocks * req.max_weight:.2f} < 1.0."
                )
            if req.min_allocation_for_selected and req.exact_k_stocks * req.min_allocation_for_selected > 1.0:
                errors.append(
                    f"Infeasible buy-in threshold: {req.exact_k_stocks} stocks * {req.min_allocation_for_selected} = "
                    f"{req.exact_k_stocks * req.min_allocation_for_selected:.2f} > 1.0."
                )

        overlap = set(req.include_tickers).intersection(set(req.exclude_tickers))
        if overlap:
            errors.append(f"Conflicting inclusion/exclusion for tickers: {list(overlap)}")

        return len(errors) == 0, errors