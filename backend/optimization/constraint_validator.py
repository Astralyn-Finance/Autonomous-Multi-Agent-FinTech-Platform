"""Independent Portfolio Validation and Hard-Constraint Compliance Checker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from backend.optimizers.base import OptimizationResult
from backend.optimization.optimizer_selector import UserPortfolioRequirements


@dataclass
class ValidationReport:
    is_valid: bool
    violations: List[str]
    audit_log: List[str]


class IndependentPortfolioValidator:
    """Verifies all hard constraints independently without optimizer bias."""

    @staticmethod
    def audit(
        result: OptimizationResult,
        req: UserPortfolioRequirements,
        sector_mapping: Optional[Dict[str, str]] = None,
        tol: float = 1e-4,
    ) -> ValidationReport:
        violations = []
        logs = []
        w_dict = result.weights

        logs.append(f"Initiated validation for optimizer: {result.algorithm}")

        # 1. Budget constraint: sum(weights) == 1.0
        budget = sum(w_dict.values())
        if abs(budget - 1.0) > tol:
            violations.append(f"Budget violated: sum={budget:.6f} != 1.0")
        else:
            logs.append("Budget check passed: sum(weights) == 1.0")

        # 2. Long-only and upper bounds
        for t, val in w_dict.items():
            if val < -tol:
                violations.append(f"Long-only violated: {t} weight={val:.4f} < 0")
            if val > req.max_weight + tol:
                violations.append(f"Max-weight violated: {t} weight={val:.4f} > {req.max_weight:.4f}")

        # 3. Exact cardinality check
        active_stocks = sum(1 for val in w_dict.values() if val > tol)
        if req.exact_k_stocks is not None and active_stocks != req.exact_k_stocks:
            violations.append(
                f"Cardinality violated: target={req.exact_k_stocks}, actual={active_stocks}"
            )
        elif req.max_k_stocks is not None and active_stocks > req.max_k_stocks:
            violations.append(
                f"Max cardinality violated: limit={req.max_k_stocks}, actual={active_stocks}"
            )
        else:
            logs.append(f"Cardinality check passed: {active_stocks} active holdings")

        # 4. Semicontinuous minimum holdings
        if req.min_allocation_for_selected:
            for t, val in w_dict.items():
                if tol < val < (req.min_allocation_for_selected - tol):
                    violations.append(
                        f"Minimum holding violated: {t} weight={val:.4f} < {req.min_allocation_for_selected}"
                    )

        # 5. Sector exposure limits
        if req.sector_limits and sector_mapping:
            sec_sums: Dict[str, float] = {}
            for t, val in w_dict.items():
                sec = sector_mapping.get(t, "Other")
                sec_sums[sec] = sec_sums.get(sec, 0.0) + val
            for sec, lim in req.sector_limits.items():
                actual_sec = sec_sums.get(sec, 0.0)
                if actual_sec > lim + tol:
                    violations.append(f"Sector limit violated: {sec} alloc={actual_sec:.3f} > {lim:.3f}")

        is_valid = len(violations) == 0
        logs.append("Independent validation verdict: " + ("PASSED" if is_valid else "FAILED"))
        return ValidationReport(is_valid=is_valid, violations=violations, audit_log=logs)