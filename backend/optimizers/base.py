"""Base interfaces and standardized output contract for all portfolio optimizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """Standardized output structure returned by all optimizer implementations."""
    algorithm: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    status: str = "optimal"
    solve_time_sec: float = 0.0
    active_stock_count: int = 0
    sector_allocations: Dict[str, float] = field(default_factory=dict)
    applied_constraints: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizerAgent:
    """Base interface for all optimizer agents."""
    name: str = "BaseOptimizer"

    @staticmethod
    def _sharpe(
        mu: pd.Series, weights: Dict[str, float], cov: pd.DataFrame, risk_free_rate: float = 0.06
    ) -> tuple[float, float, float]:
        w = np.array([weights.get(t, 0.0) for t in mu.index])
        exp_ret = float(w @ mu.values)
        exp_vol = float(np.sqrt(max(1e-8, w @ cov.values @ w)))
        sharpe = float((exp_ret - risk_free_rate) / exp_vol) if exp_vol > 0 else 0.0
        return exp_ret, exp_vol, sharpe