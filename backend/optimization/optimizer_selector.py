"""Optimizer Capability Registry and Constraint Recommender."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

class ProblemCategory(Enum):
    STANDARD_CONVEX = auto()          # Continuous, convex quadratic / linear (MVO-QP)
    NONSMOOTH_REGULARIZED = auto()    # Smooth + nonsmooth L1 turnover / sparsity (PG)
    DECOMPOSABLE_STRUCTURED = auto()  # Block-separable, consensus, sector-split (ADMM)
    DISCRETE_CARDINALITY = auto()     # Exact K, semi-continuous min-holdings, binary (MIQP)

@dataclass
class UserPortfolioRequirements:
    """User-specified investment constraints, cardinality rules, and limits."""
    target_risk_aversion: float = 2.5
    min_weight: float = 0.0
    max_weight: float = 0.35
    exact_k_stocks: Optional[int] = None
    max_k_stocks: Optional[int] = None
    min_allocation_for_selected: Optional[float] = None
    l1_sparsity_gamma: Optional[float] = None
    l1_turnover_penalty: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    decompose_by_sector: bool = False
    max_volatility: Optional[float] = None
    include_tickers: List[str] = field(default_factory=list)
    exclude_tickers: List[str] = field(default_factory=list)
    investment_amount: float = 1_000_000.0

@dataclass
class OptimizerCapability:
    name: str
    category: ProblemCategory
    supports_exact_cardinality: bool
    supports_l1_regularization: bool
    supports_sector_decomposition: bool
    supports_semicontinuous_limits: bool
    description: str


OPTIMIZER_REGISTRY: Dict[str, OptimizerCapability] = {
    "MVO_QP": OptimizerCapability(
        name="MVO_QP",
        category=ProblemCategory.STANDARD_CONVEX,
        supports_exact_cardinality=False,
        supports_l1_regularization=False,
        supports_sector_decomposition=False,
        supports_semicontinuous_limits=False,
        description="Standard convex quadratic programming for continuous budget, bounds, and sector limits."
    ),
    "PROXIMAL_GRADIENT": OptimizerCapability(
        name="PROXIMAL_GRADIENT",
        category=ProblemCategory.NONSMOOTH_REGULARIZED,
        supports_exact_cardinality=False,
        supports_l1_regularization=True,
        supports_sector_decomposition=False,
        supports_semicontinuous_limits=False,
        description="First-order proximal splitting for L1-sparse penalties and nonsmooth turnover regularization."
    ),
    "ADMM": OptimizerCapability(
        name="ADMM",
        category=ProblemCategory.DECOMPOSABLE_STRUCTURED,
        supports_exact_cardinality=False,
        supports_l1_regularization=True,
        supports_sector_decomposition=True,
        supports_semicontinuous_limits=False,
        description="Alternating Direction Method of Multipliers for block-separable constraints and sector consensus."
    ),
    "MIQP": OptimizerCapability(
        name="MIQP",
        category=ProblemCategory.DISCRETE_CARDINALITY,
        supports_exact_cardinality=True,
        supports_l1_regularization=False,
        supports_sector_decomposition=True,
        supports_semicontinuous_limits=True,
        description="Mixed-Integer Quadratic Programming for exact K-cardinality and semi-continuous holding thresholds."
    ),
}

class OptimizerRecommender:
    """Selects the single mathematically appropriate optimizer based on constraints."""

    @staticmethod
    def select_optimizer(req: UserPortfolioRequirements, universe_size: int = None) -> Tuple[str, str]:
        # 1. Discrete decisions, exact cardinality, or semi-continuous thresholds
        if (
            req.exact_k_stocks is not None
            or req.max_k_stocks is not None
            or req.min_allocation_for_selected is not None
        ):
            return (
                "MIQP",
                "Selected MIQP: exact/max cardinality or semi-continuous buy-in thresholds require binary decision variables."
            )

        # 2. Block-separable constraints or sector consensus
        if req.decompose_by_sector:
            return (
                "ADMM",
                "Selected ADMM: problem structure has sector-wise block decomposition or consensus constraints."
            )

        # 3. Nonsmooth L1 penalties or turnover regularization
        if req.l1_sparsity_gamma or req.l1_turnover_penalty:
            return (
                "PROXIMAL_GRADIENT",
                "Selected Proximal Gradient: objective contains nonsmooth L1 penalties handled via proximal soft-thresholding."
            )

        # 4. Standard continuous convex constraints
        return (
            "MVO_QP",
            "Selected MVO-QP: standard continuous convex objective with budget, box limits, and sector limits."
        )