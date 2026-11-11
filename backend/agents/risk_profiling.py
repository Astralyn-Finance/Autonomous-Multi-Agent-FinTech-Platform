"""
Risk Profiling Agent

This is the personalization layer of the platform. It is the ONLY place where
investor-facing language (risk tolerance, objective, time horizon, preferences)
gets converted into the numeric constraints the four deterministic optimizer
agents consume.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentObjective(str, Enum):
    CAPITAL_PRESERVATION = "capital_preservation"
    INCOME = "income"
    BALANCED_GROWTH = "balanced_growth"
    AGGRESSIVE_GROWTH = "aggressive_growth"


class TimeHorizon(str, Enum):
    SHORT = "short"    # < 3 years
    MEDIUM = "medium"  # 3-7 years
    LONG = "long"       # 7+ years


@dataclass
class UserProfile:
    """What the user actually sets in the UI"""
    risk_tolerance: RiskTolerance
    objective: InvestmentObjective
    time_horizon: TimeHorizon
    max_single_position: float = 0.25       # "don't let any one stock exceed X% of my portfolio"
    min_position: float = 0.0
    esg_exclude: Optional[List[str]] = None  # tickers/sectors the user wants excluded
    sector_caps: Optional[Dict[str, float]] = None  # {"IT": 0.35, "Financials": 0.40}
    target_num_holdings: Optional[int] = None        # "I want a focused portfolio of ~N stocks"


@dataclass
class OptimizationConstraints:
    """The deterministic numeric object every optimizer agent actually consumes"""
    risk_aversion: float              # higher = more risk-averse
    max_volatility: Optional[float]   # annualized volatility ceiling or None
    target_return: Optional[float]    # annualized return target or None
    max_weight: float
    min_weight: float
    l1_penalty: float                 # starting guess for sparsity strength
    target_num_holdings: Optional[int]  # desired portfolio size
    bl_confidence: float              # 0-1, how strongly sentiment "views" move Black-Litterman
    sector_caps: Dict[str, float]
    excluded_assets: List[str]


class RiskProfilingAgent:
    """Translates a UserProfile into OptimizationConstraints."""

    # Higher risk_aversion  => optimizer favors lower-variance portfolios
    _RISK_AVERSION = {
        RiskTolerance.CONSERVATIVE: 6.0,
        RiskTolerance.MODERATE: 3.0,
        RiskTolerance.AGGRESSIVE: 1.2,
    }
    _MAX_VOL = {
        RiskTolerance.CONSERVATIVE: 0.10,
        RiskTolerance.MODERATE: 0.18,
        RiskTolerance.AGGRESSIVE: 0.30,
    }
    _HORIZON_ADJ = {
        TimeHorizon.SHORT: 1.3,   # short horizon -> can tolerate less vol -> effectively more risk-averse
        TimeHorizon.MEDIUM: 1.0,
        TimeHorizon.LONG: 0.75,   # long horizon -> can ride out drawdowns -> effectively less risk-averse
    }
    _BL_CONFIDENCE = {
        RiskTolerance.CONSERVATIVE: 0.25,
        RiskTolerance.MODERATE: 0.50,
        RiskTolerance.AGGRESSIVE: 0.75,
    }

    def build_constraints(self, profile: UserProfile) -> OptimizationConstraints:
        risk_aversion = self._RISK_AVERSION[profile.risk_tolerance] * self._HORIZON_ADJ[profile.time_horizon]
        max_vol = self._MAX_VOL[profile.risk_tolerance]

        if profile.objective == InvestmentObjective.INCOME:
            max_vol *= 0.8          # income investors: tighten the risk budget
        elif profile.objective == InvestmentObjective.AGGRESSIVE_GROWTH:
            max_vol *= 1.15         # growth investors: loosen it slightly
        elif profile.objective == InvestmentObjective.CAPITAL_PRESERVATION:
            max_vol *= 0.6
            risk_aversion *= 1.5

        # This is only a *starting guess* for the sparsity penalty
        l1_penalty = 0.15 / risk_aversion

        return OptimizationConstraints(
            risk_aversion=risk_aversion,
            max_volatility=max_vol,
            target_return=None,
            max_weight=profile.max_single_position,
            min_weight=profile.min_position,
            l1_penalty=l1_penalty,
            target_num_holdings=profile.target_num_holdings,
            bl_confidence=self._BL_CONFIDENCE[profile.risk_tolerance],
            sector_caps=profile.sector_caps or {},
            excluded_assets=profile.esg_exclude or [],
        )
