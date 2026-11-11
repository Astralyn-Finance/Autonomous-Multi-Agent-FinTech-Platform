"""backend/schemas/constraint_schema.py

Pydantic validation schemas for conversational constraint extraction.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.optimization.optimizer_selector import UserPortfolioRequirements


class ExtractedConstraintsModel(BaseModel):
    """Pydantic schema validating and normalizing conversational constraints."""

    investment_amount: Optional[float] = Field(
        None, description="Total portfolio capital/investment in INR or currency."
    )
    target_risk_aversion: Optional[float] = Field(
        None, description="Risk aversion parameter (1.0 = aggressive, 2.5 = moderate, 5.0+ = conservative)."
    )
    exact_k_stocks: Optional[int] = Field(
        None, description="Exact number of stocks to pick if the user specified an exact count."
    )
    max_k_stocks: Optional[int] = Field(
        None, description="Maximum number of stocks allowed in the portfolio."
    )
    min_weight: Optional[float] = Field(
        0.0, description="Minimum weight per stock (default 0.0)."
    )
    max_weight: Optional[float] = Field(
        0.35, description="Maximum allocation limit for any single stock (default 0.35)."
    )
    min_allocation_for_selected: Optional[float] = Field(
        None, description="Minimum buy-in percentage for any chosen asset (semicontinuous limit)."
    )
    l1_sparsity_gamma: Optional[float] = Field(
        None, description="Penalty parameter for soft L1 sparsity."
    )
    l1_turnover_penalty: Optional[float] = Field(
        None, description="Nonsmooth L1 penalty on turnover against existing portfolio."
    )
    sector_limits: Optional[Dict[str, float]] = Field(
        default_factory=dict, description="Upper allocation limits per sector (e.g. {'Technology': 0.30})."
    )
    decompose_by_sector: bool = Field(
        False, description="Flag indicating block-separable or consensus sector allocation."
    )
    max_volatility: Optional[float] = Field(
        None, description="Maximum portfolio volatility limit."
    )
    include_tickers: List[str] = Field(
        default_factory=list, description="List of tickers the user explicitly wants to include."
    )
    exclude_tickers: List[str] = Field(
        default_factory=list, description="List of tickers the user explicitly wants to exclude."
    )

    @field_validator("investment_amount")
    @classmethod
    def validate_capital(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("Investment amount must be strictly positive.")
        return v

    @field_validator("target_risk_aversion")
    @classmethod
    def validate_risk(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("Risk aversion must be greater than 0.")
        return v

    @field_validator("exact_k_stocks", "max_k_stocks")
    @classmethod
    def validate_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("Stock count must be a positive integer.")
        return v

    @model_validator(mode="after")
    def validate_budget_and_cardinality(self) -> ExtractedConstraintsModel:
        if self.min_weight is not None and self.max_weight is not None:
            if self.min_weight > self.max_weight:
                raise ValueError(f"min_weight ({self.min_weight}) cannot exceed max_weight ({self.max_weight}).")

        if self.exact_k_stocks is not None and self.max_weight is not None:
            if self.exact_k_stocks * self.max_weight < 1.0:
                raise ValueError(
                    f"Infeasible: {self.exact_k_stocks} stocks * {self.max_weight:.2f} max_weight = "
                    f"{self.exact_k_stocks * self.max_weight:.2f} < 1.0. Cannot satisfy 100% allocation budget."
                )

        if self.exact_k_stocks is not None and self.min_allocation_for_selected is not None:
            if self.exact_k_stocks * self.min_allocation_for_selected > 1.0:
                raise ValueError(
                    f"Infeasible: {self.exact_k_stocks} stocks * {self.min_allocation_for_selected:.2f} min_buy = "
                    f"{self.exact_k_stocks * self.min_allocation_for_selected:.2f} > 1.0."
                )

        overlap = set(self.include_tickers).intersection(set(self.exclude_tickers))
        if overlap:
            raise ValueError(f"Conflicting tickers in both inclusion and exclusion: {list(overlap)}")

        return self

    def to_user_portfolio_requirements(self) -> UserPortfolioRequirements:
        """Converts validated Pydantic model directly into internal UserPortfolioRequirements."""
        return UserPortfolioRequirements(
            target_risk_aversion=self.target_risk_aversion if self.target_risk_aversion is not None else 2.5,
            min_weight=self.min_weight if self.min_weight is not None else 0.0,
            max_weight=self.max_weight if self.max_weight is not None else 0.35,
            exact_k_stocks=self.exact_k_stocks,
            max_k_stocks=self.max_k_stocks,
            min_allocation_for_selected=self.min_allocation_for_selected,
            l1_sparsity_gamma=self.l1_sparsity_gamma,
            l1_turnover_penalty=self.l1_turnover_penalty,
            sector_limits=self.sector_limits or None,
            decompose_by_sector=self.decompose_by_sector,
            max_volatility=self.max_volatility,
            include_tickers=self.include_tickers,
            exclude_tickers=self.exclude_tickers,
            investment_amount=self.investment_amount if self.investment_amount is not None else 1_000_000.0,
        )


class ConversationExtractionResponse(BaseModel):
    """The clean contract returned to the frontend chat interface."""

    status: Literal["collecting", "ready", "error"]
    follow_up_question: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    extracted_constraints: Dict[str, Optional[object]] = Field(default_factory=dict)
    user_portfolio_requirements: Optional[UserPortfolioRequirements] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True