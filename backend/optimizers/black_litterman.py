import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

from .base import BaseOptimizerAgent, OptimizationResult


class BlackLittermanOptimizerAgent(BaseOptimizerAgent):
    """Black-Litterman: blends a market-implied equilibrium prior with 'views'
    from any source (such as calibrated LLM news/sentiment views).

    Two entry points:
    - `compute_posterior()` returns (posterior_mu, posterior_cov, diagnostics)
      directly for downstream optimizers (MVO, ADMM, Proximal Gradient) to
      consume as the single shared expected-return vector.
    - `optimize()` calls compute_posterior() and solves for optimal weights
      directly (incorporating any dynamic asset constraints from the LLM).
    """
    name = "Black-Litterman"

    def compute_posterior(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        constraints,
        market_caps: pd.Series = None,
        views: dict = None,
        view_confidences: dict = None,
    ) -> tuple[pd.Series, pd.DataFrame, dict]:
        """Computes the Black-Litterman posterior expected return and covariance.

        Returns:
            Tuple of (posterior_mu, posterior_cov, diagnostics).
        """
        # 1. Market-implied equilibrium prior vs. fallback to empirical mu
        if market_caps is not None and not market_caps.empty:
            risk_aversion = getattr(constraints, "risk_aversion", 2.5)
            pi = market_implied_prior_returns(market_caps, risk_aversion, cov)
        else:
            pi = mu

        viewdict = views or {}
        view_confs = view_confidences or {}

        # 2. Blend views with Idzorek confidence scaling
        if viewdict:
            default_conf = getattr(constraints, "bl_confidence", 0.5)
            confidences = [
                float(max(0.01, min(1.0, view_confs.get(t, default_conf))))
                for t in viewdict
            ]
            bl = BlackLittermanModel(
                cov,
                pi=pi,
                absolute_views=viewdict,
                omega="idzorek",
                view_confidences=confidences,
            )
            posterior_mu = bl.bl_returns()
            posterior_cov = bl.bl_cov()
        else:
            posterior_mu = pi
            posterior_cov = cov

        diagnostics = {
            "n_views": len(viewdict),
            "used_market_prior": market_caps is not None and not market_caps.empty,
            "views_applied": viewdict,
            "confidences_applied": view_confs,
        }
        return posterior_mu, posterior_cov, diagnostics

    def optimize(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        constraints,
        risk_free_rate: float = 0.06,
        market_caps: pd.Series = None,
        sentiment_views: dict = None,
        view_confidences: dict = None,
        custom_constraints: list[dict] = None,
    ) -> OptimizationResult:
        """Solves for portfolio weights using BL posterior returns and LLM constraints."""
        posterior_mu, posterior_cov, diagnostics = self.compute_posterior(
            mu=mu,
            cov=cov,
            constraints=constraints,
            market_caps=market_caps,
            views=sentiment_views,
            view_confidences=view_confidences,
        )

        min_w = getattr(constraints, "min_weight", 0.0)
        max_w = getattr(constraints, "max_weight", 0.35)
        risk_aversion = getattr(constraints, "risk_aversion", 2.5)

        ef = EfficientFrontier(posterior_mu, posterior_cov, weight_bounds=(min_w, max_w))

        # Apply custom asset-level upper-bound constraints extracted by the LLM
        if custom_constraints:
            applied_constraints = []
            for item in custom_constraints:
                ticker = item.get("ticker")
                limit = item.get("max_weight")
                if ticker and ticker in posterior_mu.index and limit is not None:
                    ef.set_weights({ticker: (min_w, float(limit))})
                    applied_constraints.append(f"{ticker} <= {float(limit):.2f}")
            if applied_constraints:
                diagnostics["applied_llm_constraints"] = applied_constraints

        ef.max_quadratic_utility(risk_aversion=risk_aversion)
        clean = ef.clean_weights()
        weights = {t: round(max(0.0, w), 6) for t, w in clean.items()}

        exp_ret, exp_vol, sharpe = self._sharpe(posterior_mu, weights, posterior_cov, risk_free_rate)
        return OptimizationResult(
            algorithm=self.name,
            weights=weights,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            diagnostics=diagnostics,
        )