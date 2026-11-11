"""Proximal Gradient Optimizer for regularized portfolio problems."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseOptimizerAgent, OptimizationResult
from optimization.optimizer_selector import UserPortfolioRequirements


class ProximalGradientOptimizer(BaseOptimizerAgent):
    name = "PROXIMAL_GRADIENT"

    def optimize(
        self,
        mu_bl: pd.Series,
        sigma_bl: pd.DataFrame,
        req: UserPortfolioRequirements,
        risk_free_rate: float = 0.06,
        max_iter: int = 500,
        lr: float = 0.01,
        **kwargs,
    ) -> OptimizationResult:
        start_t = time.time()
        tickers = list(mu_bl.index)
        n = len(tickers)
        gamma = req.l1_sparsity_gamma or 0.02
        Sigma = sigma_bl.values
        mu = mu_bl.values

        w = np.ones(n) / n
        for _ in range(max_iter):
            grad = req.target_risk_aversion * (Sigma @ w) - mu
            w_step = w - lr * grad
            # Proximal L1 soft-thresholding
            w_soft = np.sign(w_step) * np.maximum(0.0, np.abs(w_step) - (lr * gamma))
            w_proj = np.clip(w_soft, req.min_weight, req.max_weight)
            sum_w = np.sum(w_proj)
            w = (w_proj / sum_w) if sum_w > 0 else np.ones(n) / n

        weights_dict = {t: round(float(w[i]), 6) for i, t in enumerate(tickers)}
        exp_ret, exp_vol, sharpe = self._sharpe(mu_bl, weights_dict, sigma_bl, risk_free_rate)

        return OptimizationResult(
            algorithm=self.name,
            weights=weights_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            status="optimal",
            solve_time_sec=time.time() - start_t,
            active_stock_count=int(sum(1 for v in weights_dict.values() if v > 1e-4)),
            applied_constraints=[f"L1 sparsity penalty gamma={gamma:.4f}"],
            diagnostics={"iterations": max_iter, "gamma": gamma},
        )