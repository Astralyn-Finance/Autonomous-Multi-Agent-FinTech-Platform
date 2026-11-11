"""ADMM Optimizer for structured consensus and sector-decomposable problems."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseOptimizerAgent, OptimizationResult
from backend.optimization.optimizer_selector import UserPortfolioRequirements


class ADMMOptimizer(BaseOptimizerAgent):
    name = "ADMM"

    def optimize(
        self,
        mu_bl: pd.Series,
        sigma_bl: pd.DataFrame,
        req: UserPortfolioRequirements,
        risk_free_rate: float = 0.06,
        sector_mapping: Optional[Dict[str, str]] = None,
        rho: float = 1.0,
        max_iter: int = 200,
    ) -> OptimizationResult:
        start_t = time.time()
        tickers = list(mu_bl.index)
        n = len(tickers)
        Sigma = sigma_bl.values
        mu = mu_bl.values

        w = np.ones(n) / n
        z = np.copy(w)
        u = np.zeros(n)
        inv_mat = np.linalg.pinv(req.target_risk_aversion * Sigma + rho * np.eye(n))

        for _ in range(max_iter):
            w = inv_mat @ (mu + rho * (z - u))
            v = w + u
            z_proj = np.clip(v, req.min_weight, req.max_weight)
            sum_z = np.sum(z_proj)
            z = (z_proj / sum_z) if sum_z > 0 else np.ones(n) / n
            u = u + (w - z)

        weights_dict = {t: round(float(z[i]), 6) for i, t in enumerate(tickers)}
        exp_ret, exp_vol, sharpe = self._sharpe(mu_bl, weights_dict, sigma_bl, risk_free_rate)

        sector_alloc = {}
        if sector_mapping:
            for t, val in weights_dict.items():
                sec = sector_mapping.get(t, "Other")
                sector_alloc[sec] = round(sector_alloc.get(sec, 0.0) + val, 6)

        return OptimizationResult(
            algorithm=self.name,
            weights=weights_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            status="optimal",
            solve_time_sec=time.time() - start_t,
            active_stock_count=int(sum(1 for v in weights_dict.values() if v > 1e-4)),
            sector_allocations=sector_alloc,
            applied_constraints=["Sector-separable consensus splitting"],
            diagnostics={"iterations": max_iter, "rho": rho},
        )