"""Mean-Variance Quadratic Programming (MVO-QP) Optimizer."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import cvxpy as cp
import numpy as np
import pandas as pd

from .base import BaseOptimizerAgent, OptimizationResult
from backend.optimization.optimizer_selector import UserPortfolioRequirements


class MVOQPOptimizer(BaseOptimizerAgent):
    name = "MVO_QP"

    def optimize(
        self,
        mu_bl: pd.Series,
        sigma_bl: pd.DataFrame,
        req: UserPortfolioRequirements,
        risk_free_rate: float = 0.06,
        sector_mapping: Optional[Dict[str, str]] = None,
        custom_constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        start_t = time.time()
        tickers = list(mu_bl.index)
        n = len(tickers)

        w = cp.Variable(n)
        risk = cp.quad_form(w, sigma_bl.values)
        ret = mu_bl.values @ w
        obj = cp.Minimize(0.5 * req.target_risk_aversion * risk - ret)

        cons = [
            cp.sum(w) == 1.0,
            w >= req.min_weight,
            w <= req.max_weight,
        ]

        for i, t in enumerate(tickers):
            if t in req.exclude_tickers:
                cons.append(w[i] == 0.0)

        if req.sector_limits and sector_mapping:
            for sector, max_alloc in req.sector_limits.items():
                sec_indices = [i for i, t in enumerate(tickers) if sector_mapping.get(t) == sector]
                if sec_indices:
                    cons.append(cp.sum(w[sec_indices]) <= max_alloc)

        applied_constraints = []
        if custom_constraints:
            for c in custom_constraints:
                t = c.get("ticker")
                lim = c.get("max_weight")
                if t in tickers and lim is not None:
                    idx = tickers.index(t)
                    cons.append(w[idx] <= float(lim))
                    applied_constraints.append(f"{t} <= {float(lim):.2f}")

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            weights_arr = np.array(w.value).flatten()
            weights_arr = np.clip(weights_arr, 0.0, 1.0)
            weights_arr /= np.sum(weights_arr)
            status = prob.status
        else:
            weights_arr = np.ones(n) / n
            status = "fallback_equal_weight"

        weights_dict = {t: round(float(weights_arr[i]), 6) for i, t in enumerate(tickers)}
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
            status=status,
            solve_time_sec=time.time() - start_t,
            active_stock_count=int(sum(1 for v in weights_dict.values() if v > 1e-4)),
            sector_allocations=sector_alloc,
            applied_constraints=applied_constraints,
            diagnostics={"objective_value": float(prob.value) if prob.value is not None else 0.0},
        )