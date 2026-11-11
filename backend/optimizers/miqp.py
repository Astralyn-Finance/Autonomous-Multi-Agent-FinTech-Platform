"""Mixed-Integer Quadratic Programming (MIQP) Optimizer."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import cvxpy as cp
import numpy as np
import pandas as pd

from .base import BaseOptimizerAgent, OptimizationResult
from backend.optimization.optimizer_selector import UserPortfolioRequirements


class MIQPOptimizer(BaseOptimizerAgent):
    name = "MIQP"

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
        z = cp.Variable(n, boolean=True)

        risk = cp.quad_form(w, sigma_bl.values)
        ret = mu_bl.values @ w
        obj = cp.Minimize(0.5 * req.target_risk_aversion * risk - ret)

        cons = [cp.sum(w) == 1.0, w >= 0.0]

        # Semicontinuous coupling: min_hold * z_i <= w_i <= max_weight * z_i
        min_buy = req.min_allocation_for_selected or req.min_weight or 0.01
        for i in range(n):
            cons.append(w[i] >= min_buy * z[i])
            cons.append(w[i] <= req.max_weight * z[i])

        if req.exact_k_stocks is not None:
            cons.append(cp.sum(z) == req.exact_k_stocks)
        elif req.max_k_stocks is not None:
            cons.append(cp.sum(z) <= req.max_k_stocks)

        # Sector limits if defined
        if req.sector_limits and sector_mapping:
            for sector, max_alloc in req.sector_limits.items():
                sec_indices = [i for i, t in enumerate(tickers) if sector_mapping.get(t) == sector]
                if sec_indices:
                    cons.append(cp.sum(w[sec_indices]) <= max_alloc)

        # Apply custom LLM weight caps
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
        try:
            prob.solve(solver=cp.SCIP, verbose=False)
        except Exception:
            prob.solve(verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            # Deterministic fallback
            ratios = mu_bl.values / np.diag(sigma_bl.values)
            top_k = req.exact_k_stocks or req.max_k_stocks or 5
            top_indices = np.argsort(ratios)[-top_k:]
            weights_arr = np.zeros(n)
            weights_arr[top_indices] = 1.0 / len(top_indices)
            status = "fallback_heuristic"
        else:
            weights_arr = np.array(w.value).flatten()
            weights_arr = np.where(weights_arr < 1e-4, 0.0, weights_arr)
            weights_arr /= np.sum(weights_arr)
            status = prob.status

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
            applied_constraints=applied_constraints + [
                f"Cardinality target: {req.exact_k_stocks or req.max_k_stocks}",
                f"Min buy-in: {min_buy:.3f}",
            ],
            diagnostics={"prob_status": prob.status},
        )