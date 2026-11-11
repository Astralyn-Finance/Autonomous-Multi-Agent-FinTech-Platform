"""
Backtester
============
Replays the pipeline over a historical window as if each rebalance date were
"today". For every rebalance date D in [eval_start, eval_end]:

  1. The pipeline sees only prices up to and including D (a rolling lookback)
     and, if a sentiment provider is configured, only sentiment published
     before the 15:30 IST close on D.
  2. It produces target weights plus its own forecast: expected return and
     volatility of that portfolio.
  3. The weights are held until the next rebalance and scored on what the
     market actually did -- realized return, realized volatility, how each
     holding contributed, and whether the forecast was right.

What gets compared, per rebalance and in aggregate:
  - predicted portfolio return / volatility  vs  realized return / volatility
    of those exact weights (and how many standard deviations apart: forecast_z)
  - the weight distribution (holdings, effective N, max weight) and each
    holding's actual contribution to the period's return
  - the model's per-stock return ranking vs the realized ranking (rank IC),
    for the raw historical mean and for the pipeline's posterior returns
  - overall realized performance vs benchmarks (equal weight, MVO alone, an
    index you supply)

Sentiment: pass a SentimentProvider (see sentiment_provider.py). If a date has
no usable sentiment -- nothing published, no scores for the universe, a
timestamp after the cutoff, a provider error -- that rebalance runs WITHOUT
sentiment and the reason is logged and printed. A missing score is never
turned into a zero. Without a provider, the report states plainly that the
run excluded sentiment.

Limits to keep in mind:
  - Survivorship bias: backtesting today's Nifty 50 over past years only
    includes companies that stayed in the index; results are flattered.
  - No historical market caps, so Black-Litterman uses its fallback prior
    (mu itself); the cap-implied prior is not exercised.
  - Costs are a flat per-unit-traded assumption (cost_rate); no slippage,
    taxes or intraday execution. Trades fill at the close of the rebalance date.
  - Tuning parameters after looking at results overfits to that history; keep
    a hold-out window you only evaluate once.
"""

import argparse
import inspect
import os
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from data.estimation import estimate_mu_cov
from data.validation import validate_ohlcv_frame
from sentiment_provider import SentimentUnavailable, decision_cutoff

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Results and metrics
# --------------------------------------------------------------------------

@dataclass
class StrategyContext:
    """Extra, point-in-time information handed to strategies that ask for it
    (declare a parameter named `context`)."""
    as_of: pd.Timestamp
    sentiment_scores: dict = field(default_factory=dict)
    sentiment_confidences: dict = field(default_factory=dict)
    sentiment_status: str = "no provider configured"


@dataclass
class BacktestResult:
    name: str
    daily_returns: pd.Series            # net of costs
    weights_history: pd.DataFrame       # target weights at each rebalance date
    rebalance_log: pd.DataFrame         # one row per rebalance
    holdings_detail: pd.DataFrame = field(default_factory=pd.DataFrame)  # per holding, per period
    metrics: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.daily_returns).cumprod()

    def _ok_rows(self) -> pd.DataFrame:
        log = self.rebalance_log
        return log[log["error"].isna()] if len(log) else log

    def calibration(self) -> dict:
        """Predicted vs realized. Predictions are what the strategy reported at
        each rebalance; realized numbers come from the prices that followed.
        NaN when the strategy reports no forecast (e.g. equal weight)."""
        ok = self._ok_rows()
        if not len(ok):
            return {}
        pred_ret = ok["predicted_return"].dropna()
        pred_vol = ok["predicted_vol"].dropna()
        z = ok["forecast_z"].dropna()
        out = {
            "avg_predicted_return": float(pred_ret.mean()) if len(pred_ret) else np.nan,
            "realized_cagr": self.metrics.get("cagr", np.nan),
            "avg_predicted_vol": float(pred_vol.mean()) if len(pred_vol) else np.nan,
            "realized_vol": self.metrics.get("ann_volatility", np.nan),
            # forecast_z = (realized - predicted) / predicted sd over the period.
            # Well calibrated: mean near 0, ~68% of periods within +/-1.
            "mean_forecast_z": float(z.mean()) if len(z) else np.nan,
            "pct_within_1sd": float((z.abs() <= 1).mean()) if len(z) else np.nan,
        }
        out["return_gap"] = out["avg_predicted_return"] - out["realized_cagr"]
        both = ok.dropna(subset=["predicted_vol", "realized_vol"])
        out["pct_periods_vol_within_prediction"] = (
            float((both["realized_vol"] <= both["predicted_vol"]).mean()) if len(both) else np.nan)
        for col, key in (("rank_ic_raw_mu", "mean_rank_ic_raw_mu"),
                         ("rank_ic_posterior", "mean_rank_ic_posterior")):
            s = ok[col].dropna()
            out[key] = float(s.mean()) if len(s) else np.nan
        ic = ok["rank_ic_posterior"].dropna()
        out["pct_periods_posterior_ic_positive"] = float((ic > 0).mean()) if len(ic) else np.nan
        return out

    def sentiment_report(self) -> dict:
        """How sentiment retrieval went, by rebalance date."""
        log = self.rebalance_log
        if not len(log) or "sentiment_status" not in log:
            return {"n_rebalances": 0, "n_used": 0, "by_status": {}, "problems": pd.DataFrame()}
        kind = log["sentiment_status"].str.split(":").str[0]
        return {
            "n_rebalances": len(log),
            "n_used": int((kind == "ok").sum()),
            "by_status": kind.value_counts().to_dict(),
            "problems": log.loc[kind != "ok", ["date", "sentiment_status"]],
        }


def compute_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.06) -> dict:
    r = daily_returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ("total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown", "calmar")}
    equity = pd.concat([pd.Series([1.0]), (1 + r.reset_index(drop=True)).cumprod()],
                       ignore_index=True)
    total = float(equity.iloc[-1] - 1)
    years = len(r) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() * TRADING_DAYS - risk_free_rate) / vol) if vol > 1e-12 else np.nan
    max_dd = float((equity / equity.cummax() - 1).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-12 and not np.isnan(cagr) else np.nan
    return {"total_return": total, "cagr": cagr, "ann_volatility": vol,
            "sharpe": sharpe, "max_drawdown": max_dd, "calmar": calmar}


def _rank_ic(predicted: pd.Series, realized: pd.Series) -> float:
    """Spearman rank correlation between predicted per-stock returns and what
    the stocks actually returned. +1 = ranked perfectly, 0 = no skill."""
    if predicted is None:
        return np.nan
    common = predicted.index.intersection(realized.index)
    p, r = predicted[common], realized[common]
    mask = p.notna() & r.notna()
    if mask.sum() < 5 or p[mask].nunique() < 2 or r[mask].nunique() < 2:
        return np.nan   # too few stocks, or a constant series (correlation undefined)
    return float(p[mask].corr(r[mask], method="spearman"))


# --------------------------------------------------------------------------
# Strategy adapters -- fn(mu, cov) or fn(mu, cov, context) -> weights
# --------------------------------------------------------------------------

def _normalize_output(out):
    """Accept an OptimizationResult-like object or a plain {ticker: weight}
    dict. Returns (weights, predicted_return, predicted_vol, posterior_mu)."""
    if isinstance(out, dict):
        return out, None, None, None
    diag = getattr(out, "diagnostics", None) or {}
    return (out.weights, getattr(out, "expected_return", None),
            getattr(out, "expected_volatility", None), diag.get("posterior_mu"))


def _call_strategy(fn, mu, cov, context):
    if "context" in inspect.signature(fn).parameters:
        return fn(mu, cov, context=context)
    return fn(mu, cov)


def equal_weight_strategy():
    def fn(mu, cov):
        n = len(mu)
        return {t: 1.0 / n for t in mu.index}
    return fn


def single_agent_strategy(agent, profile):
    """Run one optimizer agent on its own (e.g. MVOOptimizerAgent)."""
    from risk_profiling import RiskProfilingAgent
    constraints = RiskProfilingAgent().build_constraints(profile)

    def fn(mu, cov):
        return agent.optimize(mu, cov, constraints)
    return fn


def orchestrator_strategy(profile, orchestrator=None, sector_map=None,
                          use_sentiment=False, sentiment_scale=0.05):
    """The full pipeline: candidates -> coordinator -> Black-Litterman -> final
    step. With use_sentiment=True, each sentiment score s in [-1, 1] becomes a
    Black-Litterman view of  mu + s * sentiment_scale  for that stock.
    sentiment_scale (default 5% annual return per unit of sentiment) is an
    ASSUMPTION -- calibrate it; compare the with/without runs to see whether
    sentiment actually helped."""
    from orchestrator import PortfolioStrategyOrchestrator
    orch = orchestrator or PortfolioStrategyOrchestrator()

    def fn(mu, cov, context=None):
        views, confs = {}, {}
        if use_sentiment and context is not None and context.sentiment_scores:
            views = {t: float(mu[t] + s * sentiment_scale)
                     for t, s in context.sentiment_scores.items() if t in mu.index}
            confs = {t: c for t, c in context.sentiment_confidences.items() if t in views}
        out = orch.run(profile, mu, cov, sector_map=sector_map,
                       sentiment_views=views or None, sentiment_confidences=confs or None)
        result = out["recommended"]
        result.diagnostics["posterior_mu"] = out["posterior"]["mu"]
        result.diagnostics["n_sentiment_views"] = len(views)
        return result
    return fn


# --------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------

class Backtester:
    def __init__(self, prices: pd.DataFrame, eval_start=None, eval_end=None,
                 lookback_days: int = 252, rebalance_every: int = 21,
                 cost_rate: float = 0.0015, risk_free_rate: float = 0.06,
                 estimator: Callable = estimate_mu_cov, sentiment_provider=None):
        """prices: (dates x tickers) closes. It must start at least
        `lookback_days` trading days BEFORE eval_start -- that history is what
        the pipeline "knows" on the first rebalance.

        eval_start / eval_end: the window being tested. The first rebalance is
        the first trading day >= eval_start. Defaults: the earliest possible
        start, and the end of the data.
        cost_rate: fraction of traded notional paid on each buy or sell
        (0.0015 = 15 bps each way -- an assumption; use your broker's charges).
        """
        prices = prices.sort_index()
        if prices.index.has_duplicates:
            raise ValueError("price index has duplicate dates")
        if eval_end is not None:
            prices = prices.loc[:pd.Timestamp(eval_end)]

        clean, dropped = validate_ohlcv_frame(prices, source="BACKTEST")
        if dropped:
            warnings.warn(f"Backtester dropped {len(dropped)} ticker(s) failing validation: {dropped}")
        if clean.shape[1] < 2:
            raise ValueError("need at least 2 valid tickers to backtest")

        if eval_start is None:
            start_idx = lookback_days
        else:
            start_idx = int(clean.index.searchsorted(pd.Timestamp(eval_start)))
            if start_idx < lookback_days:
                have = start_idx
                raise ValueError(
                    f"eval_start {pd.Timestamp(eval_start).date()} needs {lookback_days} trading "
                    f"days of prices before it, but only {have} exist. Fetch earlier data "
                    f"(about {int(lookback_days * 1.6) + 10} calendar days before eval_start).")
        if start_idx >= len(clean) - 2:
            raise ValueError("the evaluation window has fewer than 2 trading days of data")

        self.prices = clean
        self.start_idx = start_idx
        self.lookback = lookback_days
        self.rebalance_every = rebalance_every
        self.cost_rate = cost_rate
        self.rf = risk_free_rate
        self.estimator = estimator
        self.sentiment_provider = sentiment_provider
        self._sentiment_cache = {}

    # -- inputs ---------------------------------------------------------

    def _clean_weights(self, weights: dict, universe) -> pd.Series:
        s = pd.Series(weights, dtype=float)
        outside = [t for t in s.index if t not in universe and abs(s[t]) > 1e-9]
        if outside:
            raise ValueError(f"weights on tickers outside the strategy's universe: {outside}")
        s = s.reindex(self.prices.columns).fillna(0.0)
        if (s < -1e-9).any():
            raise ValueError("negative weights (long-only backtester)")
        total = s.sum()
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"weights sum to {total:.4f}, expected ~1")
        return s / total

    def _fetch_sentiment(self, as_of, tickers):
        """-> (scores, confidences, status). Never raises, never invents data;
        status says exactly what happened."""
        if self.sentiment_provider is None:
            return {}, {}, "no provider configured"
        key = (as_of, tuple(tickers))
        if key in self._sentiment_cache:
            return self._sentiment_cache[key]

        universe = set(tickers)
        try:
            snap = self.sentiment_provider.get_snapshot(as_of, list(tickers))
        except SentimentUnavailable as e:
            res = ({}, {}, f"unavailable: {e}")
        except Exception as e:
            res = ({}, {}, f"error: {type(e).__name__}: {e}")
        else:
            cutoff = decision_cutoff(as_of)
            scores = {t: float(s) for t, s in snap.scores.items()
                      if t in universe and np.isfinite(s)}
            if snap.latest_timestamp is None:
                res = ({}, {}, "rejected: provider gave no timestamp, so point-in-time can't be verified")
            elif pd.Timestamp(snap.latest_timestamp) > cutoff:
                res = ({}, {}, f"rejected: look-ahead, newest item {pd.Timestamp(snap.latest_timestamp)} "
                               f"is after the {cutoff} cutoff")
            elif any(abs(s) > 1.0 + 1e-9 for s in scores.values()):
                res = ({}, {}, "rejected: a score is outside [-1, 1]")
            elif not scores:
                res = ({}, {}, "unavailable: no scores for the tradable universe")
            else:
                confs = {t: float(c) for t, c in snap.confidences.items()
                         if t in scores and np.isfinite(c) and 0 < c < 1}
                res = (scores, confs, "ok")
        self._sentiment_cache[key] = res
        return res

    # -- the walk-forward loop --------------------------------------------

    def run(self, name: str, strategy_fn: Callable) -> BacktestResult:
        prices, dates = self.prices, self.prices.index
        rets = prices.pct_change()
        n = len(dates)

        v = pd.Series(0.0, index=prices.columns)   # value held per asset
        cash, total = 1.0, 1.0
        daily, weight_rows, log_rows, holding_rows = {}, {}, [], []

        i = self.start_idx
        while i < n - 1:
            window = prices.iloc[i - self.lookback: i + 1]   # data up to and including date i
            usable = window.columns[window.notna().all()]
            cur_w = v / total
            error, pred_ret, pred_vol, post_mu, raw_mu = None, None, None, None, None
            scores, confs, s_status = self._fetch_sentiment(dates[i], list(usable))
            ctx = StrategyContext(dates[i], scores, confs, s_status)
            try:
                if len(usable) < 2:
                    raise ValueError("fewer than 2 tickers with full lookback history")
                raw_mu, cov = self.estimator(window[usable])
                out = _call_strategy(strategy_fn, raw_mu, cov, ctx)
                w, pred_ret, pred_vol, post_mu = _normalize_output(out)
                target = self._clean_weights(w, set(usable))
            except Exception as e:  # a failed rebalance must not silently look like a decision
                error = f"{type(e).__name__}: {e}"
                target = cur_w.copy()  # hold what we have; logged below
                pred_ret = pred_vol = post_mu = None

            traded = float((target - cur_w).abs().sum())
            cost = traded * self.cost_rate
            v = target * total
            cash = total - v.sum()

            j = min(i + self.rebalance_every, n - 1)
            block = rets.iloc[i + 1: j + 1].fillna(0.0)
            n_days = len(block)
            start_total, gross_rets = total, []
            for k, (d, r) in enumerate(block.iterrows()):
                prev_total = total
                v = v * (1 + r)
                total = v.sum() + cash
                gross_rets.append(total / prev_total - 1)
                if k == 0:   # pay the rebalance cost on the first day of the new period
                    v, cash, total = v * (1 - cost), cash * (1 - cost), total * (1 - cost)
                daily[d] = total / prev_total - 1

            gross = np.array(gross_rets)
            gross_period = float(np.prod(1 + gross) - 1)
            period_ret = total / start_total - 1
            asset_ret = (1 + block).prod() - 1      # what each stock actually did over the period

            pred_period = (1 + pred_ret) ** (n_days / TRADING_DAYS) - 1 \
                if pred_ret is not None and pred_ret > -1 else np.nan
            sd_period = pred_vol * np.sqrt(n_days / TRADING_DAYS) if pred_vol else np.nan
            forecast_z = (gross_period - pred_period) / sd_period \
                if sd_period and not np.isnan(pred_period) and sd_period > 1e-12 else np.nan

            held = target[target > 0.005]
            for t, wt in held.items():
                holding_rows.append({"date": dates[i], "ticker": t, "weight": float(wt),
                                     "actual_return": float(asset_ret[t]),
                                     "contribution": float(wt * asset_ret[t])})
            invested = target.sum()
            weight_rows[dates[i]] = target
            log_rows.append({
                "date": dates[i],
                "n_holdings": int(len(held)),
                "effective_n": float(1 / np.sum((target / invested) ** 2)) if invested > 0 else np.nan,
                "max_weight": float(target.max()),
                "turnover_one_way": 0.5 * traded,
                "cost": cost,
                "predicted_return": pred_ret,
                "predicted_return_period": pred_period,
                "predicted_vol": pred_vol,
                "gross_period_return": gross_period,
                "period_return": period_ret,
                "realized_return_ann": (1 + period_ret) ** (TRADING_DAYS / n_days) - 1
                                        if n_days else np.nan,
                "realized_vol": float(gross.std(ddof=1) * np.sqrt(TRADING_DAYS))
                                 if len(gross) > 1 else np.nan,
                "forecast_z": forecast_z,
                "rank_ic_raw_mu": _rank_ic(raw_mu, asset_ret) if error is None else np.nan,
                "rank_ic_posterior": _rank_ic(post_mu, asset_ret) if error is None else np.nan,
                "sentiment_status": s_status,
                "n_sentiment_scores": len(scores),
                "error": error,
            })
            i += self.rebalance_every

        daily_returns = pd.Series(daily).sort_index()
        log = pd.DataFrame(log_rows)
        metrics = compute_metrics(daily_returns, self.rf)
        metrics["n_rebalances"] = len(log)
        metrics["n_failed_rebalances"] = int(log["error"].notna().sum())
        metrics["avg_turnover_per_rebalance"] = float(log["turnover_one_way"].mean())
        metrics["total_cost_drag"] = float(log["cost"].sum())
        return BacktestResult(name, daily_returns, pd.DataFrame(weight_rows).T, log,
                              pd.DataFrame(holding_rows), metrics)

    def benchmark(self, name: str, index_prices: pd.Series) -> BacktestResult:
        """Buy-and-hold an index you supply (e.g. Nifty 50 closes) over the
        same window the strategies are scored on. No costs."""
        eval_dates = self.prices.index[self.start_idx + 1:]
        s = index_prices.sort_index().reindex(self.prices.index).ffill()
        r = s.pct_change().loc[eval_dates].fillna(0.0)
        return BacktestResult(name, r, pd.DataFrame(), pd.DataFrame(),
                              metrics=compute_metrics(r, self.rf))

    @staticmethod
    def compare(results: list) -> pd.DataFrame:
        rows = {}
        for res in results:
            m = res.metrics
            rows[res.name] = {
                "total_return": m.get("total_return"), "cagr": m.get("cagr"),
                "ann_volatility": m.get("ann_volatility"), "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"), "calmar": m.get("calmar"),
                "avg_turnover": m.get("avg_turnover_per_rebalance"),
                "cost_drag": m.get("total_cost_drag"),
                "failed_rebalances": m.get("n_failed_rebalances"),
            }
        return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _pct(x, signed=False):
    return "n/a" if pd.isna(x) else (f"{x:+.1%}" if signed else f"{x:.1%}")


def _num(x, fmt="{:+.3f}"):
    return "n/a" if pd.isna(x) else fmt.format(x)


def print_report(results: list, bt: "Backtester", detail: bool = False):
    table = Backtester.compare(results)
    show = table.copy()
    for c in ["total_return", "cagr", "ann_volatility", "max_drawdown", "avg_turnover", "cost_drag"]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    for c in ("sharpe", "calmar"):
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    start, end = results[0].daily_returns.index[[0, -1]]
    first_rebalance = bt.prices.index[bt.start_idx]
    print(f"\nBacktest: first rebalance {first_rebalance.date()}, returns measured {start.date()} -> "
          f"{end.date()}  |  lookback {bt.lookback}d, rebalance every {bt.rebalance_every}d, "
          f"cost {bt.cost_rate:.2%} per side of traded notional\n")
    print(show.to_string())

    print("\nPredicted vs actual (what the pipeline forecast for its own weights vs what happened):")
    for res in results:
        cal = res.calibration()
        if not cal or np.isnan(cal["avg_predicted_return"]):
            continue
        print(f"  {res.name}")
        print(f"    return:     predicted {_pct(cal['avg_predicted_return'])} a year, realized "
              f"{_pct(cal['realized_cagr'])} (gap {_pct(cal['return_gap'], True)})")
        print(f"    volatility: predicted {_pct(cal['avg_predicted_vol'])}, realized "
              f"{_pct(cal['realized_vol'])}; realized stayed within the prediction in "
              f"{_pct(cal['pct_periods_vol_within_prediction'])} of periods")
        print(f"    per-period error: realized return was within 1 predicted sd in "
              f"{_pct(cal['pct_within_1sd'])} of periods (about 68% if well calibrated); "
              f"mean z {_num(cal['mean_forecast_z'], '{:+.2f}')} (negative = realized below forecast)")
        ranking = (f"    stock ranking (rank IC, 0 = no skill): raw historical mean "
                   f"{_num(cal['mean_rank_ic_raw_mu'])}")
        if not np.isnan(cal["mean_rank_ic_posterior"]):
            ranking += (f", pipeline posterior {_num(cal['mean_rank_ic_posterior'])} "
                        f"(positive in {_pct(cal['pct_periods_posterior_ic_positive'])} of periods)")
        print(ranking)

    print("\nSentiment:")
    logged = [r for r in results if not r.rebalance_log.empty]
    rep = logged[0].sentiment_report() if logged else None
    if rep is None or set(rep["by_status"]) == {"no provider configured"}:
        print("  no sentiment provider configured -- every run excluded sentiment")
    else:
        others = ", ".join(f"{v} {k}" for k, v in rep["by_status"].items() if k != "ok")
        print(f"  retrieval: usable sentiment at {rep['n_used']} of {rep['n_rebalances']} rebalances"
              f"{' (' + others + ')' if others else ''}. On the other dates the pipeline ran without it.")
        problems = rep["problems"]
        for _, row in problems.head(8).iterrows():
            print(f"      {row['date'].date()}  {row['sentiment_status']}")
        if len(problems) > 8:
            print(f"      ... and {len(problems) - 8} more (full list in the rebalance_log CSV via --out)")
        print("  runs that used it: " + (", ".join(r.name for r in results if "+sentiment" in r.name) or "none"))

    if detail:
        cols = ["date", "n_holdings", "effective_n", "predicted_return_period", "gross_period_return",
                "predicted_vol", "realized_vol", "forecast_z", "rank_ic_posterior", "sentiment_status"]
        for res in results:
            if res.rebalance_log.empty or res.rebalance_log["predicted_vol"].isna().all():
                continue
            print(f"\nPer-rebalance detail: {res.name}")
            d = res.rebalance_log[cols].copy()
            d["date"] = d["date"].dt.date
            for c in ("predicted_return_period", "gross_period_return", "predicted_vol", "realized_vol"):
                d[c] = d[c].map(_pct)
            for c in ("effective_n", "forecast_z", "rank_ic_posterior"):
                d[c] = d[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
            d["sentiment_status"] = d["sentiment_status"].str.slice(0, 40)
            d = d.rename(columns={"sentiment_status": "sentiment_retrieval"})   # whether it COULD be fetched
            print(d.to_string(index=False))


def save_outputs(results: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for res in results:
        stem = res.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        for suffix, frame in (("rebalance_log", res.rebalance_log), ("weights", res.weights_history),
                              ("holdings_detail", res.holdings_detail)):
            if len(frame):
                frame.to_csv(os.path.join(out_dir, f"{stem}__{suffix}.csv"))
    print(f"\nSaved per-rebalance logs, weights and holding contributions to {out_dir}/")


# --------------------------------------------------------------------------
# Data + CLI
# --------------------------------------------------------------------------

def make_synthetic_prices(n_tickers: int = 20, n_days: int = 1300, seed: int = 0) -> pd.DataFrame:
    """One-factor synthetic market: for exercising the machinery only. Returns
    here are iid, so a backtest on this data says nothing about real
    performance -- it only shows the plumbing works."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.010, n_days)
    beta = rng.uniform(0.6, 1.4, n_tickers)
    drift = rng.uniform(0.0, 0.0006, n_tickers)
    idio = rng.uniform(0.008, 0.018, n_tickers)
    r = market[:, None] * beta + drift + rng.normal(0, 1, (n_days, n_tickers)) * idio
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates,
                        columns=[f"STOCK{i:02d}.NS" for i in range(n_tickers)])


def _load_prices(args, eval_start, eval_end) -> pd.DataFrame:
    if args.synthetic:
        return make_synthetic_prices()
    if args.cache and os.path.exists(args.cache):
        return pd.read_csv(args.cache, index_col=0, parse_dates=True)

    from data.nifty50 import yahoo_tickers
    tickers = yahoo_tickers()
    fetch_start = eval_start - timedelta(days=int(args.lookback * 1.6) + 10)  # history before eval_start
    fetch_end = eval_end.date() if eval_end else date.today()
    if args.source == "yfinance":
        from data import yfinance_source
        prices = yfinance_source.fetch_price_history(
            tickers, period=f"{(date.today() - fetch_start.date()).days}d")
    else:
        from data import source_router
        prices = source_router.get_historical_ohlcv(tickers, fetch_start.date(), fetch_end)
    print(f"Price data source: {prices.attrs.get('source', 'YFINANCE')}")
    if args.cache:
        prices.to_csv(args.cache)
    return prices


def main():
    from risk_profiling import UserProfile, RiskTolerance, InvestmentObjective, TimeHorizon
    from optimizers.mvo_qp import MVOOptimizerAgent
    from sentiment_provider import CSVSentimentProvider

    p = argparse.ArgumentParser(description="Walk-forward backtest of the portfolio pipeline")
    p.add_argument("--start", help="first rebalance on/after this date, YYYY-MM-DD")
    p.add_argument("--end", help="last date included, YYYY-MM-DD (default: latest data)")
    p.add_argument("--years", type=int, default=3, help="window length if --start isn't given")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=15.0, help="per side, basis points of traded notional")
    p.add_argument("--stocks", type=int, default=None, help="target number of holdings")
    p.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate")
    p.add_argument("--max-position", type=float, default=0.25)
    p.add_argument("--sentiment-csv", help="historical sentiment CSV (published_at,ticker,score[,confidence])")
    p.add_argument("--sentiment-scale", type=float, default=0.05,
                   help="annual return view per unit of sentiment score (assumption)")
    p.add_argument("--detail", action="store_true", help="print the per-rebalance table")
    p.add_argument("--out", help="directory for per-rebalance CSVs")
    p.add_argument("--synthetic", action="store_true", help="made-up prices (plumbing check only)")
    p.add_argument("--source", choices=["router", "yfinance"], default="router")
    p.add_argument("--cache", help="CSV: load prices from it if present, else save there")
    args = p.parse_args()

    eval_end = pd.Timestamp(args.end) if args.end else None
    if args.start:
        eval_start = pd.Timestamp(args.start)
    elif args.synthetic:
        eval_start = None
    else:
        eval_start = pd.Timestamp((eval_end or pd.Timestamp.today()) - pd.DateOffset(years=args.years))

    prices = _load_prices(args, eval_start, eval_end)
    provider = CSVSentimentProvider(args.sentiment_csv) if args.sentiment_csv else None
    bt = Backtester(prices, eval_start=eval_start, eval_end=eval_end, lookback_days=args.lookback,
                    rebalance_every=args.rebalance, cost_rate=args.cost_bps / 10_000,
                    sentiment_provider=provider)
    profile = UserProfile(risk_tolerance=RiskTolerance(args.risk),
                          objective=InvestmentObjective.BALANCED_GROWTH,
                          time_horizon=TimeHorizon.MEDIUM, max_single_position=args.max_position,
                          target_num_holdings=args.stocks)

    results = [bt.run("Pipeline (no sentiment)", orchestrator_strategy(profile))]
    if provider:
        results.append(bt.run("Pipeline (+sentiment)", orchestrator_strategy(
            profile, use_sentiment=True, sentiment_scale=args.sentiment_scale)))
    results += [bt.run("MVO alone", single_agent_strategy(MVOOptimizerAgent(), profile)),
                bt.run("Equal weight", equal_weight_strategy())]
    print_report(results, bt, detail=args.detail)
    if args.out:
        save_outputs(results, args.out)


if __name__ == "__main__":
    main()