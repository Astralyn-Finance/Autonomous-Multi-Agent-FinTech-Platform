"""
Authoritative Nifty 50 Data Loader.
Fetches historical prices (NSE Bhavcopy first, yfinance fallback),
computes statistical mu & cov estimates, and extracts market caps and sector maps.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .nifty50 import yahoo_tickers, sector_map
from .estimation import estimate_mu_cov  
from . import source_router


def load_nifty50_data(
        years: int = 3,
        limit: Optional[int] = None
    )-> Tuple[pd.Series, pd.DataFrame, Optional[pd.Series], Dict[str, str], str]:
    """
    Fetch historical prices (NSE Bhavcopy first, yfinance fallback).
    Returns (mu, cov, market_caps, sector_map, price_source).
    """
    all_tickers = yahoo_tickers()
    tickers = all_tickers[:limit] if limit else all_tickers

    end = date.today()
    start = end - timedelta(days=365 * years)

    try:
        prices = source_router.get_historical_ohlcv(tickers, start, end)
        # Drop columns completely missing and align
        valid_cols = [c for c in tickers if c in prices.columns]
        prices = prices[valid_cols].dropna(axis=0, how="all").ffill().bfill()
        
        mu, cov = estimate_mu_cov(prices)
        source = prices.attrs.get("source", "unknown")
    except Exception as exc:
        # Fallback generator for isolated offline testing
        valid_cols = tickers
        mu = pd.Series([0.12 + 0.005 * i for i in range(len(tickers))], index=tickers)
        np.random.seed(42)
        A = np.random.randn(len(tickers), len(tickers)) * 0.05
        cov = pd.DataFrame(A @ A.T + np.eye(len(tickers)) * 0.02, index=tickers, columns=tickers)
        source = f"FALLBACK_SIMULATED (Reason: {exc})"

    # Fundamentals / Market Caps via source_router policy
    try:
        fundamentals = source_router.get_fundamentals(list(valid_cols))
        market_caps_dict = {}
        for t, p in fundamentals.items():
            cap = getattr(p, "market_cap", None)
            if cap is None and isinstance(p, dict):
                cap = p.get("market_cap")
            if cap and float(cap) > 0:
                market_caps_dict[t] = float(cap)
        market_caps = pd.Series(market_caps_dict) if market_caps_dict else None
    except Exception:
        market_caps = pd.Series({t: 1_000_000.0 for t in valid_cols})

    sectors = sector_map()
    return mu, cov, market_caps, sectors, source


if __name__ == "__main__":
    mu, cov, caps, sectors, source = load_nifty50_data(limit=10)
    print(f"Price data source: {source}")
    print("Estimated annual returns (mu):")
    print(mu.head())
    print(f"\nMarket caps fetched for {0 if caps is None else len(caps)} / {len(mu)} tickers")