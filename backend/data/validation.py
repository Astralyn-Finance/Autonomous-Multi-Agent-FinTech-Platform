"""
Data Validation

Every piece of market data that reaches an optimizer should have gone through
here first.

`MarketDataPoint` carries its own source and timestamp, so two records for
the same ticker from different sources are never silently interchangeable
whoever consumes them can see where each one came from and decide what to do,
rather than have them quietly averaged or overwritten.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import math


@dataclass
class MarketDataPoint:
    ticker: str
    source: str                      # e.g. "NSE_REALTIME", "NSE_BHAVCOPY", "YFINANCE"
    timestamp: datetime
    price: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    is_valid: bool = True
    errors: list = field(default_factory=list)

    def invalidate(self, reason: str):
        self.is_valid = False
        self.errors.append(reason)


def _is_finite_positive(x) -> bool:
    return x is not None and math.isfinite(x) and x > 0


def validate_point(ticker: str, source: str, timestamp: datetime,
                    price: Optional[float] = None,
                    volume: Optional[float] = None,
                    market_cap: Optional[float] = None,
                    max_future_skew_seconds: int = 60) -> MarketDataPoint:
    """Validate one market data record. Fields that are None stay None in the
    result -- they are never coerced to 0. A field that IS provided but fails
    a sanity check gets recorded in `errors` and flips `is_valid` to False;
    the record is still returned (not discarded) so a caller can decide how
    to handle it rather than have that decision made silently here.
    """
    point = MarketDataPoint(ticker=ticker, source=source, timestamp=timestamp,
                             price=price, volume=volume, market_cap=market_cap)

    if not ticker or not isinstance(ticker, str):
        point.invalidate("ticker missing or not a string")

    now = datetime.now(timezone.utc)
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    if (ts - now).total_seconds() > max_future_skew_seconds:
        point.invalidate(f"timestamp {timestamp.isoformat()} is in the future")

    if price is not None and not _is_finite_positive(price):
        point.invalidate(f"price {price!r} is not a finite positive number")

    if volume is not None and (not math.isfinite(volume) or volume < 0):
        point.invalidate(f"volume {volume!r} is negative or not finite")

    if market_cap is not None and not _is_finite_positive(market_cap):
        point.invalidate(f"market_cap {market_cap!r} is not a finite positive number")

    return point


def validate_ohlcv_frame(prices, source: str, timestamp: Optional[datetime] = None):
    """Validate an entire (dates x tickers) price DataFrame in one pass.
    Returns (clean_prices, dropped_tickers): clean_prices has any column that
    failed validation removed entirely -- a single bad column doesn't get
    silently zero-filled or interpolated, it gets dropped and reported.
    """
    import pandas as pd
    import numpy as np

    ts = timestamp or datetime.now(timezone.utc)
    dropped = []
    good_cols = []

    for col in prices.columns:
        series = prices[col].dropna()
        if series.empty:
            dropped.append((col, "no data"))
            continue
        last_price = float(series.iloc[-1])
        point = validate_point(col, source, ts, price=last_price)
        if not point.is_valid:
            dropped.append((col, "; ".join(point.errors)))
            continue
        if (series <= 0).any() or not np.isfinite(series.values).all():
            dropped.append((col, "series contains non-positive or non-finite values"))
            continue
        good_cols.append(col)

    clean_prices = prices[good_cols].copy()
    return clean_prices, dropped
