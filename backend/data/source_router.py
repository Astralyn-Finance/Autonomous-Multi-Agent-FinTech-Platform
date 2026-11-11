"""
Source Router

The ONE place that decides which data source answers which kind of request.
Everything else in the pipeline should go through here rather than importing
nse_bhavcopy / yfinance_source / nse_realtime directly, so the source policy
lives in exactly one place:

  - live price/quote          -> NSE real-time subscription (nse_realtime)
  - historical/EOD OHLCV      -> NSE Bhavcopy first; yfinance ONLY as fallback
  - fundamentals/metadata     -> yfinance

"""

from datetime import date

import pandas as pd

from . import nse_bhavcopy, yfinance_source


def get_historical_ohlcv(tickers: list, start: date, end: date,
                          bhavcopy_fetcher=nse_bhavcopy.fetch_bhavcopy_range,
                          yfinance_fetcher=yfinance_source.fetch_price_history) -> pd.DataFrame:
    """Returns a (dates x tickers) close-price DataFrame from exactly one
    source. `.attrs["source"]` on the result says which one. The fetcher
    functions are injectable (default to the real ones) so this routing
    logic is testable without a network call.
    """
    try:
        prices = bhavcopy_fetcher(tickers, start, end)
        prices.attrs["source"] = "NSE_BHAVCOPY"
        return prices
    except Exception as bhavcopy_error:
        period_days = (end - start).days
        prices = yfinance_fetcher(tickers, period=f"{max(period_days, 1)}d")
        prices.attrs["source"] = "YFINANCE"
        prices.attrs["fallback_reason"] = f"Bhavcopy failed: {bhavcopy_error}"
        return prices


def get_live_price(ticker: str, live_source) -> "MarketDataPoint":
    """live_source: an instance of nse_realtime.LiveQuoteSource (your
    subscription's implementation, or MockLiveQuoteSource for testing).
    No default -- there's no sensible default live-quote source to fall
    back to silently."""
    return live_source.get_quote(ticker)


def get_fundamentals(tickers: list = None,
                      fetcher=yfinance_source.fetch_fundamentals) -> dict:
    """Fundamentals (market cap, shares outstanding) -- yfinance only, per
    the source policy. Returns {ticker: MarketDataPoint}; a ticker with no
    retrievable market cap is simply absent, never present with a 0."""
    return fetcher(tickers)
