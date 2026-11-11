"""
NSE Real-Time Source

Scope: live market prices/quotes only -- this is the one source in the
platform meant for "what is this trading at right now," not historical
analysis (that's nse_bhavcopy.py) or fundamentals (that's yfinance_source.py).

"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .validation import MarketDataPoint, validate_point


class LiveQuoteSource(ABC):
    """Will be implemented against actual NSE real-time subscription."""

    @abstractmethod
    def _fetch_raw(self, ticker: str) -> dict:
        """Return a dict with at least {'price': float, 'timestamp': datetime}
        and optionally {'volume': float}, from your subscription's API.
        Raise on failure -- do not return a dict with a fabricated price."""
        raise NotImplementedError

    def get_quote(self, ticker: str) -> MarketDataPoint:
        """Fetches and validates a live quote. 
        (fall back to the last Bhavcopy close, never silently 
        substitute one source's data as if it were the other).
        """
        raw = self._fetch_raw(ticker)
        return validate_point(
            ticker, source="NSE_REALTIME",
            timestamp=raw.get("timestamp", datetime.now(timezone.utc)),
            price=raw.get("price"), volume=raw.get("volume"),
        )


class MockLiveQuoteSource(LiveQuoteSource):
    """
    Deterministic stand-in for testing the rest of the pipeline without a
    real subscription.`
    """

    def __init__(self, fixed_quotes: dict):
        """fixed_quotes: {ticker: price}"""
        self._quotes = fixed_quotes

    def _fetch_raw(self, ticker: str) -> dict:
        if ticker not in self._quotes:
            raise KeyError(f"No mock quote configured for {ticker}")
        return {"price": self._quotes[ticker], "timestamp": datetime.now(timezone.utc)}
