"""
Yahoo Finance Source

NSE Bhavcopy (nse_bhavcopy.py) is the primary source for
historical OHLCV, yfinance is the fallback when Bhavcopy is unavailable,
not a first choice. 
"""

import time
from datetime import datetime, timezone

import pandas as pd

from .nifty50 import yahoo_tickers
from .validation import validate_ohlcv_frame, validate_point

# Yahoo's batch endpoint gets noticeably less reliable as the ticker count in
# one request grows -- ~50 tickers (the full Nifty 50) in a single call is a
# known source of partial/empty responses. Chunking into smaller batches is
# the standard, well-documented fix, independent of anything sandbox-specific.
_BATCH_SIZE = 15
_RETRIES_PER_BATCH = 3
_RETRY_BACKOFF_SECONDS = 2


def _download_batch(yf, tickers, period, interval):
    last_error = None
    for attempt in range(_RETRIES_PER_BATCH):
        try:
            raw = yf.download(tickers, period=period, interval=interval,
                               auto_adjust=True, progress=False, threads=False)
            if raw is not None and not raw.empty:
                return raw
            last_error = RuntimeError("empty response")
        except Exception as e:
            last_error = e
        time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))  # backs off: 2s, 4s, 6s
    raise RuntimeError(f"yfinance batch {tickers} failed after {_RETRIES_PER_BATCH} attempts: {last_error}")


def fetch_price_history(tickers=None, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Fallback historical price fetch. Returns a validated (dates x tickers)
    DataFrame -- any ticker whose data fails validation is dropped, not
    zero-filled or interpolated (see data/validation.py). Fetches in small
    batches with retries rather than one call for the whole universe -- see
    _BATCH_SIZE comment above for why."""
    import yfinance as yf

    tickers = tickers or yahoo_tickers()
    frames = []
    failed_batches = []
    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i:i + _BATCH_SIZE]
        try:
            raw = _download_batch(yf, batch, period, interval)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if not isinstance(raw.columns, pd.MultiIndex) and len(batch) == 1:
                prices.columns = batch  # single-ticker download doesn't label the column
            frames.append(prices)
        except Exception as e:
            failed_batches.append((batch, str(e)))

    if not frames:
        raise RuntimeError(f"All yfinance batches failed: {failed_batches}")

    combined = pd.concat(frames, axis=1)
    combined = combined.dropna(how="all").ffill()

    clean, dropped = validate_ohlcv_frame(combined, source="YFINANCE")
    if dropped or failed_batches:
        import warnings
        msg = f"yfinance: dropped {len(dropped)} ticker(s) failing validation: {dropped}"
        if failed_batches:
            msg += f"; {len(failed_batches)} batch(es) failed entirely: {failed_batches}"
        warnings.warn(msg)
    return clean


def fetch_fundamentals(tickers=None) -> dict:
    """Best-effort fundamentals lookup (market cap, shares outstanding) via
    yfinance. Returns {ticker: MarketDataPoint}. A ticker whose market cap
    could not be retrieved is simply ABSENT from the result -- never present
    with a market_cap of 0. Yahoo's info endpoint is slow/unreliable across
    many tickers in one run; a partial result here is normal, not an error.
    One retry per ticker on failure before giving up on it.
    """
    import yfinance as yf

    tickers = tickers or yahoo_tickers()
    now = datetime.now(timezone.utc)
    out = {}
    for t in tickers:
        cap, shares = None, None
        for attempt in range(2):
            try:
                info = yf.Ticker(t).fast_info
                cap = info.get("marketCap") if hasattr(info, "get") else None
                shares = info.get("shares") if hasattr(info, "get") else None
                break
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                continue
        if cap is None:
            continue  # missing stays missing -- do NOT record a 0
        point = validate_point(t, source="YFINANCE", timestamp=now, market_cap=cap)
        if shares is not None:
            point.shares_outstanding = shares  # not a formal MarketDataPoint field yet;
        out[t] = point                          # attached ad hoc rather than left unused
    return out


def verify_yfinance_source(sample_tickers: list = None) -> dict:
    """Run this once, locally, with real internet access, before relying on
    this module. Usage:
    `python -c "from data.yfinance_source import verify_yfinance_source as v; v()"`
    """
    import yfinance as yf

    sample_tickers = sample_tickers or yahoo_tickers()[:3]
    report = {"tickers_tried": sample_tickers, "ok": False}

    try:
        raw = yf.download(sample_tickers, period="5d", progress=False, threads=False)
    except Exception as e:
        report["error"] = f"Connection failure: {e}"
        print(f"FAILED to connect: {e}\nCheck your own internet access first.")
        return report

    if raw is None or raw.empty:
        report["error"] = "empty response"
        print("Got an empty response. yfinance's own output above (if any) usually "
              "explains why -- commonly a rate limit or a delisted/renamed ticker.")
        return report

    report["ok"] = True
    report["shape"] = raw.shape
    report["columns"] = list(raw.columns)[:10]
    print(f"OK: fetched a {raw.shape} frame for {sample_tickers}.")
    print(f"Sample columns: {report['columns']}")
    return report
