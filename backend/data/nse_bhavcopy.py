"""
NSE Bhavcopy Source

Primary source for historical/EOD OHLCV + volume data, per the data-source
policy: NSE Bhavcopy for reliable historical data, yfinance only as fallback.

"""

from datetime import date, timedelta
from io import StringIO

import pandas as pd

from .validation import validate_point



_BHAVCOPY_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"

# NSE's Bhavcopy column names -> our normalized names. Verify against a live
# sample before trusting this -- see module docstring, point 3.
_COLUMN_MAP = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "CLOSE_PRICE": "close",
    "PREV_CLOSE": "prev_close",
    "TTL_TRD_QNTY": "volume",
    "DATE1": "date",
}


def _session():
    
    import requests
    s = requests.Session()

    # custom browser headers to bypass blocks.
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/csv,application/csv,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/all-reports",
    })

    # Warm-up hit: establishes the session cookies the actual CSV request
    # needs. Ignore its result beyond that, we only want the cookies.
    try:
        s.get("https://www.nseindia.com/all-reports", timeout=10)
    except Exception:
        pass  # if even the warm-up fails, the real request below will raise error
              
    return s


def parse_bhavcopy_csv(csv_text: str) -> pd.DataFrame:
    """Pure parsing step."""
    raw = pd.read_csv(StringIO(csv_text))
    raw.columns = [c.strip() for c in raw.columns]
    missing_cols = set(_COLUMN_MAP) - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Bhavcopy format mismatch -- expected columns not found: {missing_cols}. "
                          f"NSE likely changed the CSV schema; update _COLUMN_MAP.")

    df = raw.rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"]
    return df


def fetch_bhavcopy(for_date: date) -> pd.DataFrame:
    """Fetch and parse one day's Bhavcopy. Returns a DataFrame of EQ-series
    rows only (excludes other series like debt instruments), columns
    normalized per _COLUMN_MAP, with source-tagged validation applied.
    Raises on network failure or unexpected format, callers (source_router)
    should catch and fall back to yfinance rather than let this crash silently.
    """
    url = _BHAVCOPY_URL.format(ddmmyyyy=date.today().strftime("%d%m%Y"))
    resp = _session().get(url, timeout=15)
    resp.raise_for_status()

    df = parse_bhavcopy_csv(resp.text)
    return _validate_bhavcopy_rows(df, for_date)


def _validate_bhavcopy_rows(df: pd.DataFrame, for_date: date) -> pd.DataFrame:
    """Row-level validation: a row with a non-positive close or negative
    volume is dropped (not zero-filled) and not silently included."""
    import datetime as dt
    ts = dt.datetime.combine(for_date, dt.time(15, 30), tzinfo=dt.timezone.utc)  # NSE close, IST-ish

    keep = []
    for _, row in df.iterrows():
        point = validate_point(row["symbol"], source="NSE_BHAVCOPY", timestamp=ts,
                                price=row.get("close"), volume=row.get("volume"))
        if point.is_valid:
            keep.append(True)
        else:
            keep.append(False)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def verify_bhavcopy_source(for_date: date = None) -> dict:
    """Run this once, locally, with real internet access, before trusting
    fetch_bhavcopy() in production. Prints a clear diagnosis instead of a
    buried stack trace, and returns a dict so you can inspect it further.
    Usage: `python -c "from data.nse_bhavcopy import verify_bhavcopy_source as v; v()"`
    """
    
    url = _BHAVCOPY_URL.format(ddmmyyyy=date.today().strftime("%d%m%Y"))
    report = {"url": url, "date": date.today(), "ok": False}

    try:
        resp = _session().get(url, timeout=15)
    except Exception as e:
        report["error"] = f"Network/connection failure: {e}"
        print(f"FAILED to connect at all: {e}\nCheck your own internet access first.")
        return report

    report["status_code"] = resp.status_code
    if resp.status_code != 200:
        report["error"] = f"HTTP {resp.status_code}"
        print(f"Got HTTP {resp.status_code} from {url}")
        print(f"First 300 chars of response: {resp.text[:300]!r}")
        print("If this is a 403/401: the warm-up cookie approach in _session() "
              "isn't enough -- NSE likely wants a fresher User-Agent or a real "
              "browser session (try Selenium/Playwright as a last resort).")
        print("If this is a 404: the URL pattern has changed -- check "
              "nseindia.com/all-reports for the current Bhavcopy link.")
        return report

    try:
        df = parse_bhavcopy_csv(resp.text)
    except ValueError as e:
        report["error"] = str(e)
        print(f"Got a 200 OK but parsing failed: {e}")
        print(f"Actual columns in the response: {list(pd.read_csv(StringIO(resp.text)).columns)}")
        print("Update _COLUMN_MAP in this file to match the columns printed above.")
        return report

    report["ok"] = True
    report["row_count"] = len(df)
    report["sample_tickers"] = df["symbol"].head(5).tolist()
    print(f"OK: fetched and parsed {len(df)} EQ rows for {date.today()}.")
    print(f"Sample tickers: {report['sample_tickers']}")
    return report


def fetch_bhavcopy_range(tickers: list, start: date, end: date) -> pd.DataFrame:
    """Fetch and stitch together a date range, returning a (dates x tickers)
    close-price DataFrame -- the shape estimate_mu_cov() expects. Trading
    holidays will simply have no Bhavcopy file for that date; those dates are
    skipped rather than raising."""
    frames = {}
    d = start
    while d <= end:
        try:
            day_df = fetch_bhavcopy(d)
            frames[d] = day_df.set_index("symbol")["close"]
        except Exception:
            pass  # likely a non-trading day or a transient fetch issue -- skip, don't crash the range
        d += timedelta(days=1)

    if not frames:
        raise RuntimeError(f"No Bhavcopy data retrieved for {start}..{end}")

    prices = pd.DataFrame(frames).T
    if tickers:
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
    return prices
