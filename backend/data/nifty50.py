"""
Nifty 50 universe: NSE symbols, Yahoo Finance tickers, and sector mapping.

"""

# NSE symbol -> sector, as currently listed
NIFTY50_SECTORS = {
    "ADANIENT": "Metals & Mining",
    "ADANIPORTS": "Services",
    "APOLLOHOSP": "Healthcare",
    "ASIANPAINT": "Consumer Durables",
    "AXISBANK": "Financial Services",
    "BAJAJ-AUTO": "Automobile and Auto Components",
    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",
    "BEL": "Capital Goods",
    "BHARTIARTL": "Telecommunication",
    "CIPLA": "Healthcare",
    "COALINDIA": "Oil, Gas & Consumable Fuels",
    "DRREDDY": "Healthcare",
    "EICHERMOT": "Automobile and Auto Components",
    "ETERNAL": "Consumer Services",
    "GRASIM": "Construction Materials",
    "HCLTECH": "Information Technology",
    "HDFCBANK": "Financial Services",
    "HDFCLIFE": "Financial Services",
    "HINDALCO": "Metals & Mining",
    "HINDUNILVR": "Fast Moving Consumer Goods",
    "ICICIBANK": "Financial Services",
    "ITC": "Fast Moving Consumer Goods",
    "INFY": "Information Technology",
    "INDIGO": "Services",
    "JSWSTEEL": "Metals & Mining",
    "JIOFIN": "Financial Services",
    "KOTAKBANK": "Financial Services",
    "LT": "Construction",
    "M&M": "Automobile and Auto Components",
    "MARUTI": "Automobile and Auto Components",
    "MAXHEALTH": "Healthcare",
    "NTPC": "Power",
    "NESTLEIND": "Fast Moving Consumer Goods",
    "ONGC": "Oil, Gas & Consumable Fuels",
    "POWERGRID": "Power",
    "RELIANCE": "Oil, Gas & Consumable Fuels",
    "SBILIFE": "Financial Services",
    "SHRIRAMFIN": "Financial Services",
    "SBIN": "Financial Services",
    "SUNPHARMA": "Healthcare",
    "TCS": "Information Technology",
    "TATACONSUM": "Fast Moving Consumer Goods",
    "TMPV": "Automobile and Auto Components",  # low-confidence ticker, see module docstring
    "TATASTEEL": "Metals & Mining",
    "TECHM": "Information Technology",
    "TITAN": "Consumer Durables",
    "TRENT": "Consumer Services",
    "ULTRACEMCO": "Construction Materials",
    "WIPRO": "Information Technology",
}

# NSE symbols that need remapping to their actual Yahoo Finance ticker
_YAHOO_OVERRIDES = {
    "M&M": "M&M.NS",       
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
}


def nse_symbols() -> list:
    """Plain NSE trading symbols, e.g. 'RELIANCE', 'TCS'."""
    return list(NIFTY50_SECTORS.keys())


def yahoo_tickers() -> list:
    """NSE symbols converted to Yahoo Finance tickers."""
    return [_YAHOO_OVERRIDES.get(s, f"{s}.NS") for s in NIFTY50_SECTORS]


def sector_map(yahoo_format: bool = True) -> dict:
    """Ticker -> sector mapping for the NIfty 50 Universe. 
    yahoo_format=True keys by the Yahoo ticker (matching what data_fetcher.py returns); 
    False keys by the plain NSE symbol."""
    if not yahoo_format:
        return dict(NIFTY50_SECTORS)
    return {
        _YAHOO_OVERRIDES.get(s, f"{s}.NS"): sector
        for s, sector in NIFTY50_SECTORS.items()
    }
