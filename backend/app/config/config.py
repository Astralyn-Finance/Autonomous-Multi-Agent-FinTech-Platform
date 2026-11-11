"""
Configuration file which will store all the necessary settings 
and connection information that the application will use
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Configure behavior for environment variables in settings models
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Core Application Metadata
    APP_NAME: str = "Fintech Platform API"
    ENVIRONMENT: str = "development"  # development | staging | production

    # Database Configuration
    DATABASE_URL: str
    DB_ECHO: bool = False            # Keeps console clean

    # Market Data Settings
    market_data_interval_minutes: int = 15     #Fetch/update market data every 15 minutes
    news_interval_minutes: int = 10            #controls how frequently your system checks for new financial news
    market_index: str = "^NSEI"                 #Monitor the NIFTY 50 index
    ticker_universe: str = "HDFCBANK.NS, ICICIBANK.NS, RELIANCE.NS, BHARTIARTL.NS, LT.NS, SBIN.NS, INFY.NS, AXISBANK.NS, BAJFINANCE.NS, M&M.NS, TCS.NS"

    @property
    def tickers(self) -> list[str]:
        return [t.strip().upper() for t in self.ticker_universe.split(",") if t.strip()]


settings = Settings()