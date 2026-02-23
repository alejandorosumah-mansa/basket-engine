"""
Fetch cross-asset benchmark data for factor modeling.

Benchmarks:
    SPY, QQQ, GLD, TLT, ^TNX, ^IRX, ^VIX, DX-Y.NYB, BTC-USD, USO

Saves daily prices to data/processed/benchmarks.parquet
"""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf


BENCHMARKS = {
    # Existing factors (keep all)
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100", 
    "GLD": "Gold",
    "TLT": "20+ Year Treasury",
    "^TNX": "10Y Treasury Yield",
    "^IRX": "3M T-Bill Rate",
    "^VIX": "VIX",
    "DX-Y.NYB": "US Dollar Index",
    "BTC-USD": "Bitcoin",
    "USO": "Oil",
    
    # US Rates (new duration ETFs and yields)
    "SHY": "1-3Y Treasury ETF",
    "^FVX": "5Y Treasury Yield",
    "TLH": "10-20Y Treasury ETF", 
    "^TYX": "30Y Treasury Yield",
    
    # Global Rates (bond ETFs)
    "IGLT.L": "UK Gilts ETF",
    "IBGL.L": "Germany Bund ETF", 
    "BNDX": "Intl Bond ex-US",
    "EMB": "Emerging Market Bonds",
    "BWX": "Intl Treasury Bond",
    "IGOV": "Intl Govt Bond ETF",
    
    # Global Equity Indices
    "^FTSE": "UK FTSE 100",
    "^GDAXI": "Germany DAX",
    "^N225": "Japan Nikkei 225",
    "000001.SS": "China Shanghai Composite",
    "^FCHI": "France CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^HSI": "Hong Kong Hang Seng",
    "^BSESN": "India BSE Sensex",
    "^BVSP": "Brazil Bovespa",
    "^KS11": "South Korea KOSPI",
    
    # Country ETFs
    "EWC": "Canada ETF",
    "EWA": "Australia ETF", 
    "EWW": "Mexico ETF",
    "EWT": "Taiwan ETF",
    "EIDO": "Indonesia ETF",
    "TUR": "Turkey ETF",
    "EZA": "South Africa ETF",
    "KSA": "Saudi Arabia ETF",
    "EWL": "Switzerland ETF",
    "EWS": "Singapore ETF",
    
    # Additional Commodities
    "NG=F": "Natural Gas",
}

OUTPUT_PATH = Path("data/processed/benchmarks.parquet")


def fetch_benchmarks(period: str = "2y") -> pd.DataFrame:
    """Fetch daily close prices for all benchmarks.
    
    Args:
        period: yfinance period string (default 2y for safety margin)
    
    Returns:
        DataFrame with DatetimeIndex and one column per ticker
    """
    tickers = list(BENCHMARKS.keys())
    print(f"Fetching {len(tickers)} benchmarks: {tickers}")
    
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True)
    
    # yf.download returns MultiIndex columns (Price, Ticker) for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = tickers
    
    # Clean column names for parquet compatibility
    closes.columns = [c.replace("^", "").replace("-", "_").replace(".", "_").replace("=", "_") for c in closes.columns]
    
    # Forward-fill gaps (weekends/holidays already excluded by yfinance)
    closes = closes.ffill()
    
    print(f"Date range: {closes.index.min()} to {closes.index.max()}")
    print(f"Shape: {closes.shape}")
    print(f"Missing values:\n{closes.isna().sum()}")
    
    return closes


def save_benchmarks(df: pd.DataFrame) -> Path:
    """Save benchmark data to parquet."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


# Mapping from clean column names back to descriptions
CLEAN_NAMES = {
    # Existing factors
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "GLD": "Gold",
    "TLT": "20+ Year Treasury",
    "TNX": "10Y Treasury Yield",
    "IRX": "3M T-Bill Rate",
    "VIX": "VIX",
    "DX_Y_NYB": "US Dollar Index",
    "BTC_USD": "Bitcoin",
    "USO": "Oil",
    
    # US Rates (new)
    "SHY": "1-3Y Treasury ETF",
    "FVX": "5Y Treasury Yield",
    "TLH": "10-20Y Treasury ETF", 
    "TYX": "30Y Treasury Yield",
    
    # Global Rates
    "IGLT_L": "UK Gilts ETF",
    "IBGL_L": "Germany Bund ETF", 
    "BNDX": "Intl Bond ex-US",
    "EMB": "Emerging Market Bonds",
    "BWX": "Intl Treasury Bond",
    "IGOV": "Intl Govt Bond ETF",
    
    # Global Equity Indices
    "FTSE": "UK FTSE 100",
    "GDAXI": "Germany DAX",
    "N225": "Japan Nikkei 225",
    "000001_SS": "China Shanghai Composite",
    "FCHI": "France CAC 40",
    "STOXX50E": "Euro Stoxx 50",
    "HSI": "Hong Kong Hang Seng",
    "BSESN": "India BSE Sensex",
    "BVSP": "Brazil Bovespa",
    "KS11": "South Korea KOSPI",
    
    # Country ETFs
    "EWC": "Canada ETF",
    "EWA": "Australia ETF", 
    "EWW": "Mexico ETF",
    "EWT": "Taiwan ETF",
    "EIDO": "Indonesia ETF",
    "TUR": "Turkey ETF",
    "EZA": "South Africa ETF",
    "KSA": "Saudi Arabia ETF",
    "EWL": "Switzerland ETF",
    "EWS": "Singapore ETF",
    
    # Additional Commodities
    "NG_F": "Natural Gas",
}


def load_benchmarks() -> pd.DataFrame:
    """Load previously saved benchmark data."""
    return pd.read_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    df = fetch_benchmarks()
    save_benchmarks(df)
