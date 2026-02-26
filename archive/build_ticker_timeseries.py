#!/usr/bin/env python3
"""
Build continuous Ticker-level time series from CUSIP chains.

Implementation of the methodology from RESEARCH-timeseries.md Section 11.
Creates TWO time series for each rollable Ticker (2+ CUSIPs):

1. Raw Levels (front-month rolling) - with price jumps at roll points
2. Return-Chained (adjusted) - continuous for correlation analysis

Author: Claude subagent
Date: February 23, 2026
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ticker_timeseries_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TickerTimeseriesBuilder:
    """Build continuous time series for rollable tickers."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.outputs_dir = self.data_dir / "outputs"
        
        # Create outputs directory
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'total_tickers': 0,
            'rollable_tickers': 0,
            'processed_successfully': 0,
            'failed_no_data': 0,
            'failed_insufficient_duration': 0,
            'failed_other': 0
        }
        
        # Cache for loaded candle data
        self.candle_cache = {}
        
    def load_data_files(self):
        """Load ticker chains, mapping, and markets metadata."""
        logger.info("Loading data files...")
        
        # Load ticker chains
        with open(self.processed_dir / "ticker_chains.json", 'r') as f:
            self.ticker_chains = json.load(f)
        
        # Load ticker mapping
        self.ticker_mapping = pd.read_parquet(self.processed_dir / "ticker_mapping.parquet")
        
        # Load markets metadata  
        self.markets = pd.read_parquet(self.processed_dir / "markets.parquet")
        
        logger.info(f"Loaded {len(self.ticker_chains)} tickers, {len(self.ticker_mapping)} mappings, {len(self.markets)} markets")
        
    def load_candle_data(self, market_id: str, platform: str, condition_id: str = None, ticker: str = None) -> Optional[pd.DataFrame]:
        """Load and cache candle/trade data for a market."""
        # Use cache if available
        cache_key = market_id
        if cache_key in self.candle_cache:
            return self.candle_cache[cache_key]
        
        df = None
        try:
            if platform == 'polymarket':
                if not condition_id:
                    logger.warning(f"No condition_id for polymarket market {market_id}")
                    return None
                
                file_path = self.raw_dir / "polymarket" / f"candles_{condition_id}.json"
                if not file_path.exists():
                    logger.warning(f"Candle file not found: {file_path}")
                    return None
                
                with open(file_path, 'r') as f:
                    candles = json.load(f)
                
                if not candles:
                    logger.warning(f"Empty candle data for {market_id}")
                    return None
                
                # Handle nested format: [candles_array, token_metadata]
                if isinstance(candles, list) and len(candles) >= 1 and isinstance(candles[0], list):
                    candles = candles[0]
                
                if not candles:
                    logger.warning(f"Empty candle data for {market_id}")
                    return None
                
                # Convert to DataFrame
                rows = []
                for candle in candles:
                    rows.append({
                        'timestamp': pd.to_datetime(candle['end_period_ts'], unit='s'),
                        'price': float(candle['price']['close_dollars']),
                        'volume': float(candle.get('volume', 0)),
                        'market_id': market_id
                    })
                
                df = pd.DataFrame(rows)
                
            elif platform == 'kalshi':
                if not ticker:
                    logger.warning(f"No ticker for kalshi market {market_id}")
                    return None
                
                file_path = self.raw_dir / "kalshi" / f"trades_{ticker}.json"
                if not file_path.exists():
                    logger.warning(f"Trade file not found: {file_path}")
                    return None
                
                with open(file_path, 'r') as f:
                    trades = json.load(f)
                
                if not trades:
                    logger.warning(f"Empty trade data for {market_id}")
                    return None
                
                # Convert trades to daily aggregated data
                rows = []
                for trade in trades:
                    rows.append({
                        'timestamp': pd.to_datetime(trade['created_time'], unit='s'),
                        'price': float(trade['yes_price_dollars']),
                        'volume': float(trade['count']),
                        'market_id': market_id
                    })
                
                trade_df = pd.DataFrame(rows)
                
                # Aggregate to daily OHLCV (using close price)
                trade_df['date'] = trade_df['timestamp'].dt.date
                daily_df = trade_df.groupby('date').agg({
                    'price': 'last',  # Use last price as close
                    'volume': 'sum',
                    'market_id': 'first'
                }).reset_index()
                
                daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
                df = daily_df[['timestamp', 'price', 'volume', 'market_id']]
            
            if df is not None and len(df) > 0:
                # Sort by timestamp and remove duplicates
                df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                df = df.reset_index(drop=True)
                
                # Cache the result
                self.candle_cache[cache_key] = df
                
        except Exception as e:
            logger.warning(f"Error loading candle data for {market_id}: {e}")
            
        return df
    
    def get_market_metadata(self, market_id: str) -> Optional[Dict]:
        """Get market metadata for a given market_id."""
        market_row = self.markets[self.markets['market_id'] == market_id]
        if len(market_row) == 0:
            return None
        
        return {
            'platform': market_row.iloc[0]['platform'],
            'condition_id': market_row.iloc[0].get('condition_id'),
            'ticker': market_row.iloc[0].get('ticker'),
            'end_date': market_row.iloc[0]['end_date'],
            'title': market_row.iloc[0]['title']
        }
    
    def build_ticker_timeseries(self, ticker_id: str, ticker_data: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Build raw and adjusted time series for a ticker.
        
        Returns:
            (raw_series, adjusted_series) or (None, None) if failed
        """
        logger.info(f"Processing ticker {ticker_id}: {ticker_data['ticker_name']}")
        
        # Sort markets by end_date for proper rolling
        markets = sorted(ticker_data['markets'], key=lambda x: x['end_date'])
        
        # Load candle data for all markets
        market_data = []
        for market in markets:
            metadata = self.get_market_metadata(market['market_id'])
            if not metadata:
                logger.warning(f"No metadata found for market {market['market_id']}")
                continue
            
            candles = self.load_candle_data(
                market['market_id'],
                metadata['platform'],
                metadata['condition_id'],
                metadata['ticker']
            )
            
            if candles is not None and len(candles) >= 1:  # At least 1 data point
                market_data.append({
                    'market_id': market['market_id'],
                    'end_date': pd.to_datetime(market['end_date']),
                    'candles': candles,
                    'title': metadata['title']
                })
        
        if len(market_data) == 0:
            logger.warning(f"No valid candle data found for ticker {ticker_id}")
            self.stats['failed_no_data'] += 1
            return None, None
        
        # Build continuous time series using front-month rolling
        raw_series = self._build_raw_series(market_data, ticker_id)
        if raw_series is None or len(raw_series) < 10:  # Less than 10 days
            logger.warning(f"Insufficient data for ticker {ticker_id}: {len(raw_series) if raw_series is not None else 0} days")
            self.stats['failed_insufficient_duration'] += 1
            return None, None
        
        # Build return-chained adjusted series
        adjusted_series = self._build_adjusted_series(raw_series)
        
        logger.info(f"Successfully built timeseries for {ticker_id}: {len(raw_series)} data points")
        self.stats['processed_successfully'] += 1
        
        return raw_series, adjusted_series
    
    def _build_raw_series(self, market_data: List[Dict], ticker_id: str) -> Optional[pd.DataFrame]:
        """Build raw levels series with front-month rolling."""
        all_dates = set()
        
        # Collect all unique dates
        for market in market_data:
            all_dates.update(market['candles']['timestamp'].dt.date)
        
        # Sort dates
        date_range = sorted(all_dates)
        
        if len(date_range) == 0:
            return None
        
        # Build the time series day by day
        series_data = []
        
        for date in date_range:
            date_pd = pd.Timestamp(date)
            
            # Find the active contract for this date (nearest expiry that hasn't expired yet)
            active_market = None
            min_days_to_expiry = float('inf')
            
            for market in market_data:
                days_to_expiry = (market['end_date'] - date_pd).days
                
                # Use contract if it hasn't expired yet (allow 1 day grace period)
                if days_to_expiry >= -1 and days_to_expiry < min_days_to_expiry:
                    # Check if we have data for this date
                    market_candles = market['candles']
                    date_mask = market_candles['timestamp'].dt.date == date
                    
                    if date_mask.any():
                        min_days_to_expiry = days_to_expiry
                        active_market = market
            
            if active_market is not None:
                # Get the price for this date
                market_candles = active_market['candles']
                date_mask = market_candles['timestamp'].dt.date == date
                date_data = market_candles[date_mask].iloc[0]
                
                # Determine if this is a roll point
                is_roll_point = False
                if len(series_data) > 0:
                    # Check if we switched contracts
                    prev_cusip = series_data[-1]['active_cusip']
                    current_cusip = active_market['market_id']
                    is_roll_point = (prev_cusip != current_cusip)
                
                series_data.append({
                    'date': date_pd,
                    'price': date_data['price'],
                    'volume': date_data['volume'],
                    'active_cusip': active_market['market_id'],
                    'cusip_end_date': active_market['end_date'],
                    'is_roll_point': is_roll_point,
                    'ticker_id': ticker_id
                })
        
        if len(series_data) == 0:
            return None
        
        return pd.DataFrame(series_data)
    
    def _build_adjusted_series(self, raw_series: pd.DataFrame) -> pd.DataFrame:
        """Build return-chained adjusted series for correlation analysis."""
        if len(raw_series) == 0:
            return pd.DataFrame()
        
        adjusted_data = []
        cumulative_adjustment = 0
        
        for i, row in raw_series.iterrows():
            if i == 0:
                # First observation - use original price as starting point
                adjusted_price = row['price']
                daily_return = 0
            else:
                prev_row = raw_series.iloc[i-1]
                
                if row['is_roll_point']:
                    # At roll points, set daily return to 0 to avoid artificial jumps
                    daily_return = 0
                else:
                    # Normal daily return (probability change, not percentage change)
                    daily_return = row['price'] - prev_row['price']
                
                # Build adjusted price by cumulating returns from arbitrary starting point
                adjusted_price = raw_series.iloc[0]['price'] + cumulative_adjustment + daily_return
                cumulative_adjustment += daily_return
            
            adjusted_data.append({
                'date': row['date'],
                'price_raw': row['price'],
                'price_adjusted': adjusted_price,
                'daily_return': daily_return,
                'volume': row['volume'],
                'active_cusip': row['active_cusip'],
                'cusip_end_date': row['cusip_end_date'],
                'is_roll_point': row['is_roll_point'],
                'ticker_id': row['ticker_id']
            })
        
        return pd.DataFrame(adjusted_data)
    
    def process_all_tickers(self):
        """Process all rollable tickers and build time series."""
        logger.info("Starting ticker timeseries processing...")
        
        # Process ALL tickers (including single-contract)
        rollable_tickers = {
            tid: data for tid, data in self.ticker_chains.items() 
            if data['market_count'] >= 1
        }
        
        self.stats['total_tickers'] = len(self.ticker_chains)
        self.stats['rollable_tickers'] = len(rollable_tickers)
        
        logger.info(f"Processing {len(rollable_tickers)} tickers out of {len(self.ticker_chains)} total")
        
        raw_series_list = []
        adjusted_series_list = []
        
        # Process each ticker
        for ticker_id, ticker_data in tqdm(rollable_tickers.items(), desc="Processing tickers"):
            try:
                raw_series, adjusted_series = self.build_ticker_timeseries(ticker_id, ticker_data)
                
                if raw_series is not None:
                    raw_series_list.append(raw_series)
                    adjusted_series_list.append(adjusted_series)
                    
            except Exception as e:
                logger.error(f"Error processing ticker {ticker_id}: {e}")
                self.stats['failed_other'] += 1
        
        # Combine all series
        if raw_series_list:
            self.raw_timeseries = pd.concat(raw_series_list, ignore_index=True)
            self.adjusted_timeseries = pd.concat(adjusted_series_list, ignore_index=True)
            
            logger.info(f"Built timeseries for {len(raw_series_list)} tickers")
            logger.info(f"Raw series: {len(self.raw_timeseries)} total data points")
            logger.info(f"Adjusted series: {len(self.adjusted_timeseries)} total data points")
        else:
            logger.warning("No timeseries data was successfully built")
            self.raw_timeseries = pd.DataFrame()
            self.adjusted_timeseries = pd.DataFrame()
    
    def save_results(self):
        """Save the timeseries data and statistics."""
        logger.info("Saving results...")
        
        # Save raw timeseries
        if len(self.raw_timeseries) > 0:
            raw_path = self.processed_dir / "ticker_timeseries_raw.parquet"
            self.raw_timeseries.to_parquet(raw_path, index=False)
            logger.info(f"Saved raw timeseries to {raw_path}")
            
            # Save adjusted timeseries
            adjusted_path = self.processed_dir / "ticker_timeseries_adjusted.parquet"
            self.adjusted_timeseries.to_parquet(adjusted_path, index=False)
            logger.info(f"Saved adjusted timeseries to {adjusted_path}")
        
        # Calculate and save statistics
        stats_data = self.stats.copy()
        
        if len(self.raw_timeseries) > 0:
            # Add detailed statistics
            ticker_counts = self.raw_timeseries.groupby('ticker_id').size()
            
            stats_data.update({
                'unique_tickers_with_data': len(ticker_counts),
                'total_data_points': len(self.raw_timeseries),
                'avg_points_per_ticker': ticker_counts.mean(),
                'median_points_per_ticker': ticker_counts.median(),
                'min_points_per_ticker': ticker_counts.min(),
                'max_points_per_ticker': ticker_counts.max(),
                'date_range_start': self.raw_timeseries['date'].min().strftime('%Y-%m-%d'),
                'date_range_end': self.raw_timeseries['date'].max().strftime('%Y-%m-%d'),
                'total_roll_points': self.raw_timeseries['is_roll_point'].sum()
            })
        
        # Save statistics
        stats_path = self.processed_dir / "ticker_timeseries_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)
        
        logger.info(f"Saved statistics to {stats_path}")
        
        # Print summary
        logger.info("Processing Summary:")
        logger.info(f"  Total tickers: {stats_data['total_tickers']}")
        logger.info(f"  Rollable tickers: {stats_data['rollable_tickers']}")  
        logger.info(f"  Successfully processed: {stats_data['processed_successfully']}")
        logger.info(f"  Failed (no data): {stats_data['failed_no_data']}")
        logger.info(f"  Failed (insufficient duration): {stats_data['failed_insufficient_duration']}")
        logger.info(f"  Failed (other): {stats_data['failed_other']}")
        
        if 'total_data_points' in stats_data:
            logger.info(f"  Total data points: {stats_data['total_data_points']}")
            logger.info(f"  Date range: {stats_data['date_range_start']} to {stats_data['date_range_end']}")
            logger.info(f"  Average points per ticker: {stats_data['avg_points_per_ticker']:.1f}")
    
    def create_sample_charts(self, num_charts: int = 8):
        """Create sample charts showing both raw and adjusted series with roll points."""
        if len(self.raw_timeseries) == 0:
            logger.warning("No timeseries data available for charting")
            return
        
        logger.info(f"Creating {num_charts} sample charts...")
        
        # Select interesting tickers (those with good data coverage and multiple rolls)
        ticker_stats = self.raw_timeseries.groupby('ticker_id').agg({
            'date': ['count', 'min', 'max'],
            'is_roll_point': 'sum'
        }).round(1)
        
        ticker_stats.columns = ['data_points', 'start_date', 'end_date', 'roll_points']
        ticker_stats = ticker_stats[ticker_stats['data_points'] >= 50]  # At least 50 data points
        ticker_stats = ticker_stats.sort_values(['roll_points', 'data_points'], ascending=[False, False])
        
        # Get ticker names
        ticker_names = {}
        for tid, data in self.ticker_chains.items():
            ticker_names[tid] = data['ticker_name']
        
        selected_tickers = ticker_stats.head(num_charts).index.tolist()
        
        # Create charts directory
        charts_dir = self.outputs_dir / "ticker_timeseries_charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Create individual charts
        for i, ticker_id in enumerate(selected_tickers):
            try:
                self._create_ticker_chart(ticker_id, ticker_names.get(ticker_id, ticker_id), charts_dir)
            except Exception as e:
                logger.error(f"Error creating chart for {ticker_id}: {e}")
        
        # Create summary chart
        self._create_summary_chart(selected_tickers[:6], ticker_names, charts_dir)
        
        logger.info(f"Charts saved to {charts_dir}")
    
    def _create_ticker_chart(self, ticker_id: str, ticker_name: str, output_dir: Path):
        """Create a chart for a single ticker showing raw vs adjusted series."""
        raw_data = self.raw_timeseries[self.raw_timeseries['ticker_id'] == ticker_id].copy()
        adj_data = self.adjusted_timeseries[self.adjusted_timeseries['ticker_id'] == ticker_id].copy()
        
        if len(raw_data) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Ticker Timeseries: {ticker_name}\n({ticker_id})', fontsize=14, fontweight='bold')
        
        # Plot 1: Raw levels with roll points
        ax1.plot(raw_data['date'], raw_data['price'], 'b-', linewidth=1.5, label='Raw Price')
        
        # Mark roll points
        roll_points = raw_data[raw_data['is_roll_point']]
        if len(roll_points) > 0:
            ax1.scatter(roll_points['date'], roll_points['price'], 
                       color='red', s=50, zorder=5, label=f'Roll Points ({len(roll_points)})')
        
        ax1.set_ylabel('Probability')
        ax1.set_title('Raw Levels (with price jumps at roll points)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Adjusted (return-chained) series
        ax2.plot(adj_data['date'], adj_data['price_adjusted'], 'g-', linewidth=1.5, label='Adjusted Price')
        
        # Mark roll points on adjusted series too
        adj_roll_points = adj_data[adj_data['is_roll_point']]
        if len(adj_roll_points) > 0:
            ax2.scatter(adj_roll_points['date'], adj_roll_points['price_adjusted'], 
                       color='red', s=50, zorder=5, label=f'Roll Points ({len(adj_roll_points)})')
        
        ax2.set_ylabel('Adjusted Probability')
        ax2.set_xlabel('Date')
        ax2.set_title('Return-Chained Adjusted (continuous for correlation)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add stats text
        duration_days = (raw_data['date'].max() - raw_data['date'].min()).days
        stats_text = f"Duration: {duration_days} days | Data points: {len(raw_data)} | Rolls: {raw_data['is_roll_point'].sum()}"
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.08)
        
        # Save
        safe_name = "".join(c for c in ticker_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"ticker_{ticker_id}_{safe_name[:50]}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_summary_chart(self, ticker_ids: List[str], ticker_names: Dict[str, str], output_dir: Path):
        """Create a summary chart showing multiple tickers."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, ticker_id in enumerate(ticker_ids):
            if i >= len(axes):
                break
            
            ax = axes[i]
            raw_data = self.raw_timeseries[self.raw_timeseries['ticker_id'] == ticker_id].copy()
            adj_data = self.adjusted_timeseries[self.adjusted_timeseries['ticker_id'] == ticker_id].copy()
            
            if len(raw_data) == 0:
                continue
            
            # Plot both raw and adjusted on same chart
            ax.plot(raw_data['date'], raw_data['price'], 'b-', alpha=0.7, linewidth=1, label='Raw')
            ax.plot(adj_data['date'], adj_data['price_adjusted'], 'g-', alpha=0.7, linewidth=1, label='Adjusted')
            
            # Mark roll points
            roll_points = raw_data[raw_data['is_roll_point']]
            if len(roll_points) > 0:
                ax.scatter(roll_points['date'], roll_points['price'], 
                          color='red', s=15, zorder=5, alpha=0.8)
            
            ticker_name = ticker_names.get(ticker_id, ticker_id)
            title = ticker_name[:40] + "..." if len(ticker_name) > 40 else ticker_name
            ax.set_title(f'{title}\n({len(raw_data)} pts, {raw_data["is_roll_point"].sum()} rolls)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(len(ticker_ids), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Ticker Timeseries Sample: Raw vs Return-Chained Adjusted', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        plt.savefig(output_dir / "ticker_timeseries_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    logger.info("Starting Ticker Timeseries Builder")
    
    builder = TickerTimeseriesBuilder()
    
    try:
        # Load data
        builder.load_data_files()
        
        # Process all tickers
        builder.process_all_tickers()
        
        # Save results
        builder.save_results()
        
        # Create sample charts
        builder.create_sample_charts()
        
        logger.info("Ticker timeseries building completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()