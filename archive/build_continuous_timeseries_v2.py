#!/usr/bin/env python3
"""
Build Continuous Event Time Series v2.0 - MUCH Better Coverage

This version fixes the major issues with v1:
- Uses event_slug as natural grouping (not just multi-expiry events)
- Filters out sports/entertainment using classifications
- Lowers minimum threshold to 30 days (not 60)
- Processes ALL events, not just top 50
- Better candle data loading and error handling
- Smarter rollover logic for contract chaining

Goal: Find HUNDREDS of continuous time series, not just 4.

Author: OpenClaw Assistant
Date: 2025-02-23
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class ContinuousTimeseriesBuilderV2:
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.markets_df = None
        self.classifications_df = None
        self.candles_cache = {}
        self.continuous_series = {}
        self.filtered_markets = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load markets and classifications data"""
        print("📊 Loading markets and classifications data...")
        
        # Load markets
        self.markets_df = pd.read_parquet(f'{self.data_dir}/processed/markets.parquet')
        
        # Load classifications
        self.classifications_df = pd.read_parquet(f'{self.data_dir}/processed/market_classifications.parquet')
        
        # Convert date columns to datetime
        date_cols = ['start_date', 'end_date', 'created_date', 'resolution_date', 'active_start', 'active_end']
        for col in date_cols:
            if col in self.markets_df.columns:
                self.markets_df[col] = pd.to_datetime(self.markets_df[col], errors='coerce')
        
        print(f"✅ Loaded {len(self.markets_df)} markets")
        print(f"✅ Loaded {len(self.classifications_df)} classifications")
        
        return self.markets_df, self.classifications_df
        
    def filter_relevant_markets(self, exclude_categories: List[str] = ['sports', 'entertainment']) -> pd.DataFrame:
        """Filter out unwanted categories and select relevant markets"""
        print(f"🔍 Filtering markets (excluding: {exclude_categories})...")
        
        # Merge markets with classifications
        merged_df = self.markets_df.merge(
            self.classifications_df[['market_id', 'category']], 
            on='market_id', 
            how='left'
        )
        
        # Filter out unwanted categories
        before_count = len(merged_df)
        filtered_df = merged_df[~merged_df['category'].isin(exclude_categories)].copy()
        after_count = len(filtered_df)
        
        print(f"📉 Filtered from {before_count:,} to {after_count:,} markets ({before_count-after_count:,} excluded)")
        
        # Show category breakdown
        category_counts = filtered_df['category'].value_counts()
        print("📊 Remaining categories:")
        for cat, count in category_counts.head(10).items():
            print(f"   {cat}: {count:,} markets")
        
        self.filtered_markets = filtered_df
        return filtered_df
        
    def analyze_event_coverage(self) -> Dict[str, Dict]:
        """Analyze event_slug coverage and potential for continuous series"""
        print("🔍 Analyzing event coverage for continuous series...")
        
        event_analysis = {}
        
        # Group by event_slug
        for event_slug, group in self.filtered_markets.groupby('event_slug'):
            # Calculate basic metrics
            market_count = len(group)
            unique_end_dates = group['end_date'].nunique()
            
            # Date range analysis
            min_start = group['start_date'].min()
            max_end = group['end_date'].max()
            date_span_days = (max_end - min_start).days if pd.notna(min_start) and pd.notna(max_end) else 0
            
            # Active date analysis (if available)
            active_start_range = group['active_start'].min() if 'active_start' in group.columns else None
            active_end_range = group['active_end'].max() if 'active_end' in group.columns else None
            active_span_days = 0
            if pd.notna(active_start_range) and pd.notna(active_end_range):
                active_span_days = (active_end_range - active_start_range).days
            
            # Check for potential candle data
            condition_ids = group['condition_id'].dropna().tolist()
            tickers = group['ticker'].dropna().tolist()
            
            event_analysis[event_slug] = {
                'market_count': market_count,
                'unique_end_dates': unique_end_dates,
                'date_span_days': date_span_days,
                'active_span_days': active_span_days,
                'min_start_date': min_start,
                'max_end_date': max_end,
                'condition_ids': condition_ids,
                'tickers': tickers,
                'categories': group['category'].unique().tolist(),
                'platforms': group['platform'].unique().tolist(),
                'has_multiple_contracts': unique_end_dates > 1,
                'potential_score': self._calculate_potential_score(market_count, unique_end_dates, date_span_days, active_span_days)
            }
        
        # Sort by potential score
        sorted_events = sorted(event_analysis.items(), key=lambda x: x[1]['potential_score'], reverse=True)
        
        print(f"🎯 Analyzed {len(event_analysis)} events")
        print("🏆 Top 20 events by potential:")
        for i, (event_slug, info) in enumerate(sorted_events[:20]):
            print(f"   {i+1:2d}. {event_slug[:60]:<60} | {info['market_count']:3d} markets | {info['date_span_days']:3d} days | Score: {info['potential_score']:.1f}")
        
        return dict(sorted_events)
        
    def _calculate_potential_score(self, market_count: int, unique_end_dates: int, date_span_days: int, active_span_days: int) -> float:
        """Calculate potential score for building continuous series"""
        score = 0
        
        # Base score from market count
        score += min(market_count * 2, 20)
        
        # Bonus for multiple end dates (rollover potential)
        if unique_end_dates > 1:
            score += min(unique_end_dates * 3, 15)
        
        # Bonus for longer date spans
        if date_span_days > 30:
            score += min(date_span_days / 10, 20)
        
        # Bonus for active span
        if active_span_days > 0:
            score += min(active_span_days / 10, 15)
        
        return score
    
    def load_candles_data(self, condition_id: str, platform: str = 'polymarket') -> Optional[List[Dict]]:
        """Load price/candles data for a specific condition_id or ticker"""
        cache_key = f"{platform}_{condition_id}"
        if cache_key in self.candles_cache:
            return self.candles_cache[cache_key]
            
        if pd.isna(condition_id):
            return None
        
        # Try Polymarket first
        if platform == 'polymarket':
            candles_file = f'{self.data_dir}/raw/polymarket/candles_{condition_id}.json'
        elif platform == 'kalshi':
            candles_file = f'{self.data_dir}/raw/kalshi/trades_{condition_id}.json'
        else:
            return None
        
        if not os.path.exists(candles_file):
            return None
            
        try:
            with open(candles_file, 'r') as f:
                data = json.load(f)
                
            candles = None
            
            # Handle different JSON structures
            if isinstance(data, list) and len(data) >= 1:
                # Polymarket format: first element is the time series
                if isinstance(data[0], list):
                    candles = data[0]
                else:
                    candles = data
            elif isinstance(data, dict):
                # Kalshi format or other dict-based formats
                if 'data' in data:
                    candles = data['data']
                else:
                    candles = data
                    
            if candles and isinstance(candles, list) and len(candles) > 0:
                self.candles_cache[cache_key] = candles
                return candles
                
        except Exception as e:
            # Don't spam errors, just note missing files
            pass
            
        return None
    
    def process_candles_to_df(self, candles: List[Dict], platform: str = 'polymarket') -> pd.DataFrame:
        """Convert candles data to pandas DataFrame"""
        if not candles:
            return pd.DataFrame()
            
        records = []
        for candle in candles:
            try:
                if platform == 'polymarket':
                    # Polymarket format
                    if 'end_period_ts' in candle and 'price' in candle:
                        price_val = candle['price']
                        if isinstance(price_val, dict) and 'close_dollars' in price_val:
                            price = float(price_val['close_dollars'])
                        elif isinstance(price_val, (int, float)):
                            price = float(price_val)
                        else:
                            continue
                            
                        record = {
                            'timestamp': pd.to_datetime(candle['end_period_ts'], unit='s'),
                            'price': price,
                            'volume': candle.get('volume', 0),
                            'open_interest': candle.get('open_interest', 0)
                        }
                        records.append(record)
                        
                elif platform == 'kalshi':
                    # Kalshi format - adapt as needed
                    if 'timestamp' in candle and 'price' in candle:
                        record = {
                            'timestamp': pd.to_datetime(candle['timestamp']),
                            'price': float(candle['price']),
                            'volume': candle.get('volume', 0),
                            'open_interest': candle.get('open_interest', 0)
                        }
                        records.append(record)
                        
            except Exception as e:
                # Skip malformed records
                continue
                
        if records:
            df = pd.DataFrame(records).sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'])  # Remove duplicate timestamps
            return df
        
        return pd.DataFrame()
    
    def build_continuous_series_for_event(self, event_slug: str, event_info: Dict) -> Optional[pd.DataFrame]:
        """Build continuous time series for a specific event - MUCH more flexible approach"""
        print(f"🔗 Building continuous series for: {event_slug}")
        
        # Get all markets for this event
        event_markets = self.filtered_markets[self.filtered_markets['event_slug'] == event_slug].copy()
        
        if len(event_markets) == 0:
            return None
        
        print(f"   Found {len(event_markets)} markets")
        
        # Load price data for each market
        all_price_data = []
        markets_with_data = 0
        
        for idx, row in event_markets.iterrows():
            condition_id = row.get('condition_id')
            ticker = row.get('ticker')
            platform = row.get('platform', 'polymarket')
            end_date = row.get('end_date')
            
            # Try to load data using condition_id or ticker
            data_identifier = condition_id if pd.notna(condition_id) else ticker
            if pd.isna(data_identifier):
                continue
                
            candles = self.load_candles_data(str(data_identifier), platform)
            if candles:
                df = self.process_candles_to_df(candles, platform)
                if len(df) > 0:
                    # Add market metadata
                    df['market_id'] = row['market_id']
                    df['condition_id'] = condition_id
                    df['ticker'] = ticker
                    df['end_date'] = end_date
                    df['platform'] = platform
                    df['market_title'] = row.get('title', '')
                    
                    all_price_data.append(df)
                    markets_with_data += 1
                    
                    duration = (df['timestamp'].max() - df['timestamp'].min()).days
                    print(f"   ✅ Loaded {len(df)} points for market {row['market_id']} ({duration} days)")
        
        print(f"   📊 {markets_with_data}/{len(event_markets)} markets have price data")
        
        if len(all_price_data) == 0:
            print("   ❌ No price data found")
            return None
        
        # Strategy 1: If we have multiple markets with different end dates, chain them
        if len(all_price_data) > 1 and event_info['has_multiple_contracts']:
            continuous_df = self._chain_multiple_contracts(all_price_data)
            if continuous_df is not None:
                return continuous_df
        
        # Strategy 2: If we have one long-running market or failed chaining, use the longest single series
        longest_series = max(all_price_data, key=len)
        longest_series['active_contract'] = longest_series['market_id'].iloc[0]
        longest_series['contract_end_date'] = longest_series['end_date'].iloc[0]
        
        duration = (longest_series['timestamp'].max() - longest_series['timestamp'].min()).days
        print(f"   📈 Using single longest series: {len(longest_series)} points over {duration} days")
        
        return longest_series
    
    def _chain_multiple_contracts(self, price_data_list: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Chain multiple contracts together chronologically"""
        print(f"   🔗 Chaining {len(price_data_list)} contracts...")
        
        # Sort contracts by their end dates
        price_data_list.sort(key=lambda df: df['end_date'].iloc[0] if pd.notna(df['end_date'].iloc[0]) else datetime.max)
        
        continuous_data = []
        last_timestamp = None
        
        for i, df in enumerate(price_data_list):
            df_copy = df.copy()
            
            # Add contract metadata
            df_copy['active_contract'] = df_copy['market_id'].iloc[0]
            df_copy['contract_end_date'] = df_copy['end_date'].iloc[0]
            
            # For non-final contracts, only use data up to end date (with some buffer)
            if i < len(price_data_list) - 1:
                end_date = df_copy['end_date'].iloc[0]
                if pd.notna(end_date):
                    # Use data up to end date + 1 day buffer
                    cutoff_date = end_date + timedelta(days=1)
                    df_copy = df_copy[df_copy['timestamp'] <= cutoff_date]
            
            # Handle overlaps: start where previous contract ended
            if last_timestamp is not None:
                df_copy = df_copy[df_copy['timestamp'] > last_timestamp]
            
            if len(df_copy) > 0:
                continuous_data.append(df_copy)
                last_timestamp = df_copy['timestamp'].max()
                print(f"     Contract {i+1}: {len(df_copy)} points")
        
        if continuous_data:
            result_df = pd.concat(continuous_data, ignore_index=True)
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)
            
            duration = (result_df['timestamp'].max() - result_df['timestamp'].min()).days
            print(f"   ✅ Chained series: {len(result_df)} points over {duration} days")
            
            return result_df
        
        return None
    
    def filter_by_duration(self, event_analysis: Dict[str, Dict], min_days: int = 30, max_events: int = None) -> Dict[str, pd.DataFrame]:
        """Build continuous series for events and filter by duration"""
        print(f"⏱️  Building continuous series (minimum {min_days} days)...")
        
        qualified_events = {}
        events_processed = 0
        events_to_process = list(event_analysis.keys()) if max_events is None else list(event_analysis.keys())[:max_events]
        
        for event_slug in events_to_process:
            events_processed += 1
            if events_processed % 50 == 0:
                print(f"   📊 Processed {events_processed}/{len(events_to_process)} events, found {len(qualified_events)} qualifying...")
            
            event_info = event_analysis[event_slug]
            
            # Skip if very low potential
            if event_info['potential_score'] < 5:
                continue
            
            continuous_series = self.build_continuous_series_for_event(event_slug, event_info)
            
            if continuous_series is not None and len(continuous_series) > 0:
                # Calculate actual duration from the data
                duration = (continuous_series['timestamp'].max() - continuous_series['timestamp'].min()).days
                
                if duration >= min_days:
                    qualified_events[event_slug] = continuous_series
                    print(f"   ✅ {event_slug}: {duration} days, {len(continuous_series)} points")
                else:
                    print(f"   ❌ {event_slug}: only {duration} days (< {min_days})")
        
        print(f"\n🎯 FINAL RESULTS: Found {len(qualified_events)} events with {min_days}+ days of continuous history!")
        
        return qualified_events
    
    def save_results(self, qualified_events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Save the continuous time series dataset and summary"""
        print("💾 Saving continuous time series dataset...")
        
        all_series = []
        
        for event_slug, df in qualified_events.items():
            df_copy = df.copy()
            df_copy['event_slug'] = event_slug
            all_series.append(df_copy)
        
        if all_series:
            combined_df = pd.concat(all_series, ignore_index=True)
            
            # Save main dataset
            output_file = f'{self.data_dir}/processed/continuous_event_timeseries_v2.parquet'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_parquet(output_file, index=False)
            
            print(f"✅ Saved {len(combined_df):,} rows covering {len(qualified_events)} events to {output_file}")
            
            # Create detailed summary
            summary = {
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'total_events': len(qualified_events),
                'total_data_points': len(combined_df),
                'date_range': {
                    'start': combined_df['timestamp'].min().isoformat(),
                    'end': combined_df['timestamp'].max().isoformat(),
                    'span_days': (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
                },
                'summary_stats': {
                    'avg_points_per_event': len(combined_df) / len(qualified_events),
                    'median_duration_days': combined_df.groupby('event_slug').apply(
                        lambda x: (x['timestamp'].max() - x['timestamp'].min()).days
                    ).median(),
                    'avg_price': combined_df['price'].mean(),
                    'price_volatility': combined_df['price'].std()
                },
                'events': {}
            }
            
            # Add per-event details
            for event_slug, df in qualified_events.items():
                duration = (df['timestamp'].max() - df['timestamp'].min()).days
                summary['events'][event_slug] = {
                    'data_points': len(df),
                    'duration_days': duration,
                    'start_date': df['timestamp'].min().isoformat(),
                    'end_date': df['timestamp'].max().isoformat(),
                    'contracts_used': df['active_contract'].nunique() if 'active_contract' in df.columns else 1,
                    'avg_price': df['price'].mean(),
                    'price_volatility': df['price'].std(),
                    'categories': df.get('category', []).unique().tolist() if 'category' in df.columns else [],
                    'platforms': df['platform'].unique().tolist() if 'platform' in df.columns else []
                }
            
            # Save summary
            summary_file = output_file.replace('.parquet', '_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Saved detailed summary to {summary_file}")
            
            return combined_df
        
        return None
    
    def create_sample_charts(self, qualified_events: Dict[str, pd.DataFrame], max_charts: int = 15):
        """Create sample charts for the best continuous series"""
        print("📈 Creating sample charts...")
        
        output_dir = f'{self.data_dir}/outputs/continuous_timeseries_v2_charts'
        os.makedirs(output_dir, exist_ok=True)
        
        # Select events for charting (best ones by data points and duration)
        events_for_charts = sorted(
            qualified_events.items(), 
            key=lambda x: len(x[1]) * ((x[1]['timestamp'].max() - x[1]['timestamp'].min()).days),
            reverse=True
        )[:max_charts]
        
        # Create overview chart
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        axes = axes.flatten()
        
        for i, (event_slug, df) in enumerate(events_for_charts):
            if i >= 15:
                break
                
            ax = axes[i]
            
            # Plot price over time
            ax.plot(df['timestamp'], df['price'], linewidth=1.5, alpha=0.8)
            
            # Add contract transition lines if available
            if 'active_contract' in df.columns and df['active_contract'].nunique() > 1:
                for contract in df['active_contract'].unique()[1:]:  # Skip first
                    transition_time = df[df['active_contract'] == contract]['timestamp'].min()
                    ax.axvline(x=transition_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Formatting
            title = event_slug.replace('-', ' ').title()[:50]
            duration = (df['timestamp'].max() - df['timestamp'].min()).days
            ax.set_title(f"{title}\n({len(df)} pts, {duration} days)", fontsize=10)
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.set_ylim(0, 1)
            
            # Format y-axis as percentage
            ax.set_yticklabels([f'{int(x*100)}%' for x in ax.get_yticks()])
        
        # Remove unused subplots
        for i in range(len(events_for_charts), 15):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle("Continuous Event Time Series v2.0 - Sample Events", fontsize=18, y=0.98)
        
        chart_file = os.path.join(output_dir, 'overview_continuous_timeseries_v2.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved overview chart to {chart_file}")
        plt.close()
        
        # Create detailed charts for top 5 events
        for i, (event_slug, df) in enumerate(events_for_charts[:5]):
            self._create_detailed_event_chart(event_slug, df, output_dir)
    
    def _create_detailed_event_chart(self, event_slug: str, df: pd.DataFrame, output_dir: str):
        """Create detailed chart for a single event"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Price chart
        ax1.plot(df['timestamp'], df['price'], linewidth=2, alpha=0.8)
        
        # Color by contract if multiple contracts
        if 'active_contract' in df.columns and df['active_contract'].nunique() > 1:
            contracts = df['active_contract'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(contracts)))
            
            for j, contract in enumerate(contracts):
                contract_data = df[df['active_contract'] == contract]
                ax1.plot(contract_data['timestamp'], contract_data['price'], 
                        color=colors[j], linewidth=2, alpha=0.8, label=f'Contract {j+1}')
                
                # Add transition line
                if j > 0:
                    ax1.axvline(x=contract_data['timestamp'].min(), color='red', linestyle='--', alpha=0.7)
            
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        title = event_slug.replace('-', ' ').title()
        duration = (df['timestamp'].max() - df['timestamp'].min()).days
        ax1.set_title(f"{title}\n{len(df):,} data points over {duration} days", fontsize=14)
        ax1.set_ylabel("Probability")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_yticklabels([f'{int(x*100)}%' for x in ax1.get_yticks()])
        
        # Volume chart
        if 'volume' in df.columns:
            ax2.bar(df['timestamp'], df['volume'], alpha=0.6, width=1)
            ax2.set_ylabel("Volume")
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Date")
        
        plt.tight_layout()
        
        # Save chart
        safe_filename = event_slug.replace('/', '_').replace('\\', '_')
        chart_file = os.path.join(output_dir, f'{safe_filename}_detailed.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed chart saved: {safe_filename}")
        plt.close()


def main():
    print("🚀 Building Continuous Event Time Series v2.0 - MUCH Better Coverage!")
    print("=" * 80)
    
    # Initialize builder
    builder = ContinuousTimeseriesBuilderV2()
    
    # Load data
    builder.load_data()
    
    # Filter relevant markets (exclude sports & entertainment)
    builder.filter_relevant_markets(exclude_categories=['sports', 'entertainment'])
    
    # Analyze event coverage and potential
    event_analysis = builder.analyze_event_coverage()
    
    # Build continuous series (30 day minimum, process more events)
    qualified_events = builder.filter_by_duration(event_analysis, min_days=30, max_events=None)
    
    if not qualified_events:
        print("❌ No qualifying continuous time series found!")
        return
    
    # Save results
    combined_df = builder.save_results(qualified_events)
    
    # Create charts
    builder.create_sample_charts(qualified_events)
    
    # Final summary
    print(f"\n🎉 SUCCESS! Built {len(qualified_events)} continuous time series")
    if combined_df is not None:
        print(f"📊 Total data points: {len(combined_df):,}")
        print(f"📅 Date range: {combined_df['timestamp'].min().strftime('%Y-%m-%d')} to {combined_df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # Show top events by data points
        event_sizes = combined_df.groupby('event_slug').size().sort_values(ascending=False)
        print(f"\n🏆 Top 10 events by data points:")
        for i, (event_slug, size) in enumerate(event_sizes.head(10).items()):
            duration = (combined_df[combined_df['event_slug'] == event_slug]['timestamp'].max() - 
                       combined_df[combined_df['event_slug'] == event_slug]['timestamp'].min()).days
            print(f"   {i+1:2d}. {event_slug[:50]:<50} | {size:4d} points | {duration:3d} days")


if __name__ == "__main__":
    main()