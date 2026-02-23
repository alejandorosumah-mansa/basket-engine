#!/usr/bin/env python3
"""
Build Continuous Event Time Series from Prediction Market Contracts

This script chains prediction market contracts with different expiration dates
to create continuous time series per event, similar to futures continuous contracts.

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

class ContinuousTimeseriesBuilder:
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.markets_df = None
        self.candles_cache = {}
        self.continuous_series = {}
        
    def load_markets(self) -> pd.DataFrame:
        """Load the markets metadata"""
        print("📊 Loading markets data...")
        self.markets_df = pd.read_parquet(f'{self.data_dir}/processed/markets.parquet')
        
        # Convert date columns to datetime
        date_cols = ['start_date', 'end_date', 'created_date', 'resolution_date']
        for col in date_cols:
            if col in self.markets_df.columns:
                self.markets_df[col] = pd.to_datetime(self.markets_df[col], errors='coerce')
        
        print(f"✅ Loaded {len(self.markets_df)} markets")
        return self.markets_df
        
    def analyze_multi_expiry_events(self) -> Dict[str, int]:
        """Find events with multiple expiration dates"""
        print("🔍 Analyzing events with multiple expiration dates...")
        
        # Group by event_slug and count unique end_dates
        event_expiries = self.markets_df.groupby('event_slug')['end_date'].nunique()
        multi_expiry_events = event_expiries[event_expiries > 1].sort_values(ascending=False)
        
        print(f"📈 Found {len(multi_expiry_events)} events with multiple expiration dates")
        print("🏆 Top 10 events with most expiration dates:")
        for event, count in multi_expiry_events.head(10).items():
            print(f"   {event}: {count} contracts")
            
        return multi_expiry_events
    
    def load_candles_data(self, condition_id: str) -> Optional[List[Dict]]:
        """Load price/candles data for a specific condition_id"""
        if condition_id in self.candles_cache:
            return self.candles_cache[condition_id]
            
        if pd.isna(condition_id):
            return None
            
        candles_file = f'{self.data_dir}/raw/polymarket/candles_{condition_id}.json'
        
        if not os.path.exists(candles_file):
            return None
            
        try:
            with open(candles_file, 'r') as f:
                data = json.load(f)
                
            # Extract the actual candles data (first element of the array)
            if isinstance(data, list) and len(data) >= 1:
                candles = data[0]  # The time series data is the first element
                if isinstance(candles, list):
                    self.candles_cache[condition_id] = candles
                    return candles
                    
        except Exception as e:
            print(f"❌ Error loading candles for {condition_id}: {e}")
            
        return None
    
    def process_candles_to_df(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert candles data to pandas DataFrame"""
        if not candles:
            return pd.DataFrame()
            
        records = []
        for candle in candles:
            if 'end_period_ts' in candle and 'price' in candle:
                record = {
                    'timestamp': pd.to_datetime(candle['end_period_ts'], unit='s'),
                    'price': float(candle['price']['close_dollars']),
                    'volume': candle.get('volume', 0),
                    'open_interest': candle.get('open_interest', 0)
                }
                records.append(record)
                
        if records:
            df = pd.DataFrame(records).sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'])  # Remove duplicate timestamps
            return df
        
        return pd.DataFrame()
    
    def build_continuous_series_for_event(self, event_slug: str) -> Optional[pd.DataFrame]:
        """Build continuous time series for a specific event"""
        print(f"🔗 Building continuous series for: {event_slug}")
        
        # Get all contracts for this event
        event_markets = self.markets_df[self.markets_df['event_slug'] == event_slug].copy()
        
        if len(event_markets) == 0:
            return None
            
        # Sort by end_date
        event_markets = event_markets.sort_values('end_date')
        
        print(f"   Found {len(event_markets)} contracts")
        
        # Load price data for each contract
        contract_series = {}
        contract_info = {}
        
        for idx, row in event_markets.iterrows():
            condition_id = row['condition_id']
            end_date = row['end_date']
            
            candles = self.load_candles_data(condition_id)
            if candles:
                df = self.process_candles_to_df(candles)
                if len(df) > 0:
                    contract_series[condition_id] = df
                    contract_info[condition_id] = {
                        'end_date': end_date,
                        'title': row['title'],
                        'status': row['status']
                    }
                    end_date_str = end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else 'Unknown'
                    print(f"   ✅ Loaded {len(df)} price points for contract ending {end_date_str}")
                else:
                    end_date_str = end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else 'Unknown'
                    print(f"   ⚠️  No valid price data for contract ending {end_date_str}")
            else:
                end_date_str = end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else 'Unknown'
                print(f"   ❌ No candles file for contract ending {end_date_str}")
        
        if len(contract_series) == 0:
            print("   ⚠️  No price data found for any contracts")
            return None
            
        # Chain contracts chronologically
        continuous_df = self.chain_contracts(contract_series, contract_info)
        
        if continuous_df is not None and len(continuous_df) > 0:
            print(f"   🎯 Created continuous series with {len(continuous_df)} data points")
            print(f"   📅 Date range: {continuous_df['timestamp'].min().strftime('%Y-%m-%d')} to {continuous_df['timestamp'].max().strftime('%Y-%m-%d')}")
            return continuous_df
        else:
            print("   ❌ Failed to create continuous series")
            return None
    
    def chain_contracts(self, contract_series: Dict[str, pd.DataFrame], 
                       contract_info: Dict[str, Dict]) -> Optional[pd.DataFrame]:
        """Chain multiple contracts together chronologically"""
        
        if len(contract_series) < 2:
            # If only one contract, return it as-is
            if len(contract_series) == 1:
                single_series = list(contract_series.values())[0].copy()
                single_series['active_contract'] = list(contract_series.keys())[0]
                return single_series
            else:
                return None
        
        # Sort contracts by end date
        sorted_contracts = sorted(
            contract_info.items(),
            key=lambda x: x[1]['end_date']
        )
        
        continuous_data = []
        
        for i, (contract_id, info) in enumerate(sorted_contracts):
            if contract_id not in contract_series:
                continue
                
            df = contract_series[contract_id].copy()
            end_date = info['end_date']
            
            # Add contract identifier
            df['active_contract'] = contract_id
            df['contract_end_date'] = end_date
            
            # For contracts that aren't the last one, only use data up to the end date
            if i < len(sorted_contracts) - 1:
                # Use data up to the contract's end date
                df = df[df['timestamp'] <= end_date]
            
            # Handle overlaps: if this isn't the first contract, start from where previous ended
            if i > 0 and len(continuous_data) > 0:
                last_timestamp = continuous_data[-1]['timestamp']
                df = df[df['timestamp'] > last_timestamp]
            
            # Add this contract's data
            for _, row in df.iterrows():
                continuous_data.append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row.get('volume', 0),
                    'open_interest': row.get('open_interest', 0),
                    'active_contract': row['active_contract'],
                    'contract_end_date': row['contract_end_date']
                })
        
        if continuous_data:
            continuous_df = pd.DataFrame(continuous_data)
            continuous_df = continuous_df.sort_values('timestamp').reset_index(drop=True)
            return continuous_df
        
        return None
    
    def filter_events_by_duration(self, min_days: int = 60) -> Dict[str, pd.DataFrame]:
        """Filter events that have at least min_days of continuous history"""
        print(f"🕒 Filtering events with at least {min_days} days of continuous history...")
        
        multi_expiry_events = self.analyze_multi_expiry_events()
        qualified_events = {}
        
        for event_slug in multi_expiry_events.index[:50]:  # Process top 50 to avoid too long processing
            continuous_series = self.build_continuous_series_for_event(event_slug)
            
            if continuous_series is not None and len(continuous_series) > 0:
                # Calculate duration
                duration = (continuous_series['timestamp'].max() - 
                           continuous_series['timestamp'].min()).days
                
                if duration >= min_days:
                    qualified_events[event_slug] = continuous_series
                    print(f"   ✅ {event_slug}: {duration} days of history")
                else:
                    print(f"   ❌ {event_slug}: only {duration} days (< {min_days})")
            else:
                print(f"   ❌ {event_slug}: no continuous series created")
        
        print(f"🎯 Found {len(qualified_events)} events with {min_days}+ days of history")
        return qualified_events
    
    def save_continuous_timeseries_dataset(self, qualified_events: Dict[str, pd.DataFrame], 
                                         output_file: str = 'data/processed/continuous_event_timeseries.parquet'):
        """Save the continuous time series dataset"""
        print("💾 Saving continuous time series dataset...")
        
        all_series = []
        
        for event_slug, df in qualified_events.items():
            df_copy = df.copy()
            df_copy['event_slug'] = event_slug
            all_series.append(df_copy)
        
        if all_series:
            combined_df = pd.concat(all_series, ignore_index=True)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to parquet
            combined_df.to_parquet(output_file, index=False)
            print(f"✅ Saved {len(combined_df)} rows covering {len(qualified_events)} events to {output_file}")
            
            # Save summary statistics
            summary = {
                'total_events': len(qualified_events),
                'total_data_points': len(combined_df),
                'date_range': {
                    'start': combined_df['timestamp'].min().isoformat(),
                    'end': combined_df['timestamp'].max().isoformat()
                },
                'events': {}
            }
            
            for event_slug, df in qualified_events.items():
                summary['events'][event_slug] = {
                    'data_points': len(df),
                    'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                    'start_date': df['timestamp'].min().isoformat(),
                    'end_date': df['timestamp'].max().isoformat(),
                    'contracts_used': df['active_contract'].nunique(),
                    'avg_price': df['price'].mean(),
                    'price_volatility': df['price'].std()
                }
            
            summary_file = output_file.replace('.parquet', '_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Saved summary statistics to {summary_file}")
            
            return combined_df
        
        return None
    
    def create_sample_charts(self, qualified_events: Dict[str, pd.DataFrame], 
                           output_dir: str = 'data/outputs/continuous_timeseries_charts'):
        """Generate sample charts showing reconstructed event time series"""
        print("📈 Creating sample charts...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Select up to 6 events for charting
        sample_events = list(qualified_events.keys())[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, event_slug in enumerate(sample_events):
            if i >= 6:
                break
                
            df = qualified_events[event_slug]
            ax = axes[i]
            
            # Plot price over time
            ax.plot(df['timestamp'], df['price'], linewidth=1.5, alpha=0.8)
            
            # Add vertical lines for contract switches
            contract_switches = df.groupby('active_contract')['timestamp'].min()
            for contract_id, switch_time in contract_switches.items():
                if switch_time != df['timestamp'].min():  # Don't mark the first contract start
                    ax.axvline(x=switch_time, color='red', linestyle='--', alpha=0.5)
            
            ax.set_title(f"{event_slug.replace('-', ' ').title()}", fontsize=10, pad=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            # Format y-axis as percentage
            ax.set_ylim(0, 1)
            ax.set_yticklabels([f'{int(x*100)}%' for x in ax.get_yticks()])
        
        # Remove empty subplots
        for i in range(len(sample_events), 6):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle("Continuous Event Time Series - Sample Events", fontsize=16, y=1.02)
        
        # Save the chart
        chart_file = os.path.join(output_dir, 'sample_continuous_timeseries.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved sample chart to {chart_file}")
        plt.close()
        
        # Create individual detailed charts for top events
        for i, event_slug in enumerate(sample_events[:3]):  # Top 3 events
            df = qualified_events[event_slug]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Price chart
            ax1.plot(df['timestamp'], df['price'], linewidth=2)
            
            # Color-code by active contract
            contracts = df['active_contract'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(contracts)))
            
            for j, contract in enumerate(contracts):
                contract_data = df[df['active_contract'] == contract]
                ax1.plot(contract_data['timestamp'], contract_data['price'], 
                        color=colors[j], linewidth=2, alpha=0.8, label=f'Contract {j+1}')
            
            ax1.set_title(f"{event_slug.replace('-', ' ').title()}", fontsize=14)
            ax1.set_ylabel("Probability")
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_ylim(0, 1)
            ax1.set_yticklabels([f'{int(x*100)}%' for x in ax1.get_yticks()])
            
            # Volume chart
            ax2.bar(df['timestamp'], df['volume'], alpha=0.6, width=1)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save individual chart
            individual_chart = os.path.join(output_dir, f'{event_slug.replace("/", "_")}_detailed.png')
            plt.savefig(individual_chart, dpi=300, bbox_inches='tight')
            print(f"📊 Saved detailed chart for {event_slug} to {individual_chart}")
            plt.close()

def main():
    print("🚀 Building Continuous Event Time Series from Prediction Market Contracts")
    print("=" * 70)
    
    # Initialize the builder
    builder = ContinuousTimeseriesBuilder()
    
    # Load markets data
    builder.load_markets()
    
    # Filter events with sufficient continuous history
    qualified_events = builder.filter_events_by_duration(min_days=60)
    
    if not qualified_events:
        print("❌ No events found with sufficient continuous history")
        return
    
    # Save the continuous time series dataset
    combined_df = builder.save_continuous_timeseries_dataset(qualified_events)
    
    # Create sample charts
    builder.create_sample_charts(qualified_events)
    
    print("\n✅ Continuous time series analysis complete!")
    print(f"📊 Created continuous series for {len(qualified_events)} events")
    if combined_df is not None:
        print(f"📈 Total data points: {len(combined_df):,}")
        print(f"📅 Date range: {combined_df['timestamp'].min().strftime('%Y-%m-%d')} to {combined_df['timestamp'].max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()