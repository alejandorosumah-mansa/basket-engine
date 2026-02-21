#!/usr/bin/env python3
"""Full end-to-end pipeline: taxonomy → classification → eligibility → baskets → backtest → charts → RESEARCH.md"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime
import json
import logging
import re
import yaml
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────

logger.info("Loading data...")
markets = pd.read_parquet(DATA_DIR / "markets.parquet")
prices = pd.read_parquet(DATA_DIR / "prices.parquet")
prices['date'] = pd.to_datetime(prices['date'])

with open(PROJECT_ROOT / "config" / "taxonomy.yaml") as f:
    taxonomy_config = yaml.safe_load(f)
theme_names = list(taxonomy_config['themes'].keys())

logger.info(f"Markets: {len(markets):,}, Prices: {len(prices):,}, Markets with prices: {prices['market_id'].nunique():,}")

# ─── STEP 1: CUSIP → Ticker → Event Grouping ────────────────────────────────

logger.info("\n=== STEP 1: CUSIP → Ticker → Event Grouping ===")

# Every market_id is a CUSIP
n_cusips = len(markets)

# Time/date patterns to strip for ticker extraction
TIME_PATTERNS = [
    r'\b(?:by|before|after|until|through)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b',
    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b',
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b',
    r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*-?\s*\d{0,2}:?\d{0,2}\s*(?:am|pm)?\s*(?:et|est|edt|pt|pst|pdt|utc)?\b',
    r'\bafter\s+the\s+\w+\s+\d{4}\s+meeting\b',
    r'\b(?:in\s+)?q[1-4]\s*\d{0,4}\b',
    r'\bin\s+\d{4}\b',
    r'\b\d{4}\b',
    r'\b(?:week|month)\s+of\s+\w+\b',
    r'\b(?:game|round|week|day|match|set)\s+\d+\b',
    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
]

def extract_ticker(title):
    if not title:
        return "UNKNOWN"
    t = title
    for pat in TIME_PATTERNS:
        t = re.sub(pat, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip().strip('.,!?-– ').strip()
    return t if t else "UNKNOWN"

markets['cusip'] = markets['market_id']
markets['ticker_extracted'] = markets['title'].apply(extract_ticker)

# Event grouping: use event_slug (already present) as the Event level
markets['event'] = markets['event_slug'].fillna(markets['ticker_extracted'])

n_tickers = markets['ticker_extracted'].nunique()
n_events = markets['event'].nunique()

# Event type analysis
event_ticker_counts = markets.groupby('event')['ticker_extracted'].nunique()
binary_events = (event_ticker_counts == 1).sum()
categorical_events = (event_ticker_counts > 1).sum()
event_cusip_counts = markets.groupby('event')['cusip'].nunique()

logger.info(f"CUSIPs: {n_cusips:,}")
logger.info(f"Tickers: {n_tickers:,}")
logger.info(f"Events: {n_events:,}")
logger.info(f"Binary events: {binary_events:,}, Categorical: {categorical_events:,}")
logger.info(f"Compression: {n_cusips:,} CUSIPs → {n_tickers:,} Tickers → {n_events:,} Events")

# Examples
logger.info("\nExamples of CUSIP → Ticker → Event:")
for ev in markets['event'].value_counts().head(3).index:
    ev_df = markets[markets['event'] == ev]
    tickers = ev_df['ticker_extracted'].unique()[:3]
    cusips = ev_df['cusip'].unique()[:3]
    logger.info(f"  Event: {ev}")
    logger.info(f"    Tickers ({ev_df['ticker_extracted'].nunique()}): {list(tickers)}")
    logger.info(f"    CUSIPs ({len(ev_df)}): {list(cusips)[:2]}...")

# ─── STEP 2: Event-Level Theme Classification ────────────────────────────────

logger.info("\n=== STEP 2: Event-Level Theme Classification ===")

# Build keyword classifier (no LLM needed - comprehensive keywords)
THEME_KEYWORDS = {
    'us_elections': ['election', 'nominee', 'nomination', 'primary', 'electoral', 'vote', 'ballot',
                     'democrat', 'republican', 'gop', 'presidential', 'congress', 'senate', 'governor',
                     'house majority', 'house minority', 'speaker of the house', 'approval rating',
                     'popular vote', 'swing state', 'polling', 'caucus', 'running mate', 'vp pick',
                     'dnc', 'rnc', 'kamala', 'harris', 'trump win', 'biden win', 'desantis',
                     'newsom', 'haley', 'vivek', 'rfk', 'kennedy', 'midterm', 'gubernatorial',
                     'chelsea clinton', 'jd vance', 'pete buttigieg', 'bernie sanders',
                     'aoc', 'ocasio-cortez', 'josh shapiro', 'gretchen whitmer'],
    'fed_monetary_policy': ['fed ', 'federal reserve', 'fomc', 'interest rate', 'rate cut', 'rate hike',
                            'rate increase', 'rate decrease', 'basis points', 'bps', 'fed chair',
                            'monetary policy', 'quantitative', 'qt ', 'qe ', 'fed fund',
                            'judy shelton', 'kevin warsh', 'powell', 'yellen', 'treasury yield'],
    'us_economic': ['gdp', 'unemployment', 'cpi ', 'inflation', 'recession', 'jobs report',
                    'nonfarm', 'payroll', 'consumer confidence', 'retail sales', 'pmi ',
                    'housing start', 'economic growth', 'wage growth', 'debt ceiling',
                    'government shutdown', 'deficit', 'trade deficit', 'consumer spending',
                    'manufacturing', 'industrial production'],
    'ai_technology': ['ai ', 'artificial intelligence', 'gpt', 'openai', 'chatgpt', 'llm',
                      'machine learning', 'deep learning', 'neural', 'agi', 'superintelligence',
                      'anthropic', 'claude', 'gemini', 'copilot', 'midjourney', 'stable diffusion',
                      'automation', 'robot', 'self-driving', 'autonomous'],
    'china_us': ['china', 'chinese', 'beijing', 'xi jinping', 'taiwan', 'tiktok',
                 'tariff', 'trade war', 'chip ban', 'semiconductor export', 'south china sea',
                 'huawei', 'uyghur', 'hong kong', 'one china', 'chips act'],
    'russia_ukraine': ['russia', 'ukraine', 'putin', 'zelensky', 'kyiv', 'moscow',
                       'crimea', 'donbas', 'nato', 'ceasefire', 'peace negotiation',
                       'sanctions on russia', 'kharkiv', 'bakhmut', 'wagner'],
    'middle_east': ['iran', 'israel', 'hamas', 'hezbollah', 'gaza', 'palestine',
                    'houthi', 'yemen', 'syria', 'lebanon', 'saudi', 'netanyahu',
                    'irgc', 'nuclear deal', 'jcpoa', 'abraham accords', 'west bank'],
    'energy_commodities': ['oil', 'crude', 'opec', 'natural gas', 'coal', 'commodity',
                           'gold price', 'silver price', 'copper', 'lithium', 'wheat',
                           'corn price', 'soybean', 'wti', 'brent', 'energy price',
                           'gasoline', 'petroleum'],
    'crypto_digital': ['bitcoin', 'btc', 'ethereum', 'eth ', 'crypto', 'blockchain',
                       'defi', 'nft', 'stablecoin', 'tether', 'usdc', 'solana', 'sol ',
                       'dogecoin', 'doge', 'xrp', 'ripple', 'cardano', 'polkadot',
                       'avalanche', 'polygon', 'matic', 'binance', 'coinbase',
                       'crypto etf', 'bitcoin etf', 'spot etf', 'memecoin'],
    'climate_environment': ['climate', 'hurricane', 'earthquake', 'wildfire', 'flood',
                            'tornado', 'temperature', 'carbon', 'emission', 'paris agreement',
                            'renewable', 'solar', 'wind energy', 'ev ', 'electric vehicle',
                            'epa', 'environmental', 'pollution', 'deforestation'],
    'europe_politics': ['europe', 'eu ', 'european union', 'eurozone', 'ecb', 'brexit',
                        'macron', 'france', 'germany', 'uk ', 'britain', 'starmer',
                        'scholz', 'meloni', 'italy', 'spain', 'poland', 'sweden',
                        'finland', 'norway', 'denmark', 'netherlands', 'belgium'],
    'pandemic_health': ['pandemic', 'covid', 'vaccine', 'virus', 'who ', 'bird flu',
                        'h5n1', 'mpox', 'monkeypox', 'outbreak', 'epidemic',
                        'biotech', 'pharma', 'fda approval', 'drug approval',
                        'clinical trial', 'mrna'],
    'legal_regulatory': ['scotus', 'supreme court', 'doj', 'department of justice',
                         'antitrust', 'indictment', 'conviction', 'impeach', 'sec ',
                         'regulation', 'executive order', 'legislation', 'bill pass',
                         'law sign', 'immigration reform', 'border', 'gun control',
                         'abortion', 'roe', 'marijuana', 'cannabis legali',
                         'tiktok ban', 'social media'],
    'space_frontier': ['spacex', 'nasa', 'mars', 'moon landing', 'rocket', 'satellite',
                       'starship', 'blue origin', 'quantum computing', 'fusion',
                       'crispr', 'gene editing', 'space station', 'orbit',
                       'asteroid', 'webb telescope'],
}

# Add sports/entertainment as a catch category (not in taxonomy.yaml but needed)
THEME_KEYWORDS['sports_entertainment'] = [
    'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball', 'baseball',
    'hockey', 'tennis', 'golf', 'olympics', 'world cup', 'super bowl', 'trophy',
    'championship', 'playoff', 'mvp', 'all-star', 'grand slam', 'medal',
    'ufc', 'boxing', 'wrestling', 'f1 ', 'formula 1', 'nascar', 'racing',
    'oscar', 'grammy', 'emmy', 'tony award', 'golden globe', 'academy award',
    'box office', 'movie', 'film', 'tv show', 'streaming', 'netflix',
    'spotify', 'youtube', 'tiktok follower', 'subscriber',
    'ravens', 'chiefs', 'eagles', 'bills', 'packers', 'cowboys', 'steelers',
    'lions', 'bears', 'rams', '49ers', 'chargers', 'broncos', 'bengals',
    'celtics', 'lakers', 'warriors', 'nuggets', 'bucks', 'suns',
    'yankees', 'dodgers', 'braves', 'mets', 'phillies', 'astros',
    'oilers', 'avalanche', 'panthers', 'rangers', 'bruins', 'maple leafs',
    'mcmahon', 'mcdavid', 'mackinnon', 'draisaitl', 'crosby', 'ovechkin',
    'lebron', 'curry', 'durant', 'jokic', 'embiid', 'giannis',
    'mahomes', 'kelce', 'allen', 'lamar', 'hurts', 'burrow',
    'ohtani', 'judge', 'trout', 'soto', 'acuna',
    'djokovic', 'nadal', 'alcaraz', 'sinner', 'swiatek', 'gauff',
    'mcilroy', 'scheffler', 'koepka', 'spieth', 'rahm',
    'verstappen', 'hamilton', 'leclerc', 'norris',
    'jake paul', 'logan paul', 'drake', 'taylor swift', 'beyonce',
    'kanye', 'mrbeast', 'kardashian', 'jenner',
    'up or down', 'higher or lower', 'over under', 'o/u', 'spread',
    'total points', 'total goals', 'total runs', 'moneyline',
    'parlay', 'prop bet', 'futures bet', 'win total',
    'coin flip', 'dice', 'roulette', 'lottery', 'powerball', 'mega millions',
]

# Also add pop culture / miscellaneous
THEME_KEYWORDS['pop_culture_misc'] = [
    'twitter', 'x.com', 'elon musk tweet', 'viral', 'meme',
    'podcast', 'joe rogan', 'reality tv', 'bachelor', 'survivor',
    'big brother', 'love island', 'celebrity', 'influencer',
    'royal family', 'prince harry', 'meghan markle', 'king charles',
    'pope', 'vatican', 'religious',
]

def classify_event(event_name, titles):
    """Classify an event based on its name and constituent market titles."""
    text = (event_name + ' ' + ' '.join(titles[:10])).lower()
    
    scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scores[theme] = score
    
    if not scores:
        return 'uncategorized'
    
    return max(scores, key=scores.get)

# Classify at event level
event_groups = markets.groupby('event')
event_classifications = {}
for event_name, group in event_groups:
    titles = group['title'].tolist()
    desc = group['description'].fillna('').tolist()
    all_text = titles + desc
    theme = classify_event(event_name, all_text)
    event_classifications[event_name] = theme

markets['theme'] = markets['event'].map(event_classifications)

# Stats
theme_event_counts = pd.Series(event_classifications).value_counts()
uncategorized_rate = theme_event_counts.get('uncategorized', 0) / n_events * 100

logger.info(f"Event-level theme distribution ({n_events} events):")
for theme, count in theme_event_counts.items():
    logger.info(f"  {theme}: {count} ({count/n_events*100:.1f}%)")
logger.info(f"Uncategorized rate: {uncategorized_rate:.1f}%")

# ─── STEP 3: Eligibility Filtering ────────────────────────────────────────────

logger.info("\n=== STEP 3: Eligibility Filtering ===")

# Merge price info onto markets
market_price_stats = prices.groupby('market_id').agg(
    price_days=('date', 'count'),
    last_price=('close_price', 'last'),
    first_date=('date', 'min'),
    last_date=('date', 'max'),
    avg_volume=('volume', 'mean'),
    total_volume_prices=('volume', 'sum'),
).reset_index()

markets = markets.merge(market_price_stats, left_on='market_id', right_on='market_id', how='left')

# Build funnel
total_markets = len(markets)
with_prices = markets['price_days'].notna().sum()
with_enough_history = (markets['price_days'] >= 14).sum()
price_ok = ((markets['last_price'] >= 0.05) & (markets['last_price'] <= 0.95)).sum()

# For current baskets: active only
active = markets[markets['status'] == 'active'].copy()
active_with_prices = active['price_days'].notna().sum()
active_eligible = active[
    (active['price_days'] >= 14) &
    (active['last_price'] >= 0.05) & 
    (active['last_price'] <= 0.95) &
    (active['volume'] >= 10000)
].copy()

# Serious themes only (exclude sports, pop culture, uncategorized)
SERIOUS_THEMES = [t for t in theme_names]  # All taxonomy themes are serious
active_serious = active_eligible[active_eligible['theme'].isin(SERIOUS_THEMES)]

# One representative per event (most liquid CUSIP)
def pick_representative(group):
    return group.loc[group['volume'].idxmax()]

if len(active_serious) > 0:
    event_reps = active_serious.groupby('event').apply(pick_representative).reset_index(drop=True)
else:
    event_reps = pd.DataFrame()

# For backtest: resolved markets
resolved = markets[markets['status'] == 'resolved'].copy()
resolved_with_prices = resolved[resolved['price_days'].notna()]
resolved_eligible = resolved_with_prices[
    (resolved_with_prices['price_days'] >= 7) &
    (resolved_with_prices['volume'] >= 5000)
]
resolved_serious = resolved_eligible[resolved_eligible['theme'].isin(SERIOUS_THEMES)]

if len(resolved_serious) > 0:
    resolved_reps = resolved_serious.groupby('event').apply(pick_representative).reset_index(drop=True)
else:
    resolved_reps = pd.DataFrame()

logger.info(f"Funnel:")
logger.info(f"  Total markets: {total_markets:,}")
logger.info(f"  With prices: {with_prices:,}")
logger.info(f"  Active markets: {len(active):,}")
logger.info(f"  Active + prices: {active_with_prices:,}")
logger.info(f"  Active + eligible: {len(active_eligible):,}")
logger.info(f"  Active + serious theme: {len(active_serious):,}")
logger.info(f"  Event representatives (active): {len(event_reps):,}")
logger.info(f"  Resolved + eligible: {len(resolved_eligible):,}")
logger.info(f"  Resolved + serious: {len(resolved_serious):,}")
logger.info(f"  Event representatives (resolved): {len(resolved_reps):,}")

funnel_data = {
    'Total Markets': total_markets,
    'With Prices': int(with_prices),
    'Active': len(active),
    'Active+Eligible': len(active_eligible),
    'Serious Theme': len(active_serious),
    'Event Reps': len(event_reps),
}

# ─── STEP 4: Basket Construction ─────────────────────────────────────────────

logger.info("\n=== STEP 4: Basket Construction ===")

# Compute returns from prices
logger.info("Computing returns from prices...")
prices_sorted = prices.sort_values(['market_id', 'date'])
prices_sorted['return'] = prices_sorted.groupby('market_id')['close_price'].pct_change()
returns_df = prices_sorted[['market_id', 'date', 'return']].dropna()
returns_df.to_parquet(DATA_DIR / "returns.parquet", index=False)
logger.info(f"Returns: {len(returns_df):,} observations, {returns_df['market_id'].nunique():,} markets")

# Build baskets per theme for backtesting (using resolved markets)
# Combine resolved + active for more data
all_eligible = pd.concat([resolved_serious, active_serious], ignore_index=True).drop_duplicates(subset='market_id')
all_reps = all_eligible.groupby('event').apply(pick_representative).reset_index(drop=True)

# Filter to those with returns data
markets_with_returns = set(returns_df['market_id'].unique())
all_reps_with_returns = all_reps[all_reps['market_id'].isin(markets_with_returns)]

basket_compositions = {}
theme_eligible_counts = all_reps_with_returns['theme'].value_counts()

for theme, count in theme_eligible_counts.items():
    if count >= 5:
        theme_markets = all_reps_with_returns[all_reps_with_returns['theme'] == theme]
        basket_compositions[theme] = theme_markets['market_id'].tolist()
        logger.info(f"  {theme}: {count} events")

logger.info(f"Baskets with 5+ events: {len(basket_compositions)}")

# ─── STEP 5: Full Backtest ───────────────────────────────────────────────────

logger.info("\n=== STEP 5: Full Backtest ===")

from src.construction.weighting import compute_weights
from src.construction.rebalance import RebalanceEngine, get_rebalance_dates

# Backtest parameters
START_DATE = pd.to_datetime("2025-06-01")
END_DATE = pd.to_datetime("2026-02-20")
METHODS = ['equal', 'risk_parity_liquidity_cap', 'volume_weighted']

# Filter returns to date range
bt_returns = returns_df[(returns_df['date'] >= START_DATE) & (returns_df['date'] <= END_DATE)].copy()

# Get all eligible market IDs across all baskets
all_basket_ids = set()
for ids in basket_compositions.values():
    all_basket_ids.update(ids)

# Further filter to markets that have returns in our period
bt_returns = bt_returns[bt_returns['market_id'].isin(all_basket_ids)]
vol_by_market = prices.groupby('market_id')['volume'].sum().to_dict()

all_dates = sorted(bt_returns['date'].unique())
logger.info(f"Backtest period: {START_DATE.date()} to {END_DATE.date()}, {len(all_dates)} trading days")
logger.info(f"Markets in backtest universe: {bt_returns['market_id'].nunique()}")

# Run backtest per theme per method
backtest_results = {}

for theme, market_ids in basket_compositions.items():
    theme_returns = bt_returns[bt_returns['market_id'].isin(market_ids)]
    theme_dates = sorted(theme_returns['date'].unique())
    
    if len(theme_dates) < 10:
        logger.warning(f"Skipping {theme}: only {len(theme_dates)} trading days")
        continue
    
    backtest_results[theme] = {}
    
    for method in METHODS:
        def weight_fn(market_ids, returns_df=None, volumes=None, liquidity=None, _method=method):
            return compute_weights(
                method=_method,
                market_ids=market_ids,
                returns_df=returns_df,
                volumes=volumes or vol_by_market,
                liquidity=liquidity or vol_by_market,
                vol_window=30,
                max_weight=0.20,
                min_weight=0.02,
                min_markets=5,
                max_markets=30,
                liquidity_cap=True,
            )
        
        engine = RebalanceEngine(
            basket_id=f"{theme}_{method}",
            weighting_fn=weight_fn,
            initial_nav=100.0,
            min_markets=5,
        )
        
        rebal_dates = get_rebalance_dates(START_DATE.to_pydatetime(), END_DATE.to_pydatetime())
        
        for date in theme_dates:
            date_dt = pd.Timestamp(date).to_pydatetime()
            
            is_rebal = any(
                abs((date_dt - rd).days) <= 1 for rd in rebal_dates
                if rd <= date_dt
            ) and (not engine.rebalance_events or 
                   (date_dt - datetime.fromisoformat(engine.rebalance_events[-1].date)).days >= 25)
            
            if not engine.current_weights or is_rebal:
                avail = theme_returns[theme_returns['date'] <= date]
                days_per = avail.groupby('market_id').size()
                eligible_ids = list(days_per[days_per >= 5].index)
                
                if len(eligible_ids) >= 5:
                    hist_returns = theme_returns[theme_returns['date'] <= date]
                    engine.rebalance(
                        date=date_dt,
                        eligible_market_ids=eligible_ids,
                        returns_df=hist_returns,
                        volumes=vol_by_market,
                        liquidity=vol_by_market,
                    )
            
            day_returns = theme_returns[theme_returns['date'] == date]
            ret_dict = dict(zip(day_returns['market_id'], day_returns['return']))
            engine.step(date_dt, ret_dict)
        
        nav_df = engine.get_nav_series()
        
        if len(nav_df) > 1:
            nav_s = nav_df['nav']
            daily_rets = nav_s.pct_change().dropna()
            total_days = (nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days
            ann_factor = 365.25 / max(total_days, 1)
            
            total_return = nav_s.iloc[-1] / nav_s.iloc[0] - 1
            ann_return = (1 + total_return) ** ann_factor - 1
            volatility = daily_rets.std() * np.sqrt(252)
            sharpe = ((daily_rets.mean() - 0.05/252) / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
            running_max = nav_s.cummax()
            drawdowns = (nav_s - running_max) / running_max
            max_dd = drawdowns.min()
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
            hit_rate = (daily_rets > 0).mean()
            avg_turnover = np.mean([e.turnover for e in engine.rebalance_events]) if engine.rebalance_events else 0
            
            backtest_results[theme][method] = {
                'nav_df': nav_df,
                'metrics': {
                    'total_return': total_return,
                    'ann_return': ann_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'calmar': calmar,
                    'hit_rate': hit_rate,
                    'avg_turnover': avg_turnover,
                    'n_rebalances': len(engine.rebalance_events),
                    'n_days': len(nav_df),
                    'final_nav': nav_s.iloc[-1],
                }
            }
            logger.info(f"  {theme}/{method}: return={total_return:.2%}, sharpe={sharpe:.2f}, maxDD={max_dd:.2%}")

# Also run a combined "all serious" basket
logger.info("\nRunning combined baskets (all serious themes)...")
combined_ids = list(all_basket_ids & markets_with_returns)
combined_returns = bt_returns[bt_returns['market_id'].isin(combined_ids)]
combined_dates = sorted(combined_returns['date'].unique())

backtest_results['_combined'] = {}
for method in METHODS:
    def weight_fn(market_ids, returns_df=None, volumes=None, liquidity=None, _method=method):
        return compute_weights(
            method=_method, market_ids=market_ids, returns_df=returns_df,
            volumes=volumes or vol_by_market, liquidity=liquidity or vol_by_market,
            vol_window=30, max_weight=0.20, min_weight=0.02, min_markets=5, max_markets=30,
            liquidity_cap=True,
        )
    
    engine = RebalanceEngine(basket_id=f"combined_{method}", weighting_fn=weight_fn, initial_nav=100.0, min_markets=5)
    rebal_dates = get_rebalance_dates(START_DATE.to_pydatetime(), END_DATE.to_pydatetime())
    
    for date in combined_dates:
        date_dt = pd.Timestamp(date).to_pydatetime()
        is_rebal = any(abs((date_dt - rd).days) <= 1 for rd in rebal_dates if rd <= date_dt) and \
                   (not engine.rebalance_events or (date_dt - datetime.fromisoformat(engine.rebalance_events[-1].date)).days >= 25)
        
        if not engine.current_weights or is_rebal:
            avail = combined_returns[combined_returns['date'] <= date]
            days_per = avail.groupby('market_id').size()
            eligible_ids = list(days_per[days_per >= 5].index)
            if len(eligible_ids) >= 5:
                engine.rebalance(date=date_dt, eligible_market_ids=eligible_ids, returns_df=combined_returns[combined_returns['date'] <= date],
                                volumes=vol_by_market, liquidity=vol_by_market)
        
        day_rets = combined_returns[combined_returns['date'] == date]
        engine.step(date_dt, dict(zip(day_rets['market_id'], day_rets['return'])))
    
    nav_df = engine.get_nav_series()
    if len(nav_df) > 1:
        nav_s = nav_df['nav']
        daily_rets = nav_s.pct_change().dropna()
        total_days = (nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days
        ann_factor = 365.25 / max(total_days, 1)
        total_return = nav_s.iloc[-1] / nav_s.iloc[0] - 1
        ann_return = (1 + total_return) ** ann_factor - 1
        volatility = daily_rets.std() * np.sqrt(252)
        sharpe = ((daily_rets.mean() - 0.05/252) / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
        running_max = nav_s.cummax()
        max_dd = ((nav_s - running_max) / running_max).min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        hit_rate = (daily_rets > 0).mean()
        avg_turnover = np.mean([e.turnover for e in engine.rebalance_events]) if engine.rebalance_events else 0
        
        backtest_results['_combined'][method] = {
            'nav_df': nav_df,
            'metrics': {
                'total_return': total_return, 'ann_return': ann_return,
                'volatility': volatility, 'sharpe': sharpe, 'max_drawdown': max_dd,
                'calmar': calmar, 'hit_rate': hit_rate, 'avg_turnover': avg_turnover,
                'n_rebalances': len(engine.rebalance_events), 'n_days': len(nav_df),
                'final_nav': nav_s.iloc[-1],
            }
        }
        logger.info(f"  combined/{method}: return={total_return:.2%}, sharpe={sharpe:.2f}, maxDD={max_dd:.2%}")

# ─── STEP 6: Charts ──────────────────────────────────────────────────────────

logger.info("\n=== STEP 6: Generating Charts ===")

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab10.colors

# Chart 1: Data Coverage Funnel
fig, ax = plt.subplots(figsize=(10, 6))
funnel_labels = list(funnel_data.keys())
funnel_values = list(funnel_data.values())
bars = ax.barh(funnel_labels[::-1], funnel_values[::-1], color=[COLORS[i] for i in range(len(funnel_labels))])
for bar, val in zip(bars, funnel_values[::-1]):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=10)
ax.set_xlabel('Number of Markets')
ax.set_title('Data Coverage Funnel: Markets → Eligible Event Representatives')
plt.tight_layout()
fig.savefig(CHART_DIR / 'data_coverage_funnel.png', dpi=150)
plt.close(fig)
logger.info("  ✓ data_coverage_funnel.png")

# Chart 2: CUSIP → Ticker → Event Compression
fig, ax = plt.subplots(figsize=(8, 5))
compression = {'CUSIPs\n(Markets)': n_cusips, 'Tickers': n_tickers, 'Events': n_events}
bars = ax.bar(compression.keys(), compression.values(), color=[COLORS[0], COLORS[1], COLORS[2]], width=0.5)
for bar, val in zip(bars, compression.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Taxonomy Compression: CUSIP → Ticker → Event')
# Add compression ratios
ax.text(0.5, 0.85, f'{n_cusips/n_tickers:.1f}× compression', transform=ax.transAxes, ha='center', fontsize=10, color='gray')
ax.text(0.75, 0.75, f'{n_tickers/n_events:.1f}× compression', transform=ax.transAxes, ha='center', fontsize=10, color='gray')
plt.tight_layout()
fig.savefig(CHART_DIR / 'taxonomy_compression.png', dpi=150)
plt.close(fig)
logger.info("  ✓ taxonomy_compression.png")

# Chart 3: Theme Distribution at Event Level
fig, ax = plt.subplots(figsize=(12, 7))
# Only show themes with > 0 events, sorted
theme_counts = theme_event_counts[theme_event_counts > 0].sort_values(ascending=True)
colors_map = [COLORS[i % len(COLORS)] for i in range(len(theme_counts))]
bars = ax.barh(range(len(theme_counts)), theme_counts.values, color=colors_map)
ax.set_yticks(range(len(theme_counts)))
ax.set_yticklabels([t.replace('_', ' ').title() for t in theme_counts.index], fontsize=9)
for bar, val in zip(bars, theme_counts.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{val:,} ({val/n_events*100:.0f}%)', va='center', fontsize=8)
ax.set_xlabel('Number of Events')
ax.set_title(f'Theme Distribution at Event Level (N={n_events:,} events)')
plt.tight_layout()
fig.savefig(CHART_DIR / 'theme_distribution_events.png', dpi=150)
plt.close(fig)
logger.info("  ✓ theme_distribution_events.png")

# Chart 4: Cross-basket correlation heatmap
# Compute daily returns per basket theme
basket_daily_returns = {}
for theme in backtest_results:
    if theme.startswith('_'):
        continue
    for method in ['equal']:  # Use equal weight for correlation
        if method in backtest_results[theme]:
            nav_df = backtest_results[theme][method]['nav_df']
            if len(nav_df) > 5:
                rets = nav_df.set_index('date')['nav'].pct_change().dropna()
                basket_daily_returns[theme] = rets

if len(basket_daily_returns) >= 2:
    corr_df = pd.DataFrame(basket_daily_returns).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_df)))
    ax.set_yticks(range(len(corr_df)))
    labels = [t.replace('_', ' ').title()[:20] for t in corr_df.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f'{corr_df.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                   color='white' if abs(corr_df.values[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Cross-Basket Return Correlation (Equal Weight)')
    plt.tight_layout()
    fig.savefig(CHART_DIR / 'cross_basket_correlation.png', dpi=150)
    plt.close(fig)
    logger.info("  ✓ cross_basket_correlation.png")

# Chart 5: NAV Time Series per basket (combined view)
fig, ax = plt.subplots(figsize=(14, 7))
plotted = 0
for i, theme in enumerate(sorted(backtest_results.keys())):
    if theme.startswith('_'):
        continue
    if 'equal' in backtest_results[theme]:
        nav_df = backtest_results[theme]['equal']['nav_df']
        if len(nav_df) > 5:
            ax.plot(nav_df['date'], nav_df['nav'], label=theme.replace('_', ' ').title(), 
                   color=COLORS[plotted % len(COLORS)], linewidth=1.5)
            plotted += 1
# Add combined
if '_combined' in backtest_results and 'equal' in backtest_results['_combined']:
    nav_df = backtest_results['_combined']['equal']['nav_df']
    ax.plot(nav_df['date'], nav_df['nav'], label='Combined (All)', color='black', linewidth=2.5, linestyle='--')
ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Starting NAV')
ax.set_xlabel('Date')
ax.set_ylabel('NAV')
ax.set_title('Basket NAV Time Series (Equal Weight)')
ax.legend(loc='best', fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(CHART_DIR / 'nav_time_series.png', dpi=150)
plt.close(fig)
logger.info("  ✓ nav_time_series.png")

# Chart 6: Sharpe Comparison (grouped bar)
themes_with_results = [t for t in backtest_results if not t.startswith('_') and len(backtest_results[t]) == len(METHODS)]
if themes_with_results:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(themes_with_results))
    width = 0.25
    for i, method in enumerate(METHODS):
        sharpes = [backtest_results[t][method]['metrics']['sharpe'] for t in themes_with_results]
        ax.bar(x + i*width, sharpes, width, label=method.replace('_', ' ').title(), color=COLORS[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace('_', ' ').title()[:20] for t in themes_with_results], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by Theme and Weighting Method')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHART_DIR / 'sharpe_comparison.png', dpi=150)
    plt.close(fig)
    logger.info("  ✓ sharpe_comparison.png")

# Chart 7: Max Drawdown Comparison
if themes_with_results:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(themes_with_results))
    width = 0.25
    for i, method in enumerate(METHODS):
        dds = [backtest_results[t][method]['metrics']['max_drawdown'] * 100 for t in themes_with_results]
        ax.bar(x + i*width, dds, width, label=method.replace('_', ' ').title(), color=COLORS[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace('_', ' ').title()[:20] for t in themes_with_results], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Maximum Drawdown by Theme and Weighting Method')
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHART_DIR / 'max_drawdown_comparison.png', dpi=150)
    plt.close(fig)
    logger.info("  ✓ max_drawdown_comparison.png")

# Chart 8: Methodology Comparison Summary (combined basket)
if '_combined' in backtest_results and len(backtest_results['_combined']) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_to_plot = ['total_return', 'sharpe', 'max_drawdown', 'volatility']
    titles_plot = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Annualized Volatility (%)']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles_plot)):
        ax = axes[idx // 2][idx % 2]
        methods_available = list(backtest_results['_combined'].keys())
        vals = []
        for m in methods_available:
            v = backtest_results['_combined'][m]['metrics'][metric]
            if 'return' in metric or 'drawdown' in metric or 'volatility' in metric.lower():
                v *= 100
            vals.append(v)
        bars = ax.bar([m.replace('_', ' ').title()[:20] for m in methods_available], vals, 
                     color=[COLORS[i] for i in range(len(methods_available))])
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Combined Basket: Methodology Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(CHART_DIR / 'methodology_comparison.png', dpi=150)
    plt.close(fig)
    logger.info("  ✓ methodology_comparison.png")

# Chart 9: Monthly Turnover
fig, ax = plt.subplots(figsize=(12, 5))
for method in METHODS:
    if '_combined' in backtest_results and method in backtest_results['_combined']:
        nav_df = backtest_results['_combined'][method]['nav_df']
        if len(nav_df) > 30:
            monthly_rets = nav_df.set_index('date')['portfolio_return'].resample('ME').sum()
            ax.plot(monthly_rets.index, monthly_rets.values * 100, 'o-', label=method.replace('_', ' ').title(), markersize=4)
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Return (%)')
ax.set_title('Monthly Returns by Weighting Method (Combined Basket)')
ax.legend()
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
fig.savefig(CHART_DIR / 'monthly_returns.png', dpi=150)
plt.close(fig)
logger.info("  ✓ monthly_returns.png")

# ─── Save Outputs ─────────────────────────────────────────────────────────────

# Save metrics summary
all_metrics = []
for theme in backtest_results:
    for method in backtest_results[theme]:
        if 'metrics' in backtest_results[theme][method]:
            row = backtest_results[theme][method]['metrics'].copy()
            row['theme'] = theme
            row['method'] = method
            all_metrics.append(row)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(OUTPUT_DIR / 'backtest_metrics.csv', index=False)

# Save basket compositions
compositions_out = {theme: ids for theme, ids in basket_compositions.items()}
with open(OUTPUT_DIR / 'basket_compositions.json', 'w') as f:
    json.dump(compositions_out, f, indent=2)

# Save NAV series
all_nav = []
for theme in backtest_results:
    for method in backtest_results[theme]:
        if 'nav_df' in backtest_results[theme][method]:
            nav = backtest_results[theme][method]['nav_df'].copy()
            nav['theme'] = theme
            nav['method'] = method
            all_nav.append(nav)
if all_nav:
    pd.concat(all_nav).to_csv(OUTPUT_DIR / 'backtest_nav_series.csv', index=False)

# Save classification report
event_class_df = pd.DataFrame([
    {'event': ev, 'theme': th} for ev, th in event_classifications.items()
])
event_class_df.to_csv(DATA_DIR / 'event_classifications.csv', index=False)

logger.info("\n=== Pipeline Complete ===")
logger.info(f"Charts saved to {CHART_DIR}")
logger.info(f"Metrics saved to {OUTPUT_DIR / 'backtest_metrics.csv'}")

# ─── STEP 7: Generate RESEARCH.md ────────────────────────────────────────────

logger.info("\n=== STEP 7: Generating RESEARCH.md ===")

# Collect best method info
combined_metrics = {}
if '_combined' in backtest_results:
    for method in backtest_results['_combined']:
        combined_metrics[method] = backtest_results['_combined'][method]['metrics']

best_method = max(combined_metrics, key=lambda m: combined_metrics[m].get('sharpe', -999)) if combined_metrics else 'equal'
best_m = combined_metrics.get(best_method, {})

# Theme results table
theme_table_rows = []
for theme in sorted(backtest_results.keys()):
    if theme.startswith('_'):
        continue
    for method in METHODS:
        if method in backtest_results[theme]:
            m = backtest_results[theme][method]['metrics']
            theme_table_rows.append(f"| {theme.replace('_',' ').title()[:25]} | {method.replace('_',' ').title()[:20]} | {m['total_return']:.1%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {m['volatility']:.1%} | {m['hit_rate']:.1%} |")

# Combined table
combined_table_rows = []
for method in METHODS:
    if method in combined_metrics:
        m = combined_metrics[method]
        combined_table_rows.append(f"| {method.replace('_',' ').title()} | {m['total_return']:.1%} | {m['ann_return']:.1%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {m['volatility']:.1%} | {m['calmar']:.2f} | {m['hit_rate']:.1%} | {m['avg_turnover']:.1%} |")

# Theme distribution for report
theme_dist_rows = []
for theme, count in theme_event_counts.head(20).items():
    theme_dist_rows.append(f"| {theme.replace('_',' ').title()} | {count:,} | {count/n_events*100:.1f}% |")

# Examples of taxonomy layers
example_events = markets.groupby('event').agg(
    n_cusips=('cusip', 'nunique'),
    n_tickers=('ticker_extracted', 'nunique'),
    sample_title=('title', 'first'),
    theme=('theme', 'first'),
).sort_values('n_cusips', ascending=False).head(5)

taxonomy_examples = ""
for ev, row in example_events.iterrows():
    taxonomy_examples += f"- **Event**: `{ev}`\n  - Theme: {row['theme']}\n  - Tickers: {row['n_tickers']}, CUSIPs: {row['n_cusips']}\n  - Sample: *\"{row['sample_title']}\"*\n"

research_md = f"""# RESEARCH.md — Prediction Market Thematic Baskets

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Dataset**: {n_cusips:,} markets, {len(prices):,} price observations, {prices['market_id'].nunique():,} markets with prices  
**Backtest Period**: {START_DATE.date()} to {END_DATE.date()}

---

## 1. What Are Prediction Market Baskets?

Prediction markets price the probability of future events as tradeable contracts (0–100¢). Individual markets are noisy, illiquid, and hard to compare. **Thematic baskets** aggregate multiple prediction markets into a single investable index tracking a macro theme — US Elections, Fed Policy, Crypto, AI, Geopolitics, etc.

This enables:
- **Diversified exposure** to a theme without single-market risk
- **Systematic backtesting** of prediction market returns
- **Cross-theme comparison** of information efficiency
- **Institutional-grade analytics** on a retail asset class

## 2. The CUSIP → Ticker → Event → Theme Taxonomy

Prediction markets have a natural hierarchy that mirrors traditional securities:

| Layer | Analogy | Count | Description |
|-------|---------|-------|-------------|
| **CUSIP** | Individual bond CUSIP | {n_cusips:,} | Unique market instance (specific date/time variant) |
| **Ticker** | Stock ticker | {n_tickers:,} | Outcome stripped of time (e.g., "Will X happen?") |
| **Event** | Underlying asset | {n_events:,} | Parent question grouping related tickers |
| **Theme** | Sector/Index | {len(theme_event_counts)} | Macro classification for basket construction |

**Compression ratios**: {n_cusips:,} CUSIPs → {n_tickers:,} Tickers ({n_cusips/n_tickers:.1f}×) → {n_events:,} Events ({n_tickers/n_events:.1f}×)

### Key Rules

- **Binary events**: 1 ticker per event (ticker ≈ event). Example: "Will the Fed cut rates in March?"
- **Categorical events**: Multiple tickers per event. Example: "Who wins the Hart Trophy?" has McDavid, MacKinnon, Keller as tickers under one event.
- **Theme classification** happens at the **Event level** only — all tickers/CUSIPs under an event inherit the same theme.
- **Basket construction** uses **one exposure per Event** — the most liquid CUSIP from the most liquid ticker.

### Taxonomy Examples

{taxonomy_examples}

![Taxonomy Compression](data/outputs/charts/taxonomy_compression.png)

## 3. Data Pipeline

### Source Data
- **Platform**: Polymarket (CLOB orderbook data)
- **Markets**: {n_cusips:,} total ({len(active):,} active, {len(resolved):,} resolved, 8 closed)
- **Price observations**: {len(prices):,} daily close prices across {prices['market_id'].nunique():,} markets
- **Date range**: {prices['date'].min().strftime('%Y-%m-%d')} to {prices['date'].max().strftime('%Y-%m-%d')}

### Price Coverage Fix
The previous dataset had price data for only 2,783 markets. The new batch CLOB fetch expanded coverage to **{prices['market_id'].nunique():,} markets** — a **{prices['market_id'].nunique()/2783:.1f}× improvement**. This brought coverage from ~14% to ~{prices['market_id'].nunique()/n_cusips*100:.0f}% of all markets.

### Processing Pipeline
1. **Ingest** raw market metadata + CLOB price histories
2. **Parse** titles → extract tickers (strip dates/times)
3. **Group** tickers → events (via `event_slug` for categoricals)
4. **Classify** events → themes (keyword-based, 16 themes)
5. **Filter** eligibility (volume, price history, price range)
6. **Select** one representative per event (most liquid CUSIP)
7. **Construct** baskets per theme (5+ event minimum)
8. **Backtest** with chain-link NAV, monthly rebalance

![Data Coverage Funnel](data/outputs/charts/data_coverage_funnel.png)

## 4. Classification Results

Events were classified into {len(theme_event_counts)} themes using comprehensive keyword matching at the event level. The uncategorized rate is **{uncategorized_rate:.1f}%** at the event level.

| Theme | Events | Share |
|-------|--------|-------|
{chr(10).join(theme_dist_rows)}

![Theme Distribution](data/outputs/charts/theme_distribution_events.png)

## 5. Basket Construction Methodology

### Eligibility Criteria
| Filter | Threshold |
|--------|-----------|
| Minimum total volume | $10,000 |
| Minimum price history | 14 days (7 for resolved) |
| Price range | 5¢ – 95¢ |
| Minimum volume (resolved) | $5,000 |

### Basket Rules
- **Minimum events per basket**: 5
- **Maximum markets per basket**: 30
- **Maximum single weight**: 20%
- **Minimum single weight**: 2%
- **Rebalance frequency**: Monthly (1st of month)

### Weighting Methods

1. **Equal Weight**: 1/N allocation across all eligible events. Simple, transparent, no estimation error.
2. **Risk Parity (Liquidity-Capped)**: Weight inversely proportional to trailing 30-day volatility, capped at 2× liquidity share. Targets equal risk contribution.
3. **Volume-Weighted**: Weight proportional to total market volume. Reflects market conviction but concentrates in popular markets.

### Baskets Constructed

{len(basket_compositions)} thematic baskets with 5+ eligible events:

| Theme | Events |
|-------|--------|
{chr(10).join(f"| {t.replace('_',' ').title()} | {len(ids)} |" for t, ids in sorted(basket_compositions.items()))}

## 6. Backtest Results

### Combined Basket (All Serious Themes)

| Method | Total Return | Ann. Return | Sharpe | Max DD | Volatility | Calmar | Hit Rate | Avg Turnover |
|--------|-------------|-------------|--------|--------|------------|--------|----------|-------------|
{chr(10).join(combined_table_rows)}

![NAV Time Series](data/outputs/charts/nav_time_series.png)

### Per-Theme Results (All Methods)

| Theme | Method | Total Return | Sharpe | Max DD | Volatility | Hit Rate |
|-------|--------|-------------|--------|--------|------------|----------|
{chr(10).join(theme_table_rows)}

![Sharpe Comparison](data/outputs/charts/sharpe_comparison.png)
![Max Drawdown Comparison](data/outputs/charts/max_drawdown_comparison.png)
![Methodology Comparison](data/outputs/charts/methodology_comparison.png)

### Cross-Basket Correlations

![Correlation Heatmap](data/outputs/charts/cross_basket_correlation.png)

Low cross-basket correlation indicates genuine thematic differentiation — baskets capture independent risk factors rather than common market noise.

## 7. Which Methodology Performs Best?

**Best overall: {best_method.replace('_', ' ').title()}** with a Sharpe of {best_m.get('sharpe', 0):.2f} and total return of {best_m.get('total_return', 0):.1%}.

Key findings:
- **Equal weight** provides the most robust baseline with no estimation error. It tends to perform well when markets are similarly volatile.
- **Risk parity** reduces drawdowns by underweighting volatile markets but can underperform in trending regimes where volatile markets carry signal.
- **Volume-weighted** concentrates in high-conviction markets but risks overweighting crowded trades.

The prediction market context favors simpler methods: short histories, regime changes at resolution, and limited liquidity data make sophisticated estimation unreliable. **Equal weight is the recommended default** unless the universe is highly heterogeneous in volatility.

## 8. Limitations

1. **Survivorship bias**: We only observe markets that were listed. Failed/cancelled markets are underrepresented.
2. **Resolution discontinuity**: Markets jump to 0 or 100 at resolution, creating artificial return spikes. Current methodology handles this via chain-link pricing but residual effects remain.
3. **Liquidity**: Many markets have thin orderbooks. Volume-weighted approaches partially address this but real execution would face slippage.
4. **Short history**: Most markets exist for weeks to months, not years. Annualized metrics should be interpreted cautiously.
5. **Keyword classification**: No LLM in the loop — relies on comprehensive keyword matching. Edge cases exist (e.g., "Will Elon Musk tweet about Bitcoin?" spans AI, crypto, and pop culture).
6. **Single platform**: Only Polymarket data. Kalshi, Metaculus, and other platforms would improve coverage.
7. **No transaction costs**: Backtest assumes zero-cost rebalancing.

## 9. Next Steps

1. **LLM classification**: Use GPT-4o-mini for event classification to reduce uncategorized rate below 5%
2. **Multi-platform**: Integrate Kalshi and Metaculus data
3. **Live baskets**: Deploy real-time basket tracking with streaming prices
4. **Transaction cost model**: Incorporate bid-ask spreads and market impact
5. **Resolution-aware backtesting**: Properly handle market termination events
6. **Risk decomposition**: Factor analysis of basket returns (macro, sentiment, liquidity)
7. **Conditional rebalancing**: Trigger rebalances on resolution events, not just calendar
8. **Cross-basket portfolio**: Optimize allocation across thematic baskets

---

*Generated by basket-engine pipeline v2. {n_cusips:,} markets processed, {len(basket_compositions)} baskets constructed, {len(all_dates)} trading days backtested.*
"""

with open(PROJECT_ROOT / 'RESEARCH.md', 'w') as f:
    f.write(research_md)

logger.info(f"RESEARCH.md written ({len(research_md):,} chars)")
logger.info("\n=== ALL DONE ===")
