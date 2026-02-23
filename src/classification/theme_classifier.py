"""
Rule-based theme classifier for prediction markets.
Fast, deterministic, no API calls needed.
"""
import re
import pandas as pd

THEME_RULES = {
    'fed_monetary_policy': [
        r'\bFed\b.*(?:rate|interest|decrease|increase|cut|hike|meeting|chair|Powell|FOMC)',
        r'\bFOMC\b', r'\bFederal Reserve\b', r'\bmonetary policy\b',
        r'interest rate', r'rate cut', r'rate hike',
        r'Fed (?:chair|nominee|nominate)',
        r'Jerome Powell', r'Fed decreases', r'Fed increases',
        r'No change in Fed',
    ],
    'us_elections': [
        r'(?:2024|2025|2026|2027|2028).*(?:election|presidential|nomination|primary|caucus)',
        r'Balance of Power', r'electoral vote', r'popular vote',
        r'win the (?:Democratic|Republican).*(?:nomination|primary)',
        r'(?:Democratic|Republican).*(?:nominee|candidate)',
        r'(?:Senate|House|Governor).*(?:race|seat|win|flip)',
        r'(?:inaugurat|impeach|resign.*president)',
        r'Trump.*(?:win|president|elect|inaug|nomina)',
        r'(?:Harris|Biden|DeSantis|Haley|Ramaswamy|Newsom|Pence|Kennedy).*(?:win|president|elect|nomina)',
        r'(?:Mamdani|Cuomo|Adams).*(?:mayor|primary)',
        r'midterm', r'runoff',
    ],
    'global_politics': [
        r'(?:Prime Minister|President|Chancellor|leader).*(?:win|elect|out |resign|removed)',
        r'(?:Romania|Germany|Canada|Chile|Netherlands|Colombia|Hungary|Mexico|Turkey|France|UK|Japan|South Korea|Israel|Brazil|Argentina|Australia|India|Morocco|Indonesia|Cuba|Portugal|Norway|Sweden|Finland|Denmark|Belgium|Austria|Switzerland|Spain|Italy|Greece|Poland|Czech|Slovakia|Croatia|Serbia|Nigeria|Kenya|South Africa|Egypt).*(?:elect|president|prime minister|leader|mayor|recogni)',
        r'Macron|Erdogan|Trudeau|Starmer|Netanyahu|Modi|Lula|Milei|Zelenskyy|Maduro|Sheinbaum|Akhannouch|Listhaug|Mariani|Aboutaleb|Bontenbal',
        r'(?:NATO|EU|UN|BRICS|G7|G20).*(?:member|join|leave|expand|add)',
        r'(?:parliament|coalition|no.confidence|dissolution)',
        r'Seoul Mayor', r'Bucharest', r'Bontenbal',
        r'(?:Dutch|French|German|Italian|Spanish|Portuguese|Norwegian|Swedish|Danish|Finnish|Belgian|Austrian|Swiss|Greek|Polish|Czech|Slovak|Croatian|Serbian).*(?:government|election|coalition|prime minister)',
        r'out as.*(?:Prime Minister|President|Chancellor)',
    ],
    'russia_ukraine': [
        r'Russia.*Ukraine|Ukraine.*Russia', r'ceasefire',
        r'Zelenskyy', r'Crimea', r'Donbas', r'Donets',
        r'Minsk', r'NATO.*(?:Ukraine|Russia)',
    ],
    'middle_east': [
        r'Israel.*(?:Hamas|Iran|Hezbollah|Syria|strike|ceasefire|war)',
        r'Hamas|Hezbollah|Gaza|West Bank',
        r'Iran.*(?:strike|nuclear|sanction|enrichment|attack)',
        r'US strikes? (?:Iran|Yemen|Syria)',
        r'Khamenei|Houthi|Yemen',
        r'Saudi.*(?:oil|OPEC|Iran)',
        r'Suez Canal',
    ],
    'china_geopolitics': [
        r'China.*(?:Taiwan|US|trade|tariff|ban|sanction|military|invasion|blockade)',
        r'Taiwan.*(?:China|invasion|blockade)',
        r'TikTok.*ban', r'Xi Jinping',
        r'Xinjiang|Hong Kong.*(?:protest|autonomy)',
        r'South China Sea', r'AUKUS',
    ],
    'crypto_digital': [
        r'Bitcoin|Ethereum|BTC|ETH|Solana|Dogecoin|XRP|Cardano|Polkadot',
        r'crypto.*(?:ban|regulate|ETF|market cap|price|reach)',
        r'altcoin', r'DeFi', r'NFT\b',
        r'(?:FDV|market cap).*(?:above|below|launch)',
        r'Coinbase|Binance|Kraken|Gemini|OpenSea|Uniswap|MetaMask',
        r'IPO.*(?:Coinbase|Kraken|Circle|Ripple)',
        r'stablecoin', r'CBDC',
        r'MicroStrategy.*Bitcoin',
        r'(?:FDV|market cap).*(?:above|below|one day after launch)',
        r'launch a token', r'token.*launch',
        r'\b(?:BNB|SOL|DOGE|ADA|AVAX|MATIC|DOT|LINK|UNI|AAVE)\b.*(?:reach|dip|price|above|below)',
        r'Hyperliquid.*(?:dip|reach|price)',
        r'Fomo.*token',
    ],
    'ai_technology': [
        r'\bAI\b.*(?:model|downturn|regulate|ban|safety)',
        r'GPT|Claude|Gemini|Grok|Llama|Mistral',
        r'OpenAI|Anthropic|Google.*AI|DeepMind|xAI',
        r'AGI\b', r'artificial.*intelligence',
        r'(?:AI|ML).*(?:benchmark|frontier|safety)',
        r'self.driving|autonomous.*vehicle|FSD\b|Robotaxi',
        r'AI browser', r'AI model',
        r'Elon Musk.*(?:tweet|post|Truth Social|X \(Twitter\))',
        r'best AI model',
    ],
    'us_economic': [
        r'GDP.*(?:growth|recession|percent|quarter)',
        r'inflation.*(?:reach|above|below|percent|CPI|Annual Inflation)',
        r'unemployment.*(?:rate|reach|above|below)',
        r'recession\b', r'government shutdown',
        r'tariff|trade war|trade deal',
        r'debt ceiling', r'deficit', r'national debt',
        r'S&P.*(?:500|close|above|below|reach)',
        r'Nasdaq|Dow Jones|stock market',
        r'IPO.*(?:202[4-9]|before)',
        r'Silver.*hit|Gold.*hit|Oil.*(?:price|above|below)',
        r'(?:NVDA|NVIDIA|AMZN|Amazon|AAPL|Apple|GOOGL|Google|MSFT|Microsoft|TSLA|Tesla|META|Netflix|NFLX|HD|Home Depot).*(?:close|above|below|reach|dip|price|beat|earnings)',
        r'(?:close above|close below|reach \$|dip to \$).*(?:end of|in (?:January|February|March|April|May|June|July|August|September|October|November|December))',
        r'quarterly earnings', r'beat.*earnings',
        r'market cap.*(?:be between|above|below)',
        r'Crude Oil.*(?:hit|above|below)',
        r'(?:acquire|acquisition|merger|buy).*(?:Warner|TikTok|Twitter)',
        r'announce bankruptcy',
        r'target federal fund', r'upper bound.*target',
        r'State of the Union',
        r'\$\d+.*(?:trillion|billion).*(?:debt|deficit|raised)',
        r'Clear Street|Palantir.*(?:dip|reach|price)',
        r'Opendoor.*reach',
    ],
    'energy_commodities': [
        r'oil.*(?:price|above|below|barrel|OPEC)',
        r'OPEC', r'natural gas', r'energy.*(?:crisis|price)',
        r'(?:coal|nuclear|solar|wind).*(?:energy|power|plant)',
        r'pipeline', r'LNG\b',
    ],
    'climate_environment': [
        r'climate.*(?:change|accord|paris|target)',
        r'temperature.*(?:record|hottest|above|warming)',
        r'highest temperature in', r'lowest temperature in',
        r'earthquake', r'hurricane|typhoon|cyclone',
        r'wildfire', r'flood', r'drought',
        r'meteor.*strike', r'natural disaster',
        r'(?:CO2|carbon|emission).*(?:level|target|reduce)',
        r'hottest.*record',
        r'measles cases',
    ],
    'pandemic_health': [
        r'COVID|coronavirus|pandemic|epidemic',
        r'(?:new|next).*(?:pandemic|variant|outbreak)',
        r'vaccine.*(?:mandate|approve|rollout)',
        r'WHO.*(?:declare|emergency)',
        r'bird flu|H5N1|mpox|monkeypox',
    ],
    'legal_regulatory': [
        r'(?:charged|convicted|indicted|sentenced|arrested|sued)',
        r'(?:Supreme Court|SCOTUS).*(?:rule|overturn|uphold|accept)',
        r'Insurrection Act', r'martial law',
        r'(?:ban|regulate|legalize).*(?:marijuana|cannabis|drug|gun|abortion)',
        r'(?:sanction|embargo).*(?:impose|lift)',
        r'ICE.*(?:shoot|charged|arrest)',
        r'(?:pardon|commute|clemency)',
        r'Hegseth|Gabbard.*confirm', r'Susie Wiles', r'Epstein',
        r'(?:FBI|DOJ|SEC|FTC).*(?:investigate|charge|fine)',
        r'Trump.*(?:resign|impeach|25th)',
        r'(?:confirm|nomination|Secretary|cabinet).*(?:Defense|State|Treasury|AG|Attorney General)',
    ],
    'us_military': [
        r'US.*(?:strike|strikes|forces|troops|military).*(?:Iran|Yemen|Syria|Cuba|Venezuela|Mexico|country|countries)',
        r'anti.cartel.*operation',
        r'US military action',
        r'US x .*military clash',
    ],
    'space_frontier': [
        r'(?:SpaceX|NASA|Blue Origin|Rocket Lab).*(?:launch|land|mission)',
        r'Mars.*(?:mission|land|human)',
        r'Moon.*(?:land|mission|base)',
        r'(?:Starship|Falcon|SLS|Artemis).*(?:launch|test|orbit)',
        r'asteroid|comet.*(?:impact|near)',
        r'alien|extraterrestrial|UFO|UAP',
        r'space.*(?:station|tourism|colony)',
    ],
    'venezuela': [
        r'Venezuela|Maduro|Guaidó',
        r'US.*(?:forces|troops|military).*Venezuela',
    ],
}


def classify_market(title: str) -> str:
    """Classify a market title into a theme. Returns theme string."""
    for theme, patterns in THEME_RULES.items():
        for pattern in patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return theme
    return 'uncategorized'


def classify_markets(df: pd.DataFrame, title_col: str = 'title') -> pd.DataFrame:
    """Add 'theme' column to DataFrame."""
    df = df.copy()
    df['theme'] = df[title_col].apply(classify_market)
    return df


if __name__ == '__main__':
    import sys
    from collections import Counter
    
    df = pd.read_parquet('data/processed/markets_filtered.parquet')
    df = classify_markets(df)
    
    themes = Counter(df['theme'])
    print("Theme distribution:")
    for t, c in themes.most_common():
        print(f"  {t}: {c}")
    
    print(f"\nTotal: {len(df)}")
    print(f"Uncategorized: {themes['uncategorized']} ({themes['uncategorized']/len(df)*100:.1f}%)")
    
    # Show uncategorized samples
    uncat = df[df['theme'] == 'uncategorized']
    if len(uncat) > 0:
        print(f"\nSample uncategorized:")
        print(uncat['title'].sample(min(20, len(uncat))).to_string())
