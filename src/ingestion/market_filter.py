"""
Filter out non-financial markets (sports, entertainment) from the dataset.
Aditis focuses on economic, political, and financial prediction markets.
"""
import pandas as pd
import re

# Sports patterns
SPORTS_PATTERNS = [
    r'\bNBA\b', r'\bNFL\b', r'\bNHL\b', r'\bMLB\b', r'\bUFC\b', r'\bFIFA\b',
    r'\bF1\b', r'Super Bowl', r'Premier League', r'Champions League',
    r'World Series', r'La Liga', r'Serie A\b', r'Bundesliga',
    r'Grand Prix', r'boxing match', r'win the \d{4}.*Finals',
    r'win the \d{4}.*Series', r'win the \d{4}.*Bowl',
    r'win (?:the )?(?:202\d[-–]?\d{2} )?(?:English |Spanish |Italian |German |French )?(?:Premier|Champions|Europa|La Liga|Serie|Bundesliga)',
    r'NBA Finals', r'Stanley Cup', r'World Cup',
    r'(?:Pacers|Lakers|Celtics|Warriors|Nuggets|Heat|Bucks|Suns|Knicks|Nets|Bulls|Hawks|Cavaliers|Pistons|Rockets|Spurs|Mavericks|Grizzlies|Pelicans|Thunder|Trail Blazers|Timberwolves|Clippers|Kings|Jazz|Wizards|Hornets|Magic|Raptors) win',
    r'(?:Patriots|Chiefs|Eagles|49ers|Cowboys|Bills|Ravens|Dolphins|Bengals|Lions|Packers|Steelers|Chargers|Broncos|Jets|Raiders|Saints|Falcons|Panthers|Cardinals|Bears|Commanders|Texans|Colts|Jaguars|Titans|Browns|Seahawks|Buccaneers|Giants|Vikings|Rams) win',
    r'(?:Yankees|Dodgers|Braves|Astros|Phillies|Padres|Mets|Orioles|Rangers|Twins|Rays|Guardians|Mariners|Red Sox|Blue Jays|Cubs|Diamondbacks|Brewers|Pirates|Reds|Cardinals|Rockies|Marlins|Nationals|White Sox|Royals|Tigers|Athletics|Angels) win',
    r'(?:Manchester|Liverpool|Arsenal|Chelsea|Tottenham|Aston Villa|Newcastle|West Ham|Brighton|Nottingham|Everton|Fulham|Bournemouth|Wolves|Crystal Palace|Brentford|Southampton|Leicester|Leeds|Ipswich) (?:United |City |Forest |Palace )?wins?',
    r'(?:Barcelona|Real Madrid|Atletico|Sevilla|Valencia|Villarreal|Real Sociedad|Athletic Bilbao|Betis|Getafe|Girona|Celta|Mallorca|Rayo|Osasuna|Alaves|Valladolid|Leganes|Las Palmas|Espanyol) wins?',
    r'(?:Bayern|Borussia|Leipzig|Leverkusen|Freiburg|Frankfurt|Hoffenheim|Wolfsburg|Stuttgart|Union Berlin|Werder|Mainz|Augsburg|Heidenheim|Darmstadt|Koln|Bochum|Gladbach) wins?',
    r'(?:Juventus|Inter Milan|AC Milan|Napoli|Roma|Lazio|Atalanta|Fiorentina|Bologna|Torino|Monza|Genoa|Lecce|Cagliari|Empoli|Udinese|Verona|Sassuolo|Salernitana|Frosinone) wins?',
    r'(?:PSG|Paris Saint|Olympique|Lyon|Marseille|Monaco|Lille|Rennes|Nice|Strasbourg|Lens|Nantes|Toulouse|Montpellier|Reims|Brest|Lorient|Clermont|Metz|Havre) wins?',
    r'(?:Ajax|PSV|Feyenoord|Benfica|Porto|Sporting|Celtic|Rangers|Galatasaray|Fenerbahce|Besiktas|Olympiakos|Club Brugge|Anderlecht|Union Saint|Shakhtar|Dinamo|Slavia|Sparta|Red Bull Salzburg) wins?',
]

# Entertainment patterns  
ENTERTAINMENT_PATTERNS = [
    r'top grossing movie', r'box office', r'gross most',
    r'Spotify artist', r'Grammy', r'Oscar', r'Academy Award',
    r'album.*chart', r'Billboard', r'Emmy',
    r'Ariana Grande.*(?:artist|song|album)', r'Taylor Swift.*(?:artist|song|album)',
    r'Drake.*(?:artist|song|album)', r'Kanye.*(?:artist|song|album)',
    r'top.*(?:song|album|artist).*(?:202\d|of the year)',
    r'(?:Despicable|Jurassic|Fantastic Four|Avatar|Star Wars|Marvel|Disney).*(?:gross|box office|top)',
]

# Compile all patterns
_SPORTS_RE = re.compile('|'.join(SPORTS_PATTERNS), re.IGNORECASE)
_ENTERTAINMENT_RE = re.compile('|'.join(ENTERTAINMENT_PATTERNS), re.IGNORECASE)


def is_sports(title: str) -> bool:
    return bool(_SPORTS_RE.search(title))


def is_entertainment(title: str) -> bool:
    return bool(_ENTERTAINMENT_RE.search(title))


def filter_markets(df: pd.DataFrame, title_col: str = 'title') -> pd.DataFrame:
    """Remove sports and entertainment markets. Returns filtered DataFrame."""
    mask_sports = df[title_col].apply(is_sports)
    mask_ent = df[title_col].apply(is_entertainment)
    mask_remove = mask_sports | mask_ent
    
    n_sports = mask_sports.sum()
    n_ent = mask_ent.sum()
    n_total = mask_remove.sum()
    
    print(f"[market_filter] Removing {n_total} non-financial markets:")
    print(f"  Sports: {n_sports}")
    print(f"  Entertainment: {n_ent}")
    print(f"  Remaining: {len(df) - n_total} / {len(df)}")
    
    return df[~mask_remove].copy()


if __name__ == '__main__':
    import sys
    df = pd.read_parquet('data/processed/markets.parquet')
    filtered = filter_markets(df)
    
    if '--save' in sys.argv:
        filtered.to_parquet('data/processed/markets_filtered.parquet', index=False)
        print(f"Saved to data/processed/markets_filtered.parquet")
