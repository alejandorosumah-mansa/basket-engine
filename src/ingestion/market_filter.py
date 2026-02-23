"""
Filter out non-financial markets (sports, entertainment, pop culture) from the dataset.
Aditis focuses on economic, political, and financial prediction markets.
"""
import pandas as pd
import re

# ============================================================
# SPORTS — comprehensive patterns
# ============================================================
SPORTS_PATTERNS = [
    # League names
    r'\bNBA\b', r'\bNFL\b', r'\bNHL\b', r'\bMLB\b', r'\bUFC\b', r'\bFIFA\b',
    r'\bMLS\b', r'\bWNBA\b', r'\bNCAAF?\b', r'\bPGA\b', r'\bATP\b', r'\bWTA\b',
    r'\bF1\b', r'\bNASCAR\b', r'\bXFL\b', r'\bUSFL\b',
    r'Super Bowl', r'Premier League', r'Champions League', r'Europa League',
    r'World Series', r'La Liga', r'Serie A\b', r'Bundesliga', r'Ligue 1',
    r'Grand Prix', r'Stanley Cup', r'World Cup', r'Davis Cup',
    r'March Madness', r'Final Four', r'College Football',
    r'Conference Finals', r'Eastern Conference', r'Western Conference',
    r'Playoff', r'Postseason',
    
    # Event types
    r'boxing match', r'\bPPV\b', r'title fight', r'title bout',
    r'Poker Championship', r'poker', r'Poker',
    r'Olympic', r'Olympics', r'Paralympic',
    r'medal at the', r'gold medal', r'most medals',
    r'win the \d{4}.*Finals', r'win the \d{4}.*Series',
    r'win the \d{4}.*Bowl', r'win the \d{4}.*Cup',
    
    # Game lines / spreads / player props (Polymarket sports betting)
    r'vs\.\s', r'\bvs\b',  # "Team vs. Team" or "Team vs Team"
    r'O/U \d', r'Over/Under', r'Over \d+\.\d', r'Under \d+\.\d',
    r'Anytime Touchdown', r'First Touchdown',
    r': Points O', r': Points Over', r': Points Under',
    r': Rebounds O', r': Rebounds Over', r': Rebounds Under',
    r': Assists O', r': Assists Over', r': Assists Under',
    r': Steals O', r': Steals Over',
    r': Blocks O', r': Blocks Over',
    r': Three.?Pointers', r': Home Runs', r': Strikeouts',
    r': Passing Yards', r': Rushing Yards', r': Receiving Yards',
    r': Touchdowns', r': Receptions',
    r'1H O/U', r'1Q O/U', r'Moneyline',
    r'Spread [+-]', r'Point Spread',
    r'Total [OoUu]', r'Game Total',
    
    # Team names - NBA
    r'\b(?:76ers|Blazers|Bucks|Bulls|Cavaliers|Celtics|Clippers|Grizzlies|Hawks|Heat|Hornets|Jazz|Kings|Knicks|Lakers|Magic|Mavericks|Nets|Nuggets|Pacers|Pelicans|Pistons|Raptors|Rockets|Spurs|Suns|Thunder|Timberwolves|Trail Blazers|Warriors|Wizards)\b',
    # Team names - NFL
    r'\b(?:49ers|Bears|Bengals|Bills|Broncos|Browns|Buccaneers|Cardinals|Chargers|Chiefs|Colts|Commanders|Cowboys|Dolphins|Eagles|Falcons|Giants|Jaguars|Jets|Lions|Packers|Panthers|Patriots|Raiders|Rams|Ravens|Saints|Seahawks|Steelers|Texans|Titans|Vikings)\b',
    # Team names - MLB
    r'\b(?:Yankees|Dodgers|Braves|Astros|Phillies|Padres|Mets|Orioles|Rangers|Twins|Rays|Guardians|Mariners|Red Sox|Blue Jays|Cubs|Diamondbacks|Brewers|Pirates|Reds|Rockies|Marlins|Nationals|White Sox|Royals|Tigers|Athletics|Angels)\b',
    # Team names - NHL
    r'\b(?:Avalanche|Blackhawks|Blue Jackets|Bruins|Canadiens|Canucks|Capitals|Coyotes|Devils|Ducks|Flames|Flyers|Golden Knights|Hurricanes|Islanders|Kraken|Kings|Lightning|Maple Leafs|Oilers|Penguins|Predators|Red Wings|Sabres|Senators|Sharks|Stars|Wild)\b',
    
    # Soccer / Football clubs
    r'\b(?:Manchester United|Manchester City|Liverpool|Arsenal|Chelsea|Tottenham|Aston Villa|Newcastle|West Ham|Brighton|Nottingham Forest|Everton|Fulham|Bournemouth|Wolves|Crystal Palace|Brentford|Southampton|Leicester|Leeds|Ipswich)\b',
    r'\b(?:Barcelona|Real Madrid|Atletico|Sevilla|Valencia|Villarreal|Real Sociedad|Athletic Bilbao|Betis|Getafe|Girona|Celta|Mallorca|Osasuna|Espanyol)\b',
    r'\b(?:Bayern Munich|Borussia Dortmund|RB Leipzig|Bayer Leverkusen|Eintracht Frankfurt|VfB Stuttgart|Union Berlin|Werder Bremen|SC Freiburg)\b',
    r'\b(?:Juventus|Inter Milan|AC Milan|Napoli|AS Roma|Lazio|Atalanta|Fiorentina|Bologna|Torino)\b',
    r'\b(?:PSG|Paris Saint.Germain|Olympique|Lyon|Marseille|Monaco|Lille|Rennes|Nice|Lens)\b',
    r'\b(?:Ajax|PSV|Feyenoord|Benfica|Porto|Sporting CP|Celtic|Rangers|Galatasaray|Fenerbahce|Besiktas|Olympiakos|Club Brugge|Anderlecht|Shakhtar|Red Bull Salzburg|Slavia Prague)\b',
    r'\b(?:Palermo|Südtirol|Como|Sampdoria|Bari|Brescia|Catanzaro|Cosenza|Cremonese|Frosinone|Modena|Pisa|Reggiana|Salernitana|Sassuolo|Spezia|Sudtirol)\b',
    r'Union Saint.Gilloise',
    
    # Player prop patterns (name: stat)
    r'[A-Z][a-z]+ [A-Z][a-z]+: (?:Points|Rebounds|Assists|Steals|Blocks|Three|Passing|Rushing|Receiving|Touchdowns|Receptions|Home Runs|Strikeouts|Hits|RBIs|Saves|Goals|Shots)',
    
    # Generic sports
    r'\bcoach\b.*\bfired\b', r'\bhead coach\b',
    r'\btrade deadline\b', r'\bfree agent\b', r'\broster\b',
    r'\bdraft pick\b', r'\bMVP\b(?!.*(?:election|president|award show))',
    r'win (?:the )?(?:202\d[-–]?\d{0,2} )?(?:English |Spanish |Italian |German |French )?(?:Premier|Champions|Europa|Bundesliga)',
    # Golf
    r'(?:Masters|PGA|Open Championship|US Open|Ryder Cup).*(?:tournament|golf|win)',
    r'(?:Schauffele|McIlroy|Scheffler|Rahm|Koepka|DeChambeau|Morikawa|Hovland|Spieth|Thomas|Woodland|Bridgeman|Conners).*(?:win|golf|Masters|tournament|Genesis)',
    r'win the 202\d.*(?:Masters|Open|PGA|Genesis|Players)',
    # Tennis  
    r'(?:Sinner|Djokovic|Alcaraz|Medvedev|Zverev|Rublev|Ruud|Fritz|Tsitsipas|Swiatek|Sabalenka|Gauff|Rybakina|Pegula).*(?:win|tennis|Wimbledon|Roland|Australian|US Open)',
    r'win (?:Wimbledon|Roland Garros|Australian Open)',
    r'Calendar Grand Slam',
    # World Baseball Classic
    r'World Baseball Classic',
    # Catch-all: "Will X win on YYYY-MM-DD" (daily game results)
    r'win on 202\d-\d{2}-\d{2}',
    # Fantasy sports
    r'Fantasy (?:Flex|Football|Basketball|Baseball|Hockey)',
    r'Most kills', r'Series:.*kills',
    # Generic team sport patterns
    r'\d\+? (?:Touchdowns|Goals|Assists|Rebounds|Points|Strikeouts|Home Runs)',
]

# ============================================================
# ENTERTAINMENT / POP CULTURE
# ============================================================
ENTERTAINMENT_PATTERNS = [
    r'top grossing movie', r'box office', r'gross most', r'highest.grossing',
    r'Spotify artist', r'top.*artist.*(?:202\d|of the year)',
    r'Grammy', r'Oscar', r'Academy Award', r'Emmy', r'Golden Globe',
    r'BAFTA', r'SAG Award', r'Cannes', r'Venice Film', r'Sundance', r'Tony Award', r'Pulitzer',
    r'album.*chart', r'Billboard', r'number.?one.*(?:song|album|single)',
    r'(?:Ariana Grande|Taylor Swift|Drake|Kanye|Bad Bunny|The Weeknd|Beyonce|Rihanna|Ed Sheeran|BTS).*(?:artist|song|album|stream|chart|sales|debut)',
    r'top.*(?:song|album|artist).*(?:202\d|of the year)',
    r'(?:Despicable|Jurassic|Fantastic Four|Avatar|Star Wars|Marvel|Disney|Pixar|Barbie|Oppenheimer|Wicked|How to Train Your Dragon).*(?:gross|box office|top|win)',
    r'next James Bond',
    r'Bachelor(?:ette)?(?:\s|$)',
    r'Love Island', r'Love is Blind', r'Big Brother', r'Beast Games',
    r'(?:win|host).*(?:SNL|Saturday Night Live)',
    r'reality (?:TV|show|series)',
    r'Best (?:Film|Actor|Actress|Director|Supporting|Picture|Screenplay|Original|Animated|Documentary|Visual|Special|Performance|Musical)',
    r'Outstanding (?:Performance|Drama|Comedy|Limited|Series)',
    r'(?:win|nominated).*(?:BAFTA|SAG|Cannes|Venice|Tony|Pulitzer)',
    r'UMK\b', r'Eurovision',
    r'SSA.*(?:names|baby|rank)',
    r'(?:boy|girl) names? on the SSA',
    r'Trump.*(?:say "|word "|Penguin)',
    r'Elon Musk.*(?:post \d|tweet \d|\d+ tweets)',
]

# Compile
_SPORTS_RE = re.compile('|'.join(SPORTS_PATTERNS), re.IGNORECASE)
_ENTERTAINMENT_RE = re.compile('|'.join(ENTERTAINMENT_PATTERNS), re.IGNORECASE)

# Whitelist: things that LOOK like sports but are actually political/financial
_WHITELIST_RE = re.compile(
    r'Senate race|Governor race|congressional|election|presidential|'
    r'party.*win|balance of power|electoral|delegate|primary|caucus|'
    r'cabinet|confirmation|impeach|GDP|inflation|Fed\b|interest rate|'
    r'tariff|trade war|sanctions|IPO|stock|Bitcoin|Ethereum|crypto|'
    r'cease.?fire|military|war |invasion|strike.*(?:Iran|Syria|Yemen|Russia)',
    re.IGNORECASE
)


def is_sports(title: str) -> bool:
    if _WHITELIST_RE.search(title):
        return False
    return bool(_SPORTS_RE.search(title))


def is_entertainment(title: str) -> bool:
    if _WHITELIST_RE.search(title):
        return False
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
    
    # Show some edge cases
    print("\n--- Sample KEPT markets (spot check) ---")
    print(filtered['title'].sample(20).to_string())
