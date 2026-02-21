"""Detect phrasing polarity and token side from market titles.

Determines whether a YES token price increase represents a "positive" or
"negative" real-world outcome, accounting for negation patterns in market
phrasing.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Negation patterns that flip polarity.
# If a title matches any of these, YES = negative phrasing (the event described
# is a non-occurrence or downside scenario).
NEGATION_PATTERNS = [
    # Explicit negation
    r"\bnot\b",
    r"\bno\b(?!\s+\d)",  # "no" but not "no. 1"
    r"\bnot\b",
    r"\bwon'?t\b",
    r"\bwill\s+not\b",
    r"\bwon'?t\b",
    r"\bdoes\s*n'?t\b",
    r"\bdo\s*n'?t\b",
    r"\bcan'?t\b",
    r"\bcannot\b",
    r"\bfail(s|ed)?\s+to\b",
    r"\brefuse[sd]?\s+to\b",
    r"\bneither\b",
    r"\bnever\b",
    # Comparative negatives
    r"\bless\s+than\b",
    r"\bunder\b(?!\s*(secretary|dog|armour|world))",
    r"\bbelow\b",
    r"\bfewer\s+than\b",
    r"\bfall(s|ing)?\b.*\b(below|under)\b",
    r"\bdecline\b",
    r"\bdecrease\b",
    r"\bdrop(s|ping)?\s+(below|under|to)\b",
    r"\blose[s]?\b",
    # Negative outcomes
    r"\bbe\s+eliminated\b",
    r"\bbe\s+fired\b",
    r"\bbe\s+removed\b",
    r"\bbe\s+defeated\b",
    r"\bdefault\b",
    r"\brecession\b",
    r"\bshutdown\b",
    r"\bveto(ed|es|ing)?\b",
    r"\bwithdraw(s|al)?\b",
]

# Compiled patterns (case-insensitive)
_NEGATION_RE = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]

# Positive outcome patterns – these override negation if both match
# (e.g., "Will Bitcoin NOT fall below 50K?" has negation of a negative → positive)
DOUBLE_NEGATION_PATTERNS = [
    r"\bnot\b.*\b(fall|drop|decline|decrease|lose|fail)\b",
    r"\bwon'?t\b.*\b(fall|drop|decline|decrease|lose|fail)\b",
    r"\bavoid\b.*\b(recession|default|shutdown)\b",
]
_DOUBLE_NEGATION_RE = [re.compile(p, re.IGNORECASE) for p in DOUBLE_NEGATION_PATTERNS]

# Over/under patterns for sports/numeric markets
OVER_UNDER_PATTERNS = [
    (r"\b(over|above|more\s+than|higher\s+than|exceed|surpass)\b", "positive"),
    (r"\b(under|below|less\s+than|lower\s+than|fewer\s+than)\b", "negative"),
]
_OVER_UNDER_RE = [(re.compile(p, re.IGNORECASE), d) for p, d in OVER_UNDER_PATTERNS]


def detect_phrasing_polarity(title: str) -> str:
    """Detect whether a market title is phrased positively or negatively.

    Returns:
        'positive' - YES means something happens/increases (default)
        'negative' - YES means something does NOT happen or decreases
        'neutral'  - ambiguous / categorical outcome
    """
    if not title:
        return "neutral"

    title_clean = title.strip()

    # Check double negation first (negation of a negative = positive)
    for pat in _DOUBLE_NEGATION_RE:
        if pat.search(title_clean):
            return "positive"

    # Check negation patterns
    negation_count = sum(1 for pat in _NEGATION_RE if pat.search(title_clean))
    if negation_count > 0:
        return "negative"

    # Check over/under patterns
    for pat, direction in _OVER_UNDER_RE:
        if pat.search(title_clean):
            return direction

    return "positive"


def detect_token_side(
    title: str,
    outcomes: Optional[list[str]] = None,
    tracked_token_index: int = 0,
) -> str:
    """Determine which token side we're tracking.

    For binary markets, we track the YES token (index 0).
    For categorical markets, the specific outcome name.

    Returns:
        'YES', 'NO', or the outcome name for categorical markets.
    """
    if outcomes and len(outcomes) > 2:
        # Categorical market - return the specific outcome
        if tracked_token_index < len(outcomes):
            return outcomes[tracked_token_index]
        return outcomes[0]

    # Binary market
    if outcomes and tracked_token_index < len(outcomes):
        return outcomes[tracked_token_index].upper()
    return "YES"


def compute_exposure_direction(
    phrasing_polarity: str,
    token_side: str,
) -> str:
    """Compute the real-world exposure direction.

    Combines phrasing polarity with token side to determine what "price goes up" means
    in terms of real-world outcomes.

    Returns:
        'long'  - price up = positive real-world outcome (event happens, value increases)
        'short' - price up = negative real-world outcome (event doesn't happen, value decreases)
    """
    # Truth table:
    # positive phrasing + YES token = long  (event happens = good)
    # positive phrasing + NO token  = short (event doesn't happen)
    # negative phrasing + YES token = short (bad thing confirmed)
    # negative phrasing + NO token  = long  (bad thing avoided)
    # neutral + YES = long (default assumption)
    # neutral + NO  = short

    is_yes = token_side.upper() in ("YES", "")
    is_negative = phrasing_polarity == "negative"

    if is_yes:
        return "short" if is_negative else "long"
    else:
        return "long" if is_negative else "short"


def detect_side_batch(
    markets_df,
    title_col: str = "title",
    outcomes_col: Optional[str] = None,
) -> "pd.DataFrame":
    """Add side/exposure columns to a markets DataFrame.

    Adds columns:
        - token_side: YES/NO/outcome name
        - phrasing_polarity: positive/negative/neutral
        - exposure_direction: long/short
        - normalized_direction: 1.0 (long) or -1.0 (short)

    Returns:
        DataFrame with new columns added.
    """
    import pandas as pd

    df = markets_df.copy()

    df["phrasing_polarity"] = df[title_col].apply(detect_phrasing_polarity)

    # Token side
    if outcomes_col and outcomes_col in df.columns:
        df["token_side"] = df.apply(
            lambda r: detect_token_side(
                r[title_col],
                r[outcomes_col] if isinstance(r[outcomes_col], list) else None,
            ),
            axis=1,
        )
    else:
        df["token_side"] = "YES"

    df["exposure_direction"] = df.apply(
        lambda r: compute_exposure_direction(r["phrasing_polarity"], r["token_side"]),
        axis=1,
    )

    df["normalized_direction"] = df["exposure_direction"].map({"long": 1.0, "short": -1.0})

    polarity_counts = df["phrasing_polarity"].value_counts()
    logger.info(f"Side detection: {dict(polarity_counts)}")

    return df
