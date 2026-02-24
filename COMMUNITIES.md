# COMMUNITIES.md - Final Market Baskets

This document describes the 16 high-quality market communities identified after implementing strict clustering and LLM validation.

## Executive Summary

- **Total Communities:** 16 (down from hundreds of unfiltered clusters)
- **Total Markets Included:** 1,332 (down from 1,800 after filtering)
- **Markets Excluded:** 468 markets (removed as outliers or in communities too small)
- **Quality Filters Applied:**
  - Minimum 30 overlapping days for correlation calculation
  - Minimum 0.5 correlation threshold for graph edges
  - Minimum 10 markets per community (after LLM validation: minimum 8)
  - LLM validation removed 225+ outliers across all communities

## Methodology

1. **Fixed Correlation Matrix:** Enforced 30-day minimum overlap requirement (was ignored before)
2. **Strict Graph Filtering:** Used 0.5 correlation threshold (vs previous 0.3)  
3. **Community Size Filtering:** Required minimum 10 markets per community
4. **LLM Outlier Removal:** GPT-4 analyzed each community and identified unrelated markets
5. **Final Size Filter:** Kept communities with 8+ markets after outlier removal

## Final Communities

### Community 11: Mixed Political and Economic Predictions (202 markets)
**Theme:** U.S. political events, particularly related to Donald Trump, the Federal Reserve's interest rate decisions, and upcoming elections in the U.S. and other countries. Also includes geopolitical events, especially related to Venezuela and Russia.

**Issues Identified:** This is the largest community and may still contain some heterogeneous content. The LLM removed 79 outliers but significant diversity remains.

**Quality:** Medium-High (large but coherent around US political/economic themes)

---

### Community 19: Political Predictions Cluster (164 markets)  
**Theme:** Political predictions related to the 2028 U.S. presidential election, including nominations and outcomes, as well as Federal Reserve decisions and interest rates. Some international politics included.

**Quality:** High (focused on US political future)

---

### Community 8: Political and Economic Predictions (127 markets)
**Theme:** Political events, particularly related to the U.S. and Brazil, as well as significant geopolitical developments involving Russia and Ukraine. Many markets also revolve around major companies' market capitalizations and Federal Reserve interest rate decisions.

**Quality:** High (strong geopolitical focus)

---

### Community 3: 2024 Election Predictions (110 markets)
**Theme:** The 2024 US Presidential Election, including predictions about candidates, popular vote outcomes, and state-specific results. Many markets also discuss Federal Reserve interest rate implications.

**Quality:** Very High (tightly focused on 2024 election)

---

### Community 16: Political and Economic Predictions (103 markets)
**Theme:** Political events, elections, and economic predictions, particularly related to the U.S. political landscape and cryptocurrency market dynamics. International relations and geopolitical events included.

**Quality:** High (US politics + crypto correlations)

---

### Community 24: Political and Economic Predictions (97 markets)
**Theme:** Political events, particularly elections and leadership changes in various countries, as well as economic factors related to interest rates and financial markets. Strong emphasis on Romania and Poland, alongside US Federal Reserve decisions.

**Quality:** High (international political focus)

---

### Community 10: Political and Crypto Markets (84 markets)
**Theme:** Political events, particularly related to elections and leadership changes, as well as financial predictions concerning Bitcoin and Ethereum. Strong emphasis on U.S. and international political dynamics alongside cryptocurrency movements.

**Quality:** High (politics-crypto correlation captured)

---

### Community 28: 2024 Election Predictions (82 markets)
**Theme:** The 2024 US Presidential Election, including candidates, nominations, and election outcomes, primarily involving Democratic and Republican politicians.

**Quality:** Very High (pure 2024 election focus)

---

### Community 13: U.S. Presidential Predictions (66 markets)
**Theme:** The 2028 U.S. presidential election, including nominations for both Democratic and Republican parties, as well as various political events and figures related to this election cycle.

**Quality:** Very High (pure 2028 election focus)

---

### Community 18: Political Predictions Cluster (66 markets)
**Theme:** Political events, leadership changes, and potential Nobel Peace Prize winners, particularly in relation to prominent global figures and upcoming elections. Some cryptocurrency markets included.

**Quality:** Medium-High (diverse but political-focused)

---

### Community 25: U.S. Political Predictions (63 markets)
**Theme:** U.S. political events, particularly the 2024 Presidential Election, Speaker of the House elections, and Federal Reserve interest rate decisions. Significant emphasis on German elections and political outcomes.

**Quality:** High (US politics + German elections)

---

### Community 0: Iran Conflict and Politics (59 markets)
**Theme:** Geopolitical tensions involving Iran, particularly regarding the leadership of Khamenei and potential military actions by the US and Israel. Some US presidential election markets included.

**Quality:** High (Iran geopolitical risk basket)

---

### Community 12: Political Predictions Cluster (44 markets)
**Theme:** Political events, particularly elections in France and Colombia, as well as U.S. political scenarios. Some economic factors and climate records included.

**Quality:** High (France/Colombia political focus)

---

### Community 29: Political Predictions and IPOs (32 markets)
**Theme:** Political events, particularly related to leadership changes and elections in various countries, as well as specific nominations and geopolitical relations. Notable emphasis on OpenAI market performance.

**Quality:** Medium (mixed themes but small size)

---

### Community 5: Political Elections and Predictions (19 markets)
**Theme:** Political elections, particularly the upcoming presidential elections in South Korea and other countries, along with related political events.

**Quality:** High (South Korea political focus)

---

### Community 2: 2024 Political Predictions (14 markets)
**Theme:** The 2024 Republican Vice Presidential nomination and related political outcomes, particularly within the context of upcoming presidential elections in Virginia and New Jersey.

**Quality:** Very High (2024 VP focus)

---

## Quality Assessment

### Excellent Quality (Very High):
- Community 3: 2024 Election Predictions (110 markets)
- Community 28: 2024 Election Predictions (82 markets) 
- Community 13: U.S. Presidential Predictions (66 markets)
- Community 2: 2024 Political Predictions (14 markets)

### Good Quality (High):
- Community 19: Political Predictions Cluster (164 markets)
- Community 8: Political and Economic Predictions (127 markets)
- Community 16: Political and Economic Predictions (103 markets)
- Community 24: Political and Economic Predictions (97 markets)
- Community 10: Political and Crypto Markets (84 markets)
- Community 25: U.S. Political Predictions (63 markets)
- Community 0: Iran Conflict and Politics (59 markets)
- Community 12: Political Predictions Cluster (44 markets)
- Community 5: Political Elections and Predictions (19 markets)

### Adequate Quality (Medium-High):
- Community 18: Political Predictions Cluster (66 markets)

### Needs Review (Medium):
- Community 11: Mixed Political and Economic Predictions (202 markets) - largest, most diverse
- Community 29: Political Predictions and IPOs (32 markets) - mixed themes

## Key Improvements Achieved

1. **Eliminated Spurious Correlations:** Fixed the min_overlapping_days bug that created correlations from single-day overlaps
2. **Stricter Thresholds:** Raised correlation threshold from 0.3 to 0.5
3. **Size Requirements:** Enforced minimum community sizes
4. **LLM Validation:** Removed 225+ outliers identified as unrelated
5. **Quality Focus:** Reduced from 1,800 markets to 1,332 high-confidence clustered markets

## Remaining Challenges

1. **Community 11** is still quite large (202 markets) and diverse - could potentially be split further
2. **Cross-theme contamination** - some communities have both political and economic themes
3. **Temporal misalignment** - some communities mix 2024 and 2028 election markets
4. **International mixing** - some communities blend US and international politics

## Next Steps

1. Consider splitting Community 11 if correlation sub-structure exists
2. Validate temporal alignment within communities (2024 vs 2028 events)
3. Analyze cross-correlations between communities to identify potential mergers
4. Implement market-specific risk factor analysis for each basket

## Technical Notes

- **Correlation Matrix:** 1,800 x 1,800 markets, 40.8% valid correlations (≥30 days overlap)
- **Graph Density:** 0.0057 (sparse, high-quality connections only)
- **Removed:** 188 isolated markets (no connections above 0.5 threshold)
- **Clustering Algorithm:** Louvain community detection on weighted correlation graph
- **Validation Model:** GPT-4o-mini with structured prompts for outlier identification