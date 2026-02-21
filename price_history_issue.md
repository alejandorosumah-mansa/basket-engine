# GitHub Issue: Price History Coverage Gap

## Issue
Only 2,783 out of 20,366 markets (13.7%) have price history data, leaving 17,583 markets without price data.

## Root Cause Analysis
1. **Volume Filter Too Restrictive**: Polymarket ingestion only fetches candlesticks for markets with >$5M volume
2. **API Endpoint Limitations**: Some markets may not be available via the candlesticks endpoint
3. **Error Handling**: Failed API calls result in no data, no retry with alternative methods

## Impact
- Severe data coverage gap reduces basket construction universe
- Only 340 markets pass eligibility (need 14 days price history)
- Missing high-volume, active markets due to arbitrary threshold

## Platform Breakdown
- **Kalshi**: 175/175 markets have price data (100%)
- **Polymarket**: 2,608/20,191 markets have price data (12.9%)

## Status Breakdown
- **Closed markets**: 23.8% have price data
- **Open markets**: Only 3.7% have price data (this is critical for basket construction!)

## Proposed Fixes
1. ✅ **Reduced volume threshold** from $5M to $500K 
2. ✅ **Added fallback method** using point-in-time price API when candlesticks fail
3. 🔄 **TODO**: Test alternative endpoints (Dome vs native CLOB)
4. 🔄 **TODO**: Implement retry logic with exponential backoff
5. 🔄 **TODO**: Consider batch price fetching for efficiency

## Expected Impact
- Should increase Polymarket coverage from 12.9% to 40-60%
- More markets eligible for basket construction
- Better representation of active trading universe

## Testing Plan
- Re-run ingestion with new settings
- Verify coverage improvement
- Check data quality of fallback method
- Monitor API rate limits and performance