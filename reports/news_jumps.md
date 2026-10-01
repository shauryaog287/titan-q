# NJ1: news-backed vs no-news jumps

Implements reports/preregistration_news_jumps.md. Jump = +5% close-to-close on 3x the 20-day volume, liquid stocks (>= Rs. 1 crore/day). Excess vs the equal-weight liquid universe from the close of the jump day.

## Development (event dates 2017-01 to 2020-12)

| Group | Events | Mean excess (net of 0.5% for buys) | Date-clustered t | Hit rate |
|---|---|---|---|---|
| **Primary: news-backed jumps, 5 days** | 1,399 | -0.25% | -1.07 | 41.9% |
| no_news_5d | 4,258 | -1.00% | -6.37 | 36.0% |
| news_1d | 1,399 | -0.28% | -1.48 | 39.0% |
| news_20d | 1,399 | -0.05% | +0.12 | 43.3% |
| no_news_1d | 4,258 | -0.38% | -4.36 | 36.7% |
| no_news_20d | 4,258 | -1.39% | -5.51 | 36.9% |
| news_jump_10pct_5d | 362 | -0.70% | -1.53 | 37.8% |
| down_news_5d | 624 | -1.27% | -3.37 | 40.9% |
| down_no_news_5d | 597 | +18.43% | +1.00 | 50.3% |

Down moves are shown without cost (they are not trades here).

Rule: news-backed 5-day net excess >= +0.5%, t >= 2.0, >= 150 events.

**Decision: does not qualify; the test stays sealed.**

