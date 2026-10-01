# SM1: industry momentum on NSE

Implements reports/preregistration_sector_momentum.md. Each month end: industry score = equal-weight 12-1 month return of its eligible members (industries with >= 5 members); hold the eligible stocks of the top 3 industries equally from the next day to the next month end; benchmark = the equal-weight eligible universe; 25 bps per unit of turnover.

Industry labels: Yahoo `industry` for 1,009 tickers (986 labelled). These are today's labels. Unlabelled stocks are dropped from the portfolios (they stay in the benchmark).

## Development (month ends 2016-01 to 2020-12)

| Portfolio | Months | Return/yr (net) | Excess/yr vs EW | Month t | Months beating EW |
|---|---|---|---|---|---|
| **Primary: top 3 industries, 12-1** | 51 | +13.6% | +2.3% | +0.37 | 55% |
| Top 1 industry, 12-1 | 51 | +14.3% | +3.0% | +0.31 | 49% |
| Top 5 industries, 12-1 | 51 | +13.2% | +1.8% | +0.45 | 53% |
| Top 3 industries, 6-1 | 51 | +13.5% | +2.2% | +0.38 | 47% |
| Bottom 3 industries, 12-1 | 51 | -2.3% | -13.6% | -2.21 | 39% |

Equal-weight eligible universe: 11.3% a year over 51 months; about 203 eligible stocks (202 labelled) and 13 ranked industries per month.

Share of months with these among the top 3 (12-1): Aerospace & Defense 0%, utilities 0%, banks 6%, energy 25%.

Rule: excess >= +3.0% a year, month t >= 2.0, beats EW in >= 55% of months.

Note: the first month end used is 2016-10-28, not 2016-01. `eligible_at` needs 2 years (504 rows) of price history and the daily cache starts on 2014-09-30, so no stock is eligible earlier. This follows from the pre-registered eligibility rule, not from a choice made after seeing results.

**Decision: does not qualify; the test stays sealed.**

