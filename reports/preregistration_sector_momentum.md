# Pre-registration: SM1, industry momentum on NSE (written 2026-09-29, BEFORE any data and BEFORE any result)

## Question

Stock-level momentum did not qualify in the whole-market study (reports/whole_market_research.md).
Industry momentum is a separate, well-documented effect (Moskowitz and Grinblatt 1999):
industries that did well over the past year tend to keep doing well. India's 2023-24
public-sector and defence rally, and its later reversal, are the obvious recent example. Does
holding the strongest industries beat the market?

## Data (runs in the cloud)

- Universe, prices and volumes: `universe.py` and the Yahoo daily panels of
  `whole_market_research.load_panels`.
- Eligibility: `whole_market_research.eligible_at`, point in time (median 60-day traded value
  at least Rs. 10 crore, minimum price and history).
- Industry of each stock: Yahoo's `industry` field (yfinance `Ticker(...).info`), fetched once
  for every stock that is eligible at any month end.
  - These are today's labels. A company's industry rarely changes, but this is a
    look-ahead-free assumption only approximately.
  - Stocks without an industry are dropped.

## Portfolio

- **When:** at each month end (row a; all data up to and including a).
- **Industry score:** the equal-weight mean of eligible member stocks' 12-1 month returns,
  where 12-1 means from a-252 to a-21 (the price 21 rows before a, over the price 252 rows
  before a). Only industries with at least 5 eligible members are ranked.
- **Holding:** the eligible stocks of the 3 highest-scoring industries, equal weight, from the
  next row (a+1) to the next month end.
- **Benchmark:** the equal-weight eligible universe over the same days (the same machinery
  as `whole_market_research.monthly_study`).
- **Cost:** 25 bps per unit of turnover.

## Measure

- The monthly excess return of the portfolio over the benchmark: annualised mean, t-statistic
  over months, and the share of months it beats the benchmark.

## Periods

- **Development:** month ends 2016-01 to 2020-12. This needs data back to 2015, which the
  12-year daily cache covers.
- **Test:** 2021-01 to 2026-08. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - the excess is at least +3.0% a year;
  - the monthly t is at least 2.0;
  - the portfolio beats the benchmark in at least 55% of months.
- **Test, once:** the excess is above 0.
- **Only then:** a shadow forward test, like the composite's.

## Also reported (not part of the decision)

- The top 1 and top 5 industries instead of 3.
- A 6-1 month lookback instead of 12-1.
- The bottom 3 industries, to check that the ranking is monotonic.
- The share of months in which "Aerospace & Defense" and the utilities, banks and energy
  industries (where government-owned companies cluster) were among the top 3.

## Caveats

- Survivorship: today's listed stocks only.
- Yahoo's industry names are coarse for India.
- With about 60 development months, the test has limited power.
