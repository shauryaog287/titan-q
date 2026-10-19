# Pre-registration: IPO1, anchor lock-in expiry (written 2026-10-19, BEFORE the IPO list was downloaded and BEFORE any return was computed)

## Question

Anchor investors in an Indian IPO get their shares a day before the issue opens to the
public, and SEBI's rules lock them in for 30 days from allotment. Since 2022, only half
unlocks at 30 days and the other half at 90. The unlock date is known from the day of
listing. Anchors are often funds that took an allocation to flip.

When the lock-in ends, a predictable block of supply becomes free to sell. Do newly listed
stocks underperform the market around their anchor lock-in expiry? If so, the system should
not buy them then. This is a forced, scheduled flow that the public rarely looks at.

## Data (fixed now)

- **IPOs:** NSE's past-issues feed (`/api/public-past-issues`), downloaded per calendar year
  by `ipo_lockin_research.py --fetch`.
  - Mainboard issues only (securityType "EQ", not SME), with a listing date.
- **Prices:** the Yahoo daily panels (`whole_market_research.load_panels`), and NSE symbols
  mapped through `universe.py`. IPOs whose stock has no price series are skipped.

## Events (fixed now)

- **Anchor 30-day unlock date U:** the listing date plus 29 calendar days.
  - Allotment is about one day before listing, and the lock-in runs 30 days from allotment.
  - If U is not a trading day, the next trading day is used.
- **Window:** from the close of the 3rd trading day before U to the close of the 10th trading
  day after U. Starting early covers a day or two of error in U and selling ahead of the
  date.
- **Liquidity:** the median daily traded value from listing up to the entry day must be at
  least Rs. 1 crore, over at least 10 trading days, and the entry close at least Rs. 20. The
  250-day history rule is dropped, because these are new listings.
- **Measure:** the stock's return over the window minus the equal-weight return of the liquid
  universe (NJ1 definition) over the same days.
  - No cost, because this is an avoid flag.
- **t-statistic:** the mean per calendar month of U, then t over months.

## Periods

- **Development:** unlock dates from 2017-01 to 2021-12. About 90 mainboard IPOs listed in
  2017-2020, too few on their own, so 2021 is included.
- **Test (sealed):** 2022-01 to 2026-08. Revealed once, only if development qualifies.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - at least 100 events;
  - the mean excess is -1.0% or lower;
  - the month t is -2.0 or lower.
- **Test passes** if the mean is below 0 and the month t is -1.0 or lower.
- **If it passes:** no buys of a stock from 3 trading days before an anchor unlock to 10
  after it (the NJ1b no-buy mechanism), plus a tag in mover alerts. The 50 large caps rarely
  include recent IPOs, so this mainly guards future whole-market strategies and the alerts.

## Also reported (not part of the decision)

- **Anchor 90-day unlock:** issues closing from 2022-04, at listing plus 89 days, same window.
- **Pre-IPO shareholder unlock:** listing plus 6 months minus 1 day for issues closing from
  2021-08-13, otherwise listing plus 1 year minus 1 day; same window.
- The run-up part (close of day U-3 to the close of the day before U) against the after part.
- **Stocks above or below their issue price** at entry. Holders in profit may be quicker to
  sell.

## Caveats

- **U is approximate:** the listing date is used for allotment plus one day.
- **Few, clustered IPOs:** IPOs cluster in hot markets, so events cluster in time. Month
  clustering accounts for part of that.
- **Survivorship:** issues whose stock is no longer listed are missing.
