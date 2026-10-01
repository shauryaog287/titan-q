# Forward test: seven-signal composite, top 30 (pre-registered 2026-09-28)

## Why

In development (2016-2020) the equal-weight z-score composite of seven signals looked
promising: rank IC +0.060, top-decile excess +8.5% a year, 55.0% of picks profitable after
costs, and it beat the universe in 61% of months. But 51 months cannot rule out luck (t = 1.34),
so it failed the pre-registered bar (`reports/whole_market_research_round2.md`). Forward paper
trading is the evidence that cannot be overfitted: every month recorded here happens after
this rule was written.

## Strategy (fixed)

- **Universe (point in time):** NSE equities with a median 60-day traded value of at least
  Rs. 10 crore, a price of at least Rs. 20, at least 2 years of history, and a complete
  253-day window (`whole_market_research.eligible_at`).
- **Signal:** COMPOSITE7, the equal-weighted mean of the cross-sectional z-scores of
  MOM_12_1, MOM_6_1, REV_1M, LOWVOL, LOWBETA, HI52 and TREND200.
- **Book:** the top 30 by COMPOSITE7, equal-weighted and fully invested.
- **Timing:**
  - signals are computed from closes up to the last trading day of each month;
  - trades execute at the close of the first trading day of the next month;
  - on the first run, the book is built at that day's close from the previous day's data.
- **Costs:** the paper broker's equity-delivery cost stack plus 5 bps slippage each way.
- **Account:** a separate paper account (`paper_account_shadow.json`, Rs. 5,00,000), which
  never interacts with the live account.

## What is recorded, daily after the close (`forward_test.py`)

`composite_top30.csv` records, for each day:
- the shadow account's equity, cash, number of holdings and fees;
- NIFTY's close;
- the equal-weight point-in-time universe index (rebalanced monthly, no costs; a generous
  benchmark);
- the live v3.1 paper account's equity, when that account exists.

The file is committed to git regularly, so each day's result carries a timestamp.

## Evaluation (fixed now)

- **Checkpoints at 6 and 12 months** report excess return over the equal-weight universe,
  the monthly win rate, and the stock hit rates (picks beating the universe median; picks
  profitable after costs).
- **Adoption,** as the live selection rule after 12 months, only if ALL hold:
  1. forward excess return over the equal-weight universe is > 0 after costs;
  2. the monthly win rate is > 50%;
  3. development and forward months pooled give an excess-return t-statistic >= 2.0.
- If it fails, the forward record stays as evidence and nothing is adopted.

## Caveats

- One year is still short. The adoption rule pools development and forward data for
  exactly this reason.
- The universe list is today's listed stocks. Going forward this no longer matters:
  delistings that happen from now on will show up in the data.

## Corrections

- **2026-09-29:** the row recorded at 15:48 was removed, and the data was not retouched in any
  other way.
  - The network had dropped minutes earlier, so the price refresh failed (0 of 2,703 symbols).
    Only the ~50 stocks that the live loop had fetched that morning had a row for the day.
  - The holdings had no price, so the equity shown was their cost basis, and the equal-weight
    benchmark stayed at 100.00. No trades were made that day.
  - `forward_test.py` now refuses to record unless 90% of the stocks that closed the day
    before also have the day's close.
  - The day is re-recorded if the data arrives before midnight. Otherwise the record simply
    skips it.
- **2026-10-01:** two bookkeeping errors were corrected in the evening. The composite's own
  equity and trades were right and were not changed.
  - The equal-weight benchmark. On a rebalance day, `forward_test.py` re-based the benchmark
    at that day's close BEFORE applying the day's return, so the first trading day of each
    month was skipped. The 1 Oct level (the day the market fell) became 99.5846, unchanged.
    Recomputed with the old basket's 1 Oct closes, it is 98.3300. The state now re-bases from
    98.33, and the code applies the day's return before re-basing.
  - The live account column. It valued the live v3 account only at stocks with a Yahoo close
    that day, and UPL.NS (14 shares) had none on 1 Oct. That made the account look Rs. 7,557
    poorer: 489,579.09 instead of 497,135.59. The code now uses the last available close.
