# Pre-registration: INS2, pledge invocations as a red flag (written 2026-09-28, BEFORE any count or result)

## Question

When lenders invoke shares that a company's promoters had pledged (sell or take them over
because a loan went bad), does the stock do worse than the market over the next three
months? If so, "pledge invoked" becomes a hold-nothing flag: Titan-Q would not buy the
stock, and would exit if it held it. This is a defensive filter, not a short-selling
strategy.

## Data

- The same NSE PIT disclosures as INS1 (`insider_research.py --fetch`, 2016-01 to about
  2026-04), already on disk. Before this was written, the only thing seen about invocations
  was one month's category count (2021-01: 16 rows of type "Pledge Invoke"). No returns were
  looked at.
- Prices: the Yahoo daily panel of the whole NSE universe.

## Events

A disclosure counts if all of these hold:
- the transaction type is "Pledge Invoke" OR the acquisition mode is "Invocation of pledge";
- the person category is "Promoters" or "Promoter Group";
- there is no minimum value, because the value field is often empty or not meaningful for
  invocations.

Dates and eligibility follow INS1:
- **Timing:** entry is at the close of the first trading day after NSE's broadcast date.
- **Merging:** disclosures for one stock with the same entry day merge into one event.
- **Eligibility:** the stock must pass `eligible_at` at entry (a stock Titan-Q could
  actually hold).

## Measure

- Excess return per event: the stock's return over 60 trading days minus the equal-weight
  return of eligible stocks over the same days.
- No trading cost is deducted, because the flag works by not holding the stock.
- Events are averaged by month of entry, and the t-statistic is computed over months.

## Periods

- **Development:** entries 2016-01 to 2020-12.
- **Test:** entries 2021-01 onward. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** as a red flag if all three hold:
  - the mean excess is -2.0% or worse;
  - the month-clustered t is -2.0 or lower;
  - there are at least 30 events.
- **Test, once:** the mean excess is below 0.
- **If both pass:** add the flag to Titan-Q's stock selection (no buys, and exit held
  positions, for 60 trading days after an invocation). Also list flagged stocks on the
  disclosure radar's watchlist.

## Also reported (not part of the decision)

- 20 and 120 trading-day horizons.
- Invocations by any person category.
- New pledges ("Pledge") and pledge releases ("Pledge Revoke") by promoters, as milder
  signals in each direction.
- The hit rate: the share of events with negative excess.

## Caveats

- **Survivorship:** the panel holds today's listed stocks. The worst cases, where an
  invocation was followed by collapse and delisting, are missing, so any measured damage is
  an underestimate.
- **Eligibility filter:** distressed stocks often fall below the liquidity bar before the
  invocation, which removes some events.
