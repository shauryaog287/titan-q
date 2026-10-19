# Pre-registration: FB1, stocks entering the F&O ban (written 2026-10-19, BEFORE any ban-list data was downloaded and BEFORE any return was computed)

## Question

NSE puts a stock "in ban" when open positions in its futures and options exceed 95% of the
market-wide position limit (MWPL). While it is in ban, traders may only reduce positions, not
add to them, until open interest falls below 80%.

A ban usually follows a burst of leveraged speculation. During the ban, new speculative demand
is locked out and existing positions have to unwind. This is a forced, price-insensitive
flow. Do stocks entering the ban underperform the market over the following two weeks? If so,
the system should avoid buying them. All 50 stocks in the live strategy trade in F&O, so this
would apply directly.

## Data (fixed now)

- **Ban lists:** NSE's daily ban-list file `fo_secban_DDMMYYYY.csv` from the nsearchives F&O
  archive, one per trading day from 2017-01. Downloaded by `fo_ban_research.py --fetch`.
  - The list for trade date t is published on the evening of t-1, so it is known before t
    opens.
- **Prices:** the Yahoo daily panels (`whole_market_research.load_panels`), and NSE symbols
  mapped through `universe.py`.

## Events (fixed now)

- **Ban entry:** a stock that is in the list on trading day t but was in none of the
  previous 10 trading days' lists. Repeat entries during one ban spell, or soon after it, do
  not count.
- **Measure:** the stock's return from the close of t, the first ban day, to the close of
  t+10, minus the equal-weight return of the liquid universe over the same days (as NJ1).
  - No cost, because this is an avoid flag, not a trade.
- **t-statistic:** averaged per calendar month of t, then computed over months.

## Periods

- **Development:** ban entries from 2017-01 to 2020-12.
- **Test (sealed):** 2021-01 to 2026-08. Revealed once, only if development qualifies.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - at least 150 ban entries;
  - the mean 10-day excess is -1.0% or lower;
  - the month t is -2.0 or lower.
- **Test passes** if the mean is below 0 and the month t is -1.0 or lower.
- **If it passes:** the live trading loop holds back buys of a stock while it is in the ban
  list, and for 10 trading days after it enters. The block uses the same no-buy mechanism as
  NJ1b; sells are never blocked.

## Also reported (not part of the decision)

- The first ban day's own move (close t-1 to close t), which is not tradeable.
- 5-day and 20-day horizons.
- **Ban exit:** the 10 days after the first trading day a stock is out of the list after a
  spell.
- Ban spells longer and shorter than 5 days.

## Caveats

- **Survivorship:** the universe is today's listed stocks, and stocks that left F&O
  (delisted or merged) may be missing prices.
- **Few, clustered events:** ban entries cluster in volatile months, and month clustering
  accounts for that.
