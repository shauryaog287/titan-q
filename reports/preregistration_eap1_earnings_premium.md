# Pre-registration: EAP1, the earnings announcement premium (written 2026-10-19, BEFORE any board-meeting data was downloaded and BEFORE any return was computed)

## Question

Stocks tend to earn more than usual in the few days before and around their scheduled
results. This is the "earnings announcement premium", documented in the US and other
markets. A common explanation is that investors crowd into a stock as its scheduled results
draw attention.

In India the results date is public in advance: SEBI's listing rules make a company announce
its results board meeting days ahead. So a buyer can know the date before entering. Does
buying a few days before the meeting and selling the day after it beat the market, net of
costs?

## Data (fixed now)

- **Board meetings:** NSE's corporate-board-meetings feed (`/api/corporate-board-meetings`).
  - Meetings whose purpose (`bm_purpose`) or description (`bm_desc`) mentions results
    ("result", any case).
  - Fields used: the meeting date `bm_date`, and `bm_timestamp`, the time the meeting was
    announced (when the market learned the date).
  - Downloaded for meeting dates 2017-01 onward by `earnings_premium.py --fetch`.
  - If a stock announced the same meeting date more than once, the earliest announcement
    counts.
- **Prices:** the Yahoo daily panels (`whole_market_research.load_panels`), and NSE symbols
  mapped to Yahoo tickers through `universe.py`.

## Trade (fixed now)

- **Known day:** the trading day of the announcement time if it was before 15:30, otherwise
  the next trading day.
- **Entry:** at the close of the later of these two days:
  - the known day;
  - the 5th trading day before the meeting date.
- **Exit:** at the close of the first trading day after the meeting date, so the reaction to
  the results is inside the hold whether they came out during market hours or after the
  close.
- The entry must be at least one trading day before the meeting date; otherwise there is no
  trade.
- **Liquidity at entry** (as NJ1):
  - median traded value of at least Rs. 1 crore over the previous 60 days;
  - previous close of at least Rs. 20;
  - at least 250 days of history.
- **Result per trade:** the stock's return minus the equal-weight return of the liquid
  universe over the same days, minus 0.50% round-trip cost.
- A stock's meetings less than 20 trading days apart count once (the first).

## Measure

- The mean net result per trade.
- A month-clustered t: the mean per calendar month of entry, then t over months.
- The share of trades with a positive net result.

## Periods

- **Development:** meeting dates from 2017-01 to 2020-12.
- **Test (sealed):** 2021-01 to 2026-08. Revealed once, only if development qualifies.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - at least 500 trades;
  - the mean net result is at least +0.30% per trade;
  - the month t is at least 2.0.
- **Test passes** if the mean net result is above 0 and the month t is at least 1.0.
- **If it passes:** first an observe-mode list of upcoming results meetings ("EAP window
  open") on the dashboard. A paper strategy built on it gets its own pre-registration.

## Also reported (not part of the decision)

- The gross result (no cost).
- The pre-meeting part only (entry to the close of the last trading day before the meeting)
  and the announcement part only.
- Size thirds by 60-day median traded value (small, mid, large).
- Lead time: the date announced more than 7 calendar days ahead, or less.

## Caveats

- **Survivorship:** the universe is today's listed stocks.
- **Results timing:** results are not always released on the meeting date (late in the
  evening or the next day). The exit on the first trading day after the meeting covers most
  of this.
- **Costs:** the 0.50% assumption covers delivery STT, stamp duty, exchange charges and
  slippage for liquid stocks. For small caps it may be optimistic.
