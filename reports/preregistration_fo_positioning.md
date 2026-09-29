# Pre-registration: FO1, who is positioned how in NIFTY futures (written 2026-09-29, BEFORE any data and BEFORE any result)

## Question

India's retail F&O traders lose money in aggregate (SEBI's studies: about 9 in 10
individuals lost over FY22-FY24). If their crowding is informative, NIFTY should do worse
after retail is unusually net long index futures, and better after they are unusually net
short. Foreign investors (FII) are the other side of most of those trades.

## Data

- NSE's daily participant-wise open interest (`fao_participant_oi_DDMMYYYY.csv`), 2016-01
  onward. Rows are Client (retail and other individuals), DII, FII and Pro. Columns include
  Future Index Long and Future Index Short, in contracts. Nothing has been downloaded yet.
- NIFTY daily closes (Yahoo ^NSEI).

## Signal (primary)

- **Client net position:** net = (Future Index Long - Future Index Short) / (Long + Short),
  for the "Client" row, at the close of day t.
- **Crowding:** the percentile of net on day t within the Client's own previous 250 trading
  days (at least 200 values).
- **Groups:**
  - "retail crowded long" = percentile of at least 0.80;
  - "retail crowded short" = percentile of at most 0.20;
  - "normal" = the rest.

## Measure

- **Sampling:** one observation per week, on the last trading day of each week, so forward
  windows do not overlap. The forward return is NIFTY's return from the close of t+1 to the
  close of t+6 (5 trading days). It starts at t+1 because the participant file for day t is
  published after that day's close.
- **Primary statistic:** the mean forward return of "normal" weeks minus that of "retail
  crowded long" weeks, with a Welch t-statistic.

## Periods

- **Development:** signal dates 2016-01 to 2020-12.
- **Test:** signal dates 2021-01 to 2026-08. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - the difference (normal minus crowded long) is at least +0.30% per week;
  - the Welch t is at least 2.0;
  - there are at least 40 crowded-long weeks.
- **Test, once:** the difference is above 0.
- **If both pass:** a shadow exposure overlay (for example, hold 50% instead of 100% of the
  planned exposure in retail-crowded-long weeks), forward-tested before it touches the live
  paper book.

## Also reported (not part of the decision)

- Crowded short minus normal (a contrarian bounce).
- The same test for the FII net index-futures position, where the expected sign is to follow
  FII.
- A put/call measure from the Client row's index options (put longs / call longs).

## Caveats

- One index and about 260 development weeks: the test has low power, so a real but small
  effect can easily fail it. That is accepted rather than tuned around.
- "Client" includes more than retail (for example, some HNI and corporate accounts).
