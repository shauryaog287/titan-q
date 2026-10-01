# Pre-registration: NJ1b, no-news jumps fade (written 2026-10-01, BEFORE the 2021-2026 data is looked at for this measure)

## Why a second registration

NJ1's primary hypothesis (news-backed jumps keep rising) failed in development, so its test
period stayed sealed. NJ1's development table also showed something it was not testing: jumps
with NO news fell back strongly (2017-2020: 4,258 events, 5-day excess -0.50% before costs,
date-clustered t -6.37). Because that came from looking at development data, it can only be
CONFIRMED on data not yet seen. The 2021-2026 period is still unseen for every NJ1 measure.

## Definition (identical to NJ1, reports/preregistration_news_jumps.md)

- **A jump:** at least +5% close to close, on at least 3 times the 20-day average volume.
- **Liquidity:** the stock trades at least Rs. 1 crore a day (median over 60 days), its price
  is at least Rs. 20, and it has at least 250 days of history.
- **No news:** no non-routine NSE announcement between 15:30 on the previous trading day and
  15:30 on the jump day.
- **Merging:** a stock's events less than 10 days apart count once.
- **Measure:** the excess return from the close of the jump day to the close 5 trading days
  later, against the equal-weight liquid universe. No cost, because it is used as an avoid
  flag, not a trade. Events are averaged by date, and t is computed over dates.

## Confirmation test (fixed now, run once)

- **Data:** event dates from 2021-01 to 2026-08.
- **Confirmed** if the mean 5-day excess of no-news jumps is below 0 AND the date-clustered t
  is -2.0 or lower.

## If confirmed

- The mover scanner tags no-news jumps "no news - historically tends to fade (NJ1b)".
- No strategy may buy a stock on the day of a no-news jump, or for the next 5 trading days.
- Shorting is not considered: the paper system is long-only in the cash market.

## If not confirmed

The development pattern is treated as a 2017-2020 artefact and nothing changes.
