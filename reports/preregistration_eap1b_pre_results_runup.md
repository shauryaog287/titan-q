# Pre-registration: EAP1b, the pre-results run-up (written 2026-10-19, BEFORE any 2021-2026 return was computed for any EAP measure)

## Why a second registration

EAP1 (reports/earnings_premium.md) failed in development, so its test stays sealed. Its
development table showed a pattern it was not testing (2017-2020, 7,935 trades, gross excess
before costs):
- the pre-meeting part (entry to the close of the last trading day before the meeting):
  +0.84%, month t +2.24;
- the reaction part (to the close of the first trading day after it): -0.34%, t -2.62.

Because this was found in development data, it can only be **confirmed** on data not yet
seen. Meetings from 2021-01 to 2026-08 are unseen for every EAP measure.

## Trade (fixed now; identical to EAP1 except the exit)

- **Events:** NSE results board meetings whose date was announced in advance (`bm_timestamp`),
  as in EAP1, in liquid stocks (NJ1 definition at entry). A stock's meetings less than 20
  trading days apart count once.
- **Entry:** at the close of the later of these two days:
  - the trading day the meeting date became public (15:30 cut-off);
  - the 5th trading day before the meeting.
- **Exit:** at the close of the last trading day **before** the meeting date. A trade needs
  at least one trading day between entry and exit.
- **Result:** the stock's return minus the equal-weight liquid universe over the same days,
  minus 0.50% round-trip cost.
- **t-statistic:** the mean per calendar month of entry, then t over months.

## Confirmation (fixed now, run once)

- **Data:** meetings from 2021-01-01 to 2026-08-31.
- **Confirmed** if both hold: the mean net result is above 0, and the month t is at least
  2.0.
  - The t bar is strict because the pattern was found by looking: at t >= 2.0, a chance
    finding passes about 2.5% of the time.

## If confirmed

- First an observe-mode list on the dashboard: "results meeting in N days: pre-results
  window open (EAP1b)".
- A shadow paper strategy, like the composite forward test, before any trading on it. It gets
  its own short pre-registration, giving position size and the maximum number of names held
  at once.

## If not confirmed

The development pattern is treated as a 2017-2020 artefact and nothing changes.

## Also reported (not part of the decision)

- The gross result.
- The size thirds (small, mid, large).
- The date announced more than 7 days ahead, or less.
