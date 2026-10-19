# Pre-registration: EAP2, the small-stock pre-results run-up, FORWARD test (written 2026-10-19, before any of the data it will be judged on exists)

## Why

In EAP1b (reports/pre_results_runup.md, unseen 2021-2026 data) the pre-results run-up was
real before costs (+0.63%, t 2.67) but not after them (+0.13%, t 0.55). One side table showed
the smallest third of stocks keeping a net gain: +0.36% per trade, month t +2.77, 7,578
trades.

That side table was not the pre-registered test, and no unseen history is left to confirm
it. The only honest confirmation is on data that does not exist yet: results meetings from
2026-10-20 onward.

## Trade (fixed now; EAP1b's rule, small third only)

- **Events:** NSE results board meetings whose date was announced in advance (`bm_timestamp`,
  `earnings_premium.results_meetings`). The meeting date must be 2026-10-20 or later.
- **Entry:** at the close of the later of these two days:
  - the trading day the meeting date became public (15:30 cut-off);
  - the 5th trading day before the meeting.
- **Exit:** at the close of the last trading day before the meeting. A trade needs at least
  one trading day between entry and exit.
- **Stocks:**
  - liquid at entry (NJ1 definition);
  - in the **smallest third** of the liquid universe by 60-day median traded value at entry
    (`event_tools.Panels.size_third`).
- A stock's meetings less than 20 trading days apart count once.
- **Result:** the stock's return minus the equal-weight liquid universe over the same days,
  minus 0.50% round-trip cost.

## Evaluation (fixed now)

- **When:** at the first evaluation date, 2027-04-30, if there are at least 300 completed
  trades by then. Otherwise at the first month end with 300 or more, and no later than
  2027-10-31.
  - Until then, `runup_watch.py` reports the running numbers for information only. No
    decision is taken early, whatever they show.
- **Confirmed** if both hold:
  - the mean net result is above 0;
  - the t is at least 2.0, **clustered by ISO week of entry**. Six months hold only about 6
    calendar months but about 26 weeks, so week clustering keeps the test meaningful while
    still grouping trades that share a market week.
- **If confirmed:** a shadow paper strategy (like the composite forward test) with its own
  short pre-registration giving position size and the maximum number of names held at once.
  No live trading before that.
- **If not confirmed:** the small-stock pattern is treated as a 2021-2026 artefact.

## Logged daily (information only)

- The stocks whose pre-results window is open (a results meeting within the next 5 trading
  days, the date already public, in the small third), with entry date and planned exit, in
  `data/board_meetings/open_windows_<date>.json`.
- The running forward results, for all three size thirds, in reports/runup_watch.md.
