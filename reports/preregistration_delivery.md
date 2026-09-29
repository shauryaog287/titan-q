# Pre-registration: DEL1, delivery percentage (written 2026-09-29, BEFORE the data is downloaded and BEFORE any result)

## Data

- **Delivery data:** NSE's daily "Security-wise Delivery Position" files (MTO_DDMMYYYY.DAT):
  traded quantity, deliverable quantity and delivery % per stock, 2016-01 onward.
  - Checked on 2026-09-29 for 2016, 2017 and 2026 dates: header and format only.
  - No delivery value has been looked at.
- **Prices and volumes:** the Yahoo daily panel of the whole NSE universe (the same panels as
  `whole_market_research.py`).
- **Eligibility:** `eligible_at` (median 60-day traded value at least Rs. 10 crore, minimum
  price and history), point in time.

Delivery % is the share of the day's traded shares that were actually delivered, meaning
taken home rather than squared off the same day. A low value means mostly intraday
speculation; a high value means buying (or selling) with conviction.

## DEL1a: abnormal delivery, monthly cross-section

- **Signal:** at each month end (row a, data up to and including a), the stock's mean delivery
  % over the last 20 trading days minus its mean over the 230 trading days before those
  (a-250 to a-21). At least 150 of the 250 days must have a value.
- **Measure:** exactly the machinery and thresholds of the whole-market single-signal study
  (`whole_market_research.monthly_study` and `summarise`):
  - the monthly Spearman IC against the next month's return;
  - the top-decile and top-30 equal-weight portfolios against the equal-weight eligible
    universe, net of 25 bps per unit of turnover.
- **Qualifies in development** if all four hold:
  - IC t is at least 2.7;
  - the decile excess is above 0;
  - the decile excess t is at least 2.0;
  - the decile hit rate against the median is above 50%.
- Also reported: the raw 20-day delivery % level as a second signal, which is not part of
  the decision.

## DEL1b: retail-frenzy reversal, daily events

An event is a (stock, day t) where all of these hold:
- the close-to-close return on t is at least +5%;
- volume on t is at least 3 times the stock's 20-day average volume (days before t);
- delivery % on t is below the 25th percentile of the stock's own previous 250 days;
- the stock passes `eligible_at` on t.

The rule is applied as follows:
- **Merging:** events for the same stock less than 10 trading days apart count once (the
  first).
- **Measure:** the excess return from the close of t+1 to the close of t+11 (10 trading
  days), against the equal-weight eligible universe over the same days. Entry is at t+1 because
  the day's delivery figure is published after the close of t.
- **Clustering:** events are averaged by event date, and t is computed over dates.
- **Qualifies in development**, as an avoid-filter (do not buy after a low-delivery frenzy),
  if all three hold:
  - the mean excess is -1.0% or worse;
  - the date-clustered t is -2.5 or lower;
  - there are at least 100 events.

  The t threshold is stricter than 2.0 because two hypotheses are tested here.
- Also reported:
  - 5 and 20-day horizons;
  - the same jump with HIGH delivery (above the 75th percentile), to see whether conviction
    buying continues;
  - big down days (-5% or worse) on low delivery (panic selling), which may bounce.

## Periods

- **Development:** 2016-01 to 2020-12. For DEL1a these are month ends; for DEL1b, event dates.
- **Test:** 2021-01 to 2026-08. Blind until the development decisions are committed.
- **Test, once:**
  - DEL1a: the decile excess is above 0 and the IC is above 0.
  - DEL1b: the mean excess is below 0.

## If a part passes

- **DEL1a:** a shadow forward test, like the composite's.
- **DEL1b:** a stock-level avoid-flag in the radar, still in observe mode until a forward check
  agrees.

## Caveats

- The price panel holds today's listed stocks (survivorship).
- MTO files describe the rolling-settlement series (EQ). Stocks in the trade-for-trade (BE)
  series have 100% delivery by rule; they are excluded.
- A public holiday has no file; missing days are skipped, not filled.
