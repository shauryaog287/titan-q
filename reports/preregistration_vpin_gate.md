# Pre-registration: does the VPIN buy gate help? (written 2026-09-28, BEFORE any VPIN result on history)

## Question

Since 2026-09-28 the live system computes VPIN toxicity but does not act on it
(`vpin_gate="observe"`). In `"block"` mode, a buy is held back while the stock's toxicity
score is above 0.90. This test asks whether holding buys back that way gets better prices
than buying at once, net of the market's move over the delay. The gate stays in observe mode
unless the rule below is met.

## Data

- Angel One 1-minute candles (`angel_history.py`) for the 50 stocks in `titanq_config.UNIVERSE`
  (the stocks the live system trades) that have history, plus NIFTY (`NIFTY50_IDX`).
- Daily volumes from the Yahoo daily cache (as the live system uses).

## The toxicity score, exactly as live

For each stock and session D:

- bars: the 1-minute bars of the 4 sessions before D plus D's bars so far (the live feed
  holds 5 sessions), zero-volume bars dropped;
- bucket volume: the median daily volume of the 20 sessions before D, divided by 50;
- VPIN: `VPINEstimator(n_buckets=50, classifier="bvc", sigma_mode="rolling")`;
- score: `ToxicityScore(combine="geometric", weight_hawkes=0)` fitted on the VPIN of buckets
  that closed before D began; the score at time t is that of the last bucket closed by t.

This is `titanq_brain.ToxicityMonitor.score`. The research code computes the VPIN once per
stock-day and reads it at each minute. That is valid because every step (bucketing, rolling
sigma, VPIN window) uses only earlier data; the code checks this against the live function
on a sample of stock-days and stops if they differ.

## Events

- Decision times each session: 09:20, 09:30, 10:30, 11:30, 12:30, 13:30 and 14:30 IST.
- A decision uses the bars completed before it, and "the price at t" is the close of the last
  completed 1-minute bar (the paper broker fills at the latest price).
- An event is a (stock, session, decision time) where the score is above 0.90 (flagged).
  Unflagged decisions are unaffected by the gate.

## What the gate does to a flagged buy

- It buys at the first later minute t1 where the score is 0.90 or less (or not computable),
  no later than 15:19 (the live loop makes no buys in the last 10 minutes).
- If no such minute comes, it buys at 09:16 in the next session (the close of the 09:15 bar).

## Measure

For each flagged event, the improvement in bps:

    improvement = [ln P(t0) - ln P(t1)] - [ln NIFTY(t0) - ln NIFTY(t1)]

A positive value means the delayed buy was cheaper relative to the market. Each session's
events are averaged, and the day-level t-statistic is computed over sessions, because the 50
stocks' events on one day are correlated.

## Periods

- **Development:** sessions 2021-01 to 2023-12.
- **Test:** sessions 2024-01 to 2026-09; blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if the mean improvement is at least +3 bps per flagged event AND
  the day-level t is at least 2.0.
- **Test, once:** the mean improvement is above 0.
- **Only then:** switch `vpin_gate` to `"block"`. Otherwise it stays in `"observe"`.
- The threshold (0.90), bucket count, window and decision times are fixed. None of them
  will be tuned on this data.

## Also reported (not part of the decision)

- the share of decisions that are flagged;
- the median delay, and the share pushed to the next session;
- the raw improvement, not adjusted for the market;
- the mean absolute 30-minute forward return after flagged and after unflagged decisions.
  (VPIN is known mainly as a volatility warning rather than a direction signal.)

## Caveats

- Angel's 1-minute volumes stand in for Yahoo's, which the live feed uses. Both are
  exchange volumes, but they can differ at the edges of the session.
- Delays can reach the next session, so the measure includes overnight moves. That is
  correct for the gate's real effect, but it is noisy.
- TMPV.NS (listed 2025) contributes only to the test period.
