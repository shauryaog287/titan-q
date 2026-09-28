# Pre-registration, round 2: seven-signal composite (written 2026-09-28, BEFORE any composite run)

## Round 1 result (development only, `reports/whole_market_research.md`)

No single signal qualified: every IC t-statistic was below 1.6. But all seven had a
POSITIVE mean rank IC (+0.025 to +0.052) over about 200 liquid stocks a month
(2016-09 to 2020-12, 51 months). Seven weak signals all pointing the same way suggests a
small common effect that each alone lacks the power to show. The test period (2021-2026)
has NOT been revealed for the whole market.

## Hypothesis (formed after seeing round 1 development results, so the test is what counts)

COMPOSITE7 = each month, the equal-weighted mean of the cross-sectional z-scores of all seven
signals (MOM_12_1, MOM_6_1, REV_1M, LOWVOL, LOWBETA, HI52, TREND200), in the same
point-in-time universe, with the same timing and costs (25 bps per unit traded).

## Rule (fixed now)

1. Development (2015-09 to 2020-12 formation dates): COMPOSITE7 must meet the SAME bar as a
   single signal in round 1:
   - IC t-statistic >= 2.7;
   - top-decile excess over the equal-weight universe > 0 with t >= 2.0, after costs;
   - relative hit rate > 50%.

   If it fails, stop: nothing is adopted and the test period stays unrevealed.
2. If it passes, commit that result, then reveal the test ONCE (2021-01 to 2026-08). Confirm
   only if the test IC is > 0 and the top-decile excess after costs is > 0.
3. Even if confirmed, it becomes the live selection rule only after a full walk-forward
   backtest shows it beating v3.1.

## Honest caveats

- The composite was chosen after looking at round 1's development ICs. Its development
  result is therefore not independent evidence; only the test is.
- Equal weights are fixed. No weight optimisation is allowed, since that would overfit
  51 months.
