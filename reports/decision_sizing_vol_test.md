# Decision: sizing volatility, test-period confirmation (2026-09-28)

Step 4 of `reports/preregistration_sizing_vol.md`, run once, after the development decision
(`reports/decision_sizing_vol_dev.md`, commit `a854631`) chose ewma_vol_21.

| 2021-01-01 to 2026-09-25 | CAGR | Volatility | Sharpe | Max drawdown | Avg invested | Trades | Fees |
|---|---|---|---|---|---|---|---|
| live (v3 sizing: 252-day window) | 8.8% | 10.6% | 0.85 | 16.0% | 76.8% | 1,238 | Rs. 7,955 |
| ewma_vol_21 | 8.9% | 9.9% | 0.91 | 15.2% | 75.5% | 2,140 | Rs. 10,248 |
| NIFTY buy-and-hold | 9.3% | 14.1% | 0.70 | 17.2% | 100% | - | - |
| Equal-weight universe (survivorship-biased) | 16.8% | 14.7% | 1.13 | 19.3% | 100% | - | - |

**Rule:** adopt only if the test Sharpe is not below live's. 0.91 >= 0.85, so **ewma_vol_21 is
adopted**: `sizing_vol_halflife_days = 21` is now the default in `titanq_config.py`.

## Honest reading

- The test-period gain is small: +0.06 Sharpe, 0.8 points less drawdown, CAGR about equal.
  A difference of this size over under six years is within noise. What the test shows is
  that the change does no harm out of sample; the large gain in development (Sharpe 0.65 ->
  0.91) came from 2020, the episode that motivated the idea.
- Faster sizing trades more: 2,140 trades against 1,238, and about Rs. 2,300 more in fees
  over the period. That is already included in the returns above.
- Both versions were fully in cash on 102 days in 2021-2026. The regime model refit for
  2021 labelled the post-COVID rally CRASH (whitepaper 10.5). Whether the CRASH limit and
  the other catastrophe limits earn their place is the next question for the same protocol.
- Both trail the equal-weight universe by about 8 points a year. Most of that gap is the
  lower average exposure (Kelly sizing and the limits) and the survivorship bias in the
  benchmark, not stock selection.
