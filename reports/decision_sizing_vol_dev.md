# Decision: sizing volatility, development period (2026-09-28)

Applied mechanically by sizing_experiment_decide.py to the development backtests (2016-01-04 to 2020-12-31), using the rule in reports/preregistration_sizing_vol.md. Written BEFORE any test-period run.

| Variant | CAGR | Sharpe | Max drawdown | Avg invested | Checks |
|---|---|---|---|---|---|
| live | 6.2% | 0.65 | 23.9% | 66.9% | baseline |
| ewma_vol_21 | 8.5% | 0.91 | 16.3% | 67.3% | sharpe >= live + 0.05: pass, cagr >= live: pass, max drawdown <= 1.25 x live: pass |
| ewma_vol_63 | 6.6% | 0.71 | 20.2% | 64.9% | sharpe >= live + 0.05: pass, cagr >= live: pass, max drawdown <= 1.25 x live: pass |

**Decision: ewma_vol_21.** Next (step 4, once): run ewma_vol_21 and live over 2021-01-01 to the latest date with --tag _test_<name>; adopt ewma_vol_21 only if its test Sharpe is not below live's.
