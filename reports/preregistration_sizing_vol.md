# Pre-registration: sizing volatility after a shock (written 2026-09-27, BEFORE any run)

## Why

In the v3 backtest (whitepaper 10.5) most of the 2020 shortfall came from quarter Kelly
dividing by the basket variance of a 252-day window. The window kept the March 2020
volatility for a year, so target exposure stayed near 24% from May 2020 to January 2021
while NIFTY recovered.

## Candidates (fixed now; no others will be tried in this round)

| Variant | `sizing_vol_halflife_days` | Meaning |
|---|---|---|
| live | 0 | v3: variance from the 252-day Ledoit-Wolf covariance |
| ewma_vol_21 | 21 | EWMA of the basket's daily returns, 1-month half-life |
| ewma_vol_63 | 63 | EWMA of the basket's daily returns, 3-month half-life |

Nothing else changes: same universe, weights, prior, quarter Kelly, shields and costs.
Only the volatility the Kelly sizer divides by is different. The EWMA reacts faster in
BOTH directions: it cuts harder during a crash and restores exposure sooner after it.

## Protocol

1. Development run, 2016-01-01 to 2020-12-31, for all three variants (`run_sizing_experiment.bat` runs these on the PC):

       python backtest_daily.py --start 2016-01-01 --end 2020-12-31 --variant live        --tag _dev_live
       python backtest_daily.py --start 2016-01-01 --end 2020-12-31 --variant ewma_vol_21 --tag _dev_ewma_vol_21
       python backtest_daily.py --start 2016-01-01 --end 2020-12-31 --variant ewma_vol_63 --tag _dev_ewma_vol_63

2. Decision rule on development results, written before they exist. A candidate
   qualifies only if, against live, it has
   - a higher Sharpe ratio by at least 0.05, AND
   - a CAGR that is not lower, AND
   - a maximum drawdown no worse than 1.25 times live's.
   If both qualify, take the higher Sharpe. If neither qualifies, v3 stays unchanged and
   the result is recorded.
3. Write the decision into `reports/decision_<date>.md` and commit it BEFORE step 4.
4. Confirmation, once: run the chosen variant and live over 2021-01-01 to the latest date
   (`--tag _test_<name>`). Adopt it (change the config default) only if its test Sharpe is
   not below live's. Otherwise record the failure and keep v3.

## Honest caveats

- The problem was diagnosed by looking at 2020, which lies inside the development window.
  So the development result is not independent of the idea. The confirmation step is the
  real test.
- The 2021-2026 period has already been seen under the v3 settings (the v3 backtest), but
  not under either candidate. Only these two candidates are tried, to limit the damage.
- The 2016 regime model is trained on about 14 months of data (history starts 2014-11),
  so it is weaker than later ones. It is the same for all three variants.
- Survivorship, NIFTY as a price index and cash at 0% apply, as in every report here.
