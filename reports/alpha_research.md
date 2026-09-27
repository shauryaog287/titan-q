# Titan-Q alpha research

Development formation dates 2015-09-01..2020-12-31; test 2021-01-01..2026-08-31. Monthly rebalancing, one-day implementation lag, costs 16.1 bps per unit of weight traded. Excess returns are against the equal-weight universe (which shares its survivorship bias).

## Development (2015-2020)

| Signal | Mean rank IC | IC t | IC>0 | Top-4 excess/yr | t | Top-8 excess/yr | t |
|---|---|---|---|---|---|---|---|
| MOM_12_1 | +0.018 | +0.52 | 44% | -3.7% | -0.53 | -0.5% | -0.12 |
| MOM_6_1 | +0.016 | +0.55 | 54% | +3.3% | +0.48 | +2.3% | +0.45 |
| REV_1M | +0.030 | +1.33 | 59% | +1.8% | +0.24 | +0.5% | +0.11 |
| LOWVOL | -0.033 | -1.12 | 51% | +1.6% | +0.26 | -4.5% | -1.00 |
| LOWBETA | -0.031 | -0.28 | 50% | +3.7% | +0.59 | +0.2% | +0.04 |
| HI52 | +0.004 | +0.13 | 52% | -9.5% | -1.44 | -6.4% | -1.32 |
| TREND200 | +0.024 | +0.75 | 57% | -2.8% | -0.44 | +1.8% | +0.40 |

Equal-weight universe: 18.6% a year, Sharpe 0.96 (63 months).

## Test (2021-2026)

| Signal | Mean rank IC | IC t | IC>0 | Top-4 excess/yr | t | Top-8 excess/yr | t |
|---|---|---|---|---|---|---|---|
| MOM_12_1 | +0.013 | +0.43 | 52% | -8.1% | -1.45 | -5.0% | -1.09 |
| MOM_6_1 | -0.009 | -0.33 | 51% | -12.7% | -2.22 | -4.3% | -1.08 |
| REV_1M | +0.040 | +1.62 | 57% | +5.0% | +1.00 | +5.6% | +1.35 |
| LOWVOL | -0.036 | -1.29 | 40% | -9.5% | -1.87 | -7.6% | -1.99 |
| LOWBETA | -0.002 | -0.06 | 48% | -12.3% | -2.77 | -6.0% | -1.79 |
| HI52 | -0.051 | -1.81 | 40% | -15.4% | -2.68 | -10.1% | -2.99 |
| TREND200 | -0.027 | -0.97 | 43% | -1.6% | -0.31 | -5.6% | -1.30 |

Equal-weight universe: 16.6% a year, Sharpe 1.22 (67 months).

## Exposure overlays on NIFTY (daily, after costs)

| Rule | Dev CAGR | Dev Sharpe | Dev max DD | Test CAGR | Test Sharpe | Test max DD | Avg exposure (test) |
|---|---|---|---|---|---|---|---|
| buy_and_hold | 11.4% | 0.69 | 38.4% | 10.3% | 0.76 | 17.2% | 100% |
| trend_200d | 5.8% | 0.61 | 18.3% | 4.0% | 0.40 | 22.4% | 77% |
| vol_target_15 | 7.6% | 0.60 | 28.8% | 9.0% | 0.73 | 15.8% | 94% |
| trend_x_vol_target | 5.0% | 0.56 | 18.3% | 3.4% | 0.37 | 20.6% | 74% |

## Weighting the whole universe (monthly, fully invested, after costs)

| Weights | Dev CAGR | Dev Sharpe | Dev max DD | Test CAGR | Test Sharpe | Test max DD |
|---|---|---|---|---|---|---|
| equal_weight | 17.9% | 0.95 | 31.1% | 16.9% | 1.22 | 15.7% |
| inverse_vol | 17.3% | 0.96 | 29.2% | 16.3% | 1.21 | 15.4% |
| hrp | 13.4% | 0.78 | 22.2% | 12.0% | 0.87 | 18.9% |
