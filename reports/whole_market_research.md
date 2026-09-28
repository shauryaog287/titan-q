# Titan-Q whole-market signal research

Implements reports/preregistration_whole_market.md. Point-in-time liquid NSE universe (median 60-day traded value >= Rs. 10 crore, price >= Rs. 20, >= 2 years of history), monthly rebalancing, one-day lag, 25 bps per unit traded. Hit rate (vs median) = share of picked stocks that beat the universe median next month.

## Development (formation dates 2015-09 to 2020-12)

| Signal | Mean IC | IC t | Decile excess/yr | t | Decile hit (vs median) | Decile hit (>0 net) | Months decile beat EW | Top-30 excess/yr | Qualifies |
|---|---|---|---|---|---|---|---|---|---|
| MOM_12_1 | +0.031 | +1.07 | +2.8% | +0.45 | 47.7% | 51.3% | 49% | +4.6% | no |
| MOM_6_1 | +0.030 | +1.18 | +3.3% | +0.63 | 50.4% | 53.3% | 51% | +3.9% | no |
| REV_1M | +0.025 | +1.25 | -6.3% | -0.91 | 49.4% | 51.6% | 37% | -3.3% | no |
| LOWVOL | +0.052 | +1.56 | +3.6% | +0.55 | 54.1% | 54.6% | 53% | +3.1% | no |
| LOWBETA | +0.029 | +0.81 | -0.9% | -0.13 | 51.5% | 54.1% | 51% | +0.6% | no |
| HI52 | +0.036 | +1.09 | -1.4% | -0.25 | 49.9% | 53.2% | 49% | +0.8% | no |
| TREND200 | +0.025 | +0.89 | +3.1% | +0.52 | 50.1% | 52.1% | 57% | +6.1% | no |

Equal-weight universe: 11.3% a year, Sharpe 0.47, 51 months, about 203 eligible stocks per month.

**Pre-registered decision on development: candidate = none.**

