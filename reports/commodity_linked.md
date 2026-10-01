# CM1: overnight crude and gold moves and commodity-linked stocks

Implements reports/preregistration_commodity_linked.md. Yahoo daily, auto_adjust=False. A commodity date D maps to the first NSE day N after it. Trade on N if the commodity moved >= 1.5% on D: each group stock from N's open to N's close in the expected direction, minus NIFTY's open-to-close in the same direction, minus 0.10% cost.

## Development (days N from 2016-01 to 2020-12)

| Group and trigger | Trades | Days | Mean net per trade | Day-clustered t | Hit rate |
|---|---|---|---|---|---|
| **Primary: all three groups, 1.5% trigger** | 5,571 | 546 | +0.008% | +0.68 | 49.6% |
| All three groups, 1.0% trigger | 7,623 | 748 | -0.021% | +0.77 | 49.0% |
| crude producers, 1.5% | 970 | 485 | +0.102% | +1.19 | 52.1% |
| crude producers, 1.0% | 1,298 | 649 | +0.034% | +0.50 | 50.2% |
| crude users, 1.5% | 4,365 | 485 | -0.032% | -0.72 | 49.0% |
| crude users, 1.0% | 5,841 | 649 | -0.062% | -1.67 | 48.4% |
| gold lenders, 1.5% | 236 | 118 | +0.367% | +1.33 | 51.3% |
| gold lenders, 1.0% | 484 | 242 | +0.338% | +2.06 | 53.1% |
| Jewellers vs gold (exploratory, long on gold up), 1.5% | 118 | 118 | -0.197% | -0.94 | 44.1% |

## Gap capture (development, every day, not only triggered ones)

Slope of the stock's close-to-open gap on N against exposure x commodity move (pooled OLS).

| Group | Stock-days | Slope | Std. error |
|---|---|---|---|
| crude producers | 2,390 | +0.135 | 0.008 |
| crude users | 10,755 | -0.011 | 0.004 |
| gold lenders | 2,388 | +0.133 | 0.027 |
| jewellers (exploratory) | 1,194 | +0.087 | 0.032 |

Rule: mean net >= +0.10% per trade, day-clustered t >= 2.0, >= 100 trades.

Note: when an NSE holiday falls on a commodity trading day, the commodity's returns on the dates that map to the same NSE day are compounded into one move (the pre-registration did not cover this; decided before any result).

**Decision: does not qualify; the test stays sealed.**

