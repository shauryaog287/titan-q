# FO1: retail and FII positioning in NIFTY index futures

Implements reports/preregistration_fo_positioning.md.

## Development 2016-2020 (weekly, NIFTY return close t+1 to t+6)

| Group (vs normal weeks) | Crowded weeks | NIFTY next week, crowded | Normal | Normal minus crowded | Reading |
|---|---|---|---|---|---|
| **Primary: retail net long (Client >= 80th pct)** | 64 of 216 | -0.35% | +0.72% | +1.06% (t +2.18) | positive = fade retail |
| retail net short (Client <= 20th pct) | 38 of 216 | +0.01% | +0.72% | +0.70% (t +1.66) | negative = fade retail |
| FII net long (>= 80th pct) | 40 of 216 | +0.42% | +0.56% | +0.14% (t +0.32) | negative = follow FII |
| FII net short (<= 20th pct) | 67 of 216 | -0.27% | +0.56% | +0.83% (t +1.72) | positive = follow FII |
| retail put/call high (>= 80th pct) | 50 of 216 | +0.56% | +0.20% | -0.36% (t -1.01) | negative = fade fear |

Rule: normal minus retail-crowded-long >= +0.30%/week, Welch t >= 2.0, >= 40 crowded weeks.

**Decision: qualifies; reveal the test once.**

