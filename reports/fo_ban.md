# FB1: stocks entering the F&O ban (development, 2017-2020)

Implements reports/preregistration_fb1_fo_ban.md. Excess = the stock's return from the close of its first ban day minus the equal-weight liquid universe over the same days; no cost (an avoid flag). 988 trading days of ban lists; an entry counts only if the stock was in none of the previous 10 days' lists.

## Primary

| Group | Events | Mean | Median | Month t | Share > 0 |
|---|---|---|---|---|---|
| **Ban entry, next 10 trading days** | 239 | -0.06% | -0.68% | +0.08 | 44.8% |

Rule: >= 150 entries, mean <= -1.0%, month t <= -2.0.

**Decision: does not qualify; the test stays sealed.**

## Also reported (not part of the decision)

| Group | Events | Mean | Median | Month t | Share > 0 |
|---|---|---|---|---|---|
| First ban day itself (close t-1 to t, not tradeable) | 239 | -0.68% | -0.85% | -1.57 | 34.7% |
| Ban entry, 5 days | 239 | +0.51% | -0.77% | +0.89 | 44.4% |
| Ban entry, 20 days | 239 | +0.38% | -0.90% | +0.47 | 45.2% |
| Ban exit, 10 days after the first day out | 239 | -1.09% | -1.43% | -1.64 | 38.9% |
| Spells of 5+ days, entry 10 days | 96 | -0.04% | -0.63% | +0.02 | 43.8% |
| Spells under 5 days, entry 10 days | 143 | -0.08% | -0.81% | -0.06 | 45.5% |

