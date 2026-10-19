# EAP1: the earnings announcement premium (development, meetings 2017-2020)

Implements reports/preregistration_eap1_earnings_premium.md. Entry at the close of the later of the day the meeting date became public and 5 trading days before the meeting; exit at the close of the first trading day after it. Result = stock minus the equal-weight liquid universe, minus 0.50% cost. 27,662 results meetings announced in advance; 7,935 trades in liquid stocks.

## Primary

| Group | Events | Mean | Median | Month t | Share > 0 |
|---|---|---|---|---|---|
| **Net result per trade** | 7,935 | -0.02% | -0.56% | -0.99 | 45.9% |

Rule: >= 500 trades, mean net >= +0.30%, month t >= 2.0.

**Decision: does not qualify; the test stays sealed.**

## Also reported (not part of the decision)

| Group | Events | Mean | Median | Month t | Share > 0 |
|---|---|---|---|---|---|
| Gross (no cost) | 7,935 | +0.48% | -0.06% | +0.22 | 49.7% |
| Pre-meeting part only (gross) | 7,909 | +0.84% | +0.38% | +2.24 | 54.7% |
| Announcement part only (gross) | 7,935 | -0.34% | -0.58% | -2.62 | 44.2% |
| Small third (net) | 2,638 | -0.03% | -0.73% | -2.35 | 44.3% |
| Mid third (net) | 2,631 | +0.17% | -0.32% | +2.24 | 47.5% |
| Large third (net) | 2,666 | -0.19% | -0.58% | -2.16 | 45.9% |
| Date announced > 7 days ahead (net) | 5,723 | -0.15% | -0.64% | -1.79 | 45.2% |
| Date announced <= 7 days ahead (net) | 2,212 | +0.33% | -0.32% | +1.85 | 47.7% |

