# INS1: insider open-market purchases (SEBI PIT disclosures)

Implements reports/preregistration_insider_buys.md. Net excess = stock return from the close after the disclosure date over 60 trading days, minus the equal-weight eligible universe, minus 0.50% cost.

## Development (entries 2016-01 to 2020-12)

Disclosures that could not be mapped to a current ticker: 2,382.

| Events | N | Mean net excess | Median | Month-clustered t | Hit rate |
|---|---|---|---|---|---|
| **Primary: insider buys, 60 days** | 1,158 | +2.28% | +0.01% | +2.23 | 50.0% |
| insider buys, 20 days | 1,158 | +0.31% | -0.06% | +1.36 | 49.6% |
| insider buys, 120 days | 1,158 | +4.78% | +0.48% | +3.09 | 50.8% |
| promoters / promoter group only | 1,013 | +2.06% | -0.10% | +1.42 | 49.4% |
| directors / key managers only | 160 | +4.26% | +2.14% | +2.34 | 53.8% |
| buys of Rs. 1 crore or more | 828 | +2.34% | +0.40% | +2.46 | 52.4% |
| insider SALES (negative = signal) | 1,830 | -0.18% | -0.69% | -0.37 | 48.4% |

Rule: mean net excess >= +1.0%, month-clustered t >= 2.0, >= 100 events.

**Decision: qualifies; reveal the test once.**

## Test (entries 2021-01 onward, revealed once)

Disclosures that could not be mapped to a current ticker: 2,382.

| Events | N | Mean net excess | Median | Month-clustered t | Hit rate |
|---|---|---|---|---|---|
| **Primary: insider buys, 60 days** | 1,147 | -1.42% | -3.04% | -1.66 | 41.2% |
| insider buys, 20 days | 1,147 | +0.04% | -0.21% | -0.26 | 48.9% |
| insider buys, 120 days | 1,146 | -1.53% | -5.10% | -1.23 | 41.4% |
| promoters / promoter group only | 1,042 | -1.47% | -2.85% | -1.74 | 41.3% |
| directors / key managers only | 110 | -0.26% | -3.72% | +0.11 | 41.8% |
| buys of Rs. 1 crore or more | 760 | -0.74% | -2.57% | -0.83 | 41.3% |
| insider SALES (negative = signal) | 1,990 | -1.09% | -2.34% | -1.55 | 44.3% |

**Test: failed.**

