# INS2: promoter pledge invocations as a red flag

Implements reports/preregistration_pledge_invocations.md. Excess = stock return over the next 60 trading days (from the close after NSE's broadcast date) minus the equal-weight eligible universe; no cost (the flag works by not holding).

## Development (entries 2016-01 to 2020-12)

| Events | N | Mean excess | Median | Month-clustered t | Share negative |
|---|---|---|---|---|---|
| **Primary: promoter pledge invocations, 60 days** | 52 | -3.69% | -4.42% | -0.94 | 57.7% |
| invocations, 20 days | 52 | +0.32% | -2.43% | -0.16 | 57.7% |
| invocations, 120 days | 52 | -4.69% | -2.50% | -0.30 | 55.8% |
| invocations, any person category | 290 | -2.10% | +0.17% | -0.65 | 49.3% |
| new promoter pledges | 1,521 | +1.16% | -1.84% | +1.71 | 55.6% |
| promoter pledge releases | 1,313 | +3.72% | -0.44% | +1.89 | 51.4% |

Rule: mean excess <= -2.0%, month-clustered t <= -2.0, >= 30 events.

**Decision: does not qualify; the test stays sealed.**

