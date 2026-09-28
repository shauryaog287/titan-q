# Exploratory, NOT a protocol result: the "market" variant (run 2026-09-27)

`backtest_daily.py --variant market`: every regime multiplier 1.0 (CRASH included), no novel
limit, no EVT trim, full Kelly capped at 100%. This is a fully invested, inverse-volatility
portfolio of the universe. It was run AFTER the v3 results for 2019-2026 had been seen, so it
is not evidence for switching to it. Report: `backtest_daily_report_market.md`.

| 2019-01 to 2026-09 | CAGR | Volatility | Sharpe | Max drawdown | Total |
|---|---|---|---|---|---|
| market variant | 16.4% | 15.0% | 1.09 | 33.2% | +216% |
| v3 (live) | 5.9% | 10.7% | 0.59 | 24.0% | +54% |
| NIFTY buy-and-hold | 10.6% | 17.3% | 0.67 | 38.4% | +114% |
| Equal-weight universe | 16.6% | 16.9% | 1.00 | 36.7% | +221% |

How to read it:

- It matches the equal-weight universe (same survivorship bias) with slightly lower
  volatility and drawdown. That is what inverse-volatility weighting of a fully invested
  book should do, and it agrees with `alpha_research.md` (construction study, development
  and test). It is not stock-picking skill.
- Its lead over NIFTY is flattered by survivorship (today's 50 names) and by dividends
  (stocks are total return; NIFTY is a price index).
- The choice between it and v3 is mainly one of risk appetite: about 33% versus 24% maximum
  drawdown for roughly three times the return in this window. Any change to the live
  default goes through the pre-registered protocol
  (`reports/preregistration_sizing_vol.md` and its successors), not through this table.
