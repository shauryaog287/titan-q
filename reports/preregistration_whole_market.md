# Pre-registration: whole-market signal research (written 2026-09-28, BEFORE any run)

## Why

In the 50-stock large-cap universe no signal showed an edge (`reports/alpha_research.md`).
Anomalies are usually stronger and statistically easier to detect in a broad cross-section
(hundreds of stocks, mid caps included). This round asks whether any of the same seven
signals works across the whole liquid NSE market, and measures hit rate ("accuracy")
explicitly.

## Fixed now; nothing else will be tried in this round

- **Universe, point in time:** at each monthly formation date, every NSE equity from
  `universe.py` whose last 60 trading days BEFORE that date had a median traded value of at
  least Rs. 10 crore, a price of at least Rs. 20, and at least 2 years of history
  (`universe.liquid_at`). Survivorship bias remains, because only today's listed stocks are
  in the list.
- **Signals (the same seven as before):** MOM_12_1, MOM_6_1, REV_1M, LOWVOL, LOWBETA, HI52,
  TREND200, defined exactly as in `alpha_research.py`.
- **Timing:** signals at the close of the last trading day of each month; entry at the next
  day's close; held to the day after the next month-end.
- **Costs:** 25 bps per unit of weight traded. This is the 16 bps used for large caps plus
  about 10 bps of extra impact and spread for mid caps.
- **Portfolios:** long only. Top decile by signal, and top 30 names (a realistic book for
  Rs. 5 lakh), both equal-weighted, against the equal-weight point-in-time universe.
- **Periods:** development uses formation dates 2015-09 to 2020-12; the test uses 2021-01 to
  2026-08 and stays blind until the decision is committed.

## Measures

- Monthly rank IC (Spearman), its mean and t-statistic.
- Excess return of each portfolio over the equal-weight universe, annualised, and its
  t-statistic, after costs.
- **Hit rates:**
  - *relative*: the share of picked stocks that beat the universe median return next month;
  - *absolute*: the share of picked stocks with a positive net return next month;
  - *portfolio*: the share of months the portfolio beat the universe.

## Decision rule (written before any result exists)

A signal qualifies in development only if ALL of these hold:

1. mean rank IC t-statistic >= 2.7 (roughly a Bonferroni-corrected 5% level for 7 tests);
2. top-decile excess return > 0 with t >= 2.0, after costs;
3. relative hit rate of the top decile > 50%.

If two or more signals qualify, the candidate is their equal-weighted z-score composite. If
exactly one qualifies, it is the candidate. If none qualifies, nothing is adopted and the
result is recorded.

**Test (once):** the candidate is confirmed only if, over 2021-2026, its mean rank IC is
> 0 and its top-decile excess return after costs is > 0. Only then is it built into the live
system (a "rank" selection mode over the point-in-time liquid universe), and it still has to
beat v3.1 in a full walk-forward backtest before becoming the default.

## Honest caveats

- The 7 signals were already looked at in the 50-stock universe, over the same periods.
  This whole-market test is not fully independent of that, but the stocks are mostly
  different: most of the liquid universe is outside the top 50.
- Survivorship bias flatters every long-only result. Comparisons are therefore made against
  the equal-weight universe, which shares it.
- Hit rates near 50% are normal. Real edges in equities are usually 51-55%.
