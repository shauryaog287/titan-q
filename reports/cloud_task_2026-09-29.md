# Cloud task, 2026-09-29: run the CM1 and SM1 development studies

You are running two pre-registered research studies for Titan-Q in a cloud sandbox, on this
repository (branch main). Read CLAUDE.md and CLOUD.md first; their rules apply.

## Hard rules

1. **ASCII only.** Every file you write (code, comments, strings, reports, CSV) must be pure
   ASCII. Currency is written "Rs.", never with the rupee sign. After writing each file,
   verify it has zero bytes above 127.
2. **Development period only.** Each study has a sealed test period (2021-01 to 2026-08).
   - Do not compute, print or look at any test-period metric.
   - Do not add or run a `--reveal-test` path.
   - Apply the development decision mechanically from the pre-registration. Do not tune any
     parameter, threshold, stock list or window after seeing results.
3. **New files only.** Do not modify existing trading code (main_trading_loop.py,
   titanq_brain.py, paper_broker.py and so on). Import helpers, and add only
   commodity_research.py, sector_momentum.py and their reports. If a tiny additive change to a
   helper is unavoidable, keep the old behaviour identical and say so.
4. **Git.** Never commit data/ (it is git-ignored). Never push to main. Commit to a new branch
   named `research/2026-09-29-cm1-sm1` and push that branch.
5. **No logins.** No credentials exist or are needed. Do not use Angel One or anything that
   needs a login.

## Setup

- Python 3.12. `.claude/hooks/session-start.sh` builds the venv. Otherwise run:
  `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`, then
  `pip install -r requirements.txt`.
- Network access is needed to:
  - Yahoo Finance: query1.finance.yahoo.com, query2.finance.yahoo.com and fc.yahoo.com;
  - margincalculator.angelbroking.com, the public instrument list used by universe.py, which
    needs no login.

  If Yahoo is blocked, stop and report that.

## Study 1, CM1: `commodity_research.py`

Implement `reports/preregistration_commodity_linked.md` exactly.

- **Data:**
  - Yahoo daily Open and Close with `auto_adjust=False`, 2015 to today, for BZ=F, GC=F and
    ^NSEI.
  - Crude-positive stocks: ONGC.NS, OIL.NS.
  - Crude-negative stocks: BPCL.NS, HINDPETRO.NS, IOC.NS, ASIANPAINT.NS, BERGEPAINT.NS, MRF.NS,
    APOLLOTYRE.NS, CEATLTD.NS, INDIGO.NS.
  - Gold-positive stocks: MUTHOOTFIN.NS, MANAPPURAM.NS.
  - Exploratory jewellers: TITAN.NS, KALYANKJIL.NS.
- **Timing:** commodity date D maps to N, the first ^NSEI trading day strictly after D.
- **Trigger:** the commodity's return on D is at least 1.5% in absolute value.
- **Trade:** each group stock, from N's open to N's close. Direction = sign(commodity return)
  x exposure.
- **Result per trade:** direction x (stock open-to-close) - direction x (NIFTY open-to-close)
  - 0.0010. Skip a stock-day with a missing or zero open or close.
- **Development (days N from 2016-01-01 to 2020-12-31):**
  - measure the mean net result per trade, the hit rate, and a day-clustered t (average the
    trades per day N, then compute t over days);
  - it qualifies if the mean is at least +0.10%, t is at least 2.0, and there are at least
    100 trades.
- **Also report (not decisive):**
  - each group separately;
  - a 1.0% trigger;
  - the gap regression slope (the stock's close(N-1) to open(N) return on exposure x
    commodity return);
  - jewellers against gold.
- **Output:** `reports/commodity_linked.md` and `.json`, containing the development table and
  one decision line: "Decision: qualifies; reveal the test once" or "Decision: does not
  qualify; the test stays sealed".

## Study 2, SM1: `sector_momentum.py`

Implement `reports/preregistration_sector_momentum.md` exactly.

- **Universe:** run `python universe.py` (it takes a while). Then use
  `whole_market_research.load_panels()` and `eligible_at()` exactly as the existing study
  does, with the same month-end mechanics as `monthly_study`: entry at row a+1, exit at the
  next month end's a+1.
- **Industries:**
  - for every ticker eligible at any month end from 2016-01 to 2026-08, fetch
    `yfinance.Ticker(t).info['industry']` once;
  - be polite: about 1 s between calls, retry on errors, skip a ticker on persistent failure;
  - cache the result to data/industries.csv, and save a copy as `reports/industry_map.csv`
    (columns ticker, industry);
  - fetching labels for test-period stocks is fine; computing test-period returns is not.
- **Month end a:**
  - industry score = the equal-weight mean over eligible members of (price at row a-21 /
    price at row a-252 - 1);
  - rank only industries with at least 5 eligible members;
  - hold the top 3 industries' eligible stocks, equal weight;
  - benchmark: the equal-weight eligible universe;
  - cost: 25 bps per unit of turnover.
- **Development (month ends 2016-01 to 2020-12):**
  - measure the annualised mean monthly excess, the t over months, and the share of months
    that beat the benchmark;
  - it qualifies if the excess is at least +3.0% a year, t is at least 2.0, and it beats the
    benchmark in at least 55% of months.
- **Also report (development only):**
  - the top 1 and top 5 industries;
  - a 6-1 lookback;
  - the bottom 3 industries;
  - how often "Aerospace & Defense" and the utilities, banks and energy industries were in
    the top 3.
- **Output:** `reports/sector_momentum.md` and `.json`, with the decision line.

## Finish

1. Run both scripts end to end and check that the reports are ASCII.
2. Commit commodity_research.py, sector_momentum.py, reports/commodity_linked.*,
   reports/sector_momentum.* and reports/industry_map.csv to
   `research/2026-09-29-cm1-sm1`, then push.
3. In the final message, give:
   - the branch name and commit hash;
   - for each study, the development numbers and the decision;
   - any deviation from the pre-registration, and why;
   - anything that failed.

Do not reveal or estimate test-period results.
