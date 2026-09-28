# Pre-registration: ML models M2 and M3 (written 2026-09-28, BEFORE any run)

Both use walk-forward training: every prediction comes from a model fitted only on data
dated before it. Development and test periods are the same as in earlier rounds; the test
stays blind until the development decision is committed. `ml_research.py` implements both.

## M2: whole-market ML ranker (monthly)

- **Universe and timing:** exactly as `whole_market_research.py`. Point-in-time liquid NSE
  stocks; signals at the month-end close; entry at the next close; 25 bps per unit traded.
- **Features per stock:**
  - the seven signals (MOM_12_1, MOM_6_1, REV_1M, LOWVOL, LOWBETA, HI52, TREND200);
  - 1-month volatility;
  - volume trend (20-day over 252-day average volume);
  - log median 60-day traded value (size or liquidity);
  - distance from the 52-week low.

  All are cross-sectionally ranked to [0, 1] each month. Market context is added as
  constants per month: NIFTY 1-month return, India VIX level, and India VIX 1-month change.
- **Label:** the cross-sectional percentile rank of next month's return.
- **Model:** sklearn HistGradientBoostingRegressor (max_depth 3, learning_rate 0.05,
  200 iterations, min_samples_leaf 200, l2 1.0). Fixed now; no tuning.
- **Walk-forward:** at month t, train on all (features, label) pairs whose label was
  fully known before t, with at least 24 months required.
- **Compared with:** the equal-weight seven-signal composite (round 2), on the same months.
- **Qualifies in development only if ALL hold:**
  1. IC t-statistic >= 2.7;
  2. top-decile excess over the equal-weight universe > 0 with t >= 2.0 after costs;
  3. relative hit rate > 50%;
  4. mean IC > the composite's mean IC over the same months.
- **Test, once:** mean IC > 0 and top-decile excess after costs > 0.

## M3: NIFTY intraday direction from pre-open information (daily)

- **Decision:** before 09:15 IST on day D, predict the sign of NIFTY's OPEN-to-CLOSE return
  on D.
- **Features known before the open, with nothing dated D from any market that is open
  during India's session:**
  - US: S&P 500, Nasdaq and Dow returns of their last completed session; US VIX change;
    US 10-year yield change; dollar-index change;
  - commodities: Brent, gold and silver, last completed session;
  - FX: USD/INR, last completed close;
  - Asia and Europe: Nikkei, Hang Seng, Shanghai, KOSPI, FTSE and DAX, D-1 closes only
    (Asian markets are open during India's session, so their day-D values are excluded);
  - India, D-1: NIFTY close-to-close return, 5-day and 20-day returns, 20-day realised
    volatility, India VIX level and change, BANKNIFTY minus NIFTY return, Midcap 50 minus
    NIFTY return, and NIFTY's previous open-to-close return.
- **Models:** logistic regression (L2, C = 0.1, standardised features) and
  HistGradientBoostingClassifier (max_depth 3, learning_rate 0.05, 200 iterations,
  min_samples_leaf 50). Fixed now.
- **Walk-forward:** refit every 21 trading days on all earlier days (at least 500 days).
- **Trading rule for evaluation:**
  - long NIFTY from open to close when P(up) > 0.55;
  - short when P(up) < 0.45 (index futures allow shorting);
  - flat otherwise;
  - 2 bps cost per side.
- **Baselines:** always long open-to-close, and flat.
- **Qualifies in development only if ALL hold, for either model:**
  1. accuracy on traded days > 52%;
  2. mean daily net return of the rule > 0 with t >= 2.0;
  3. rule Sharpe > the always-long open-to-close Sharpe.

  If both models qualify, take the higher development Sharpe.
- **Test, once:** accuracy on traded days > 50% and net rule return > 0.

## Periods

- **Development:** predictions dated 2016-01 to 2020-12. M2 predictions start once 24
  training months exist.
- **Test:** 2021-01 to 2026-08 (M2) and 2021-01 to 2026-09 (M3); blind until the decision
  is committed.

## Caveats

- The 5 macro series and the 7 signals were chosen from the literature and from what is
  available, not searched over. Hyperparameters are fixed defaults, not tuned.
- M2's features overlap with the seven signals already studied, so its development result
  is not independent of round 1 and round 2. The test is what counts.
- Survivorship bias affects M2 as before, but not M3 (an index).
