# Pre-registration: M1, intraday direction model (written 2026-09-28, BEFORE the data is complete and BEFORE any run)

## Data

- Angel One 1-minute candles (`angel_history.py`), 2020-01 to 2026-09, for the tradable
  universe, plus NIFTY and BANKNIFTY.
- Stocks: at the start of each month, the 100 most liquid by median 60-day traded value
  from daily data dated before that month (point in time).
- Bars are resampled to 5 minutes. Decision times are every 30 minutes from 09:45 to 14:45
  IST (11 per day).

## Features at decision time t (all from bars ending at or before t)

- stock log returns over the last 5, 15, 30 and 60 minutes, and since 09:15;
- the overnight gap (today's open vs yesterday's close);
- relative volume: the last 15 minutes' volume divided by the 20-day average volume at the
  same time of day;
- realised volatility over the last 30 minutes, relative to the 20-day daily volatility;
- VWAP deviation: price vs the session VWAP, in percent;
- NIFTY and BANKNIFTY returns over the last 5, 15 and 30 minutes and since the open;
- stock minus NIFTY return over the last 30 minutes and since the open;
- minutes since the open;
- previous-day stock return and previous-day high-low range.

## Label

The sign of the stock's return from t to t + 30 minutes (5-minute close to close).

## Models (fixed; no tuning)

- Logistic regression (standardised features, C = 0.1).
- HistGradientBoostingClassifier (max_depth 4, learning_rate 0.05, 300 iterations,
  min_samples_leaf 500, l2 1.0).
- Walk-forward: refit every 3 months on all earlier data (expanding window, at least
  12 months). Predictions for a quarter come only from the model fitted before it began.

## Trading rule for evaluation

- At each decision time and for each stock: long if P(up) > 0.55, short if P(up) < 0.45
  (intraday MIS shorting), otherwise nothing.
- Hold 30 minutes.
- Cost: 10 bps per round trip (intraday brokerage, 0.025% STT on the sell side, exchange,
  SEBI and stamp charges, GST and 2 bps slippage each way).

## Measures

- accuracy on traded signals (direction right);
- mean net return per trade, in bps;
- a day-level t-statistic (daily average net P&L across trades, so correlated trades on
  the same day are not treated as independent);
- the Sharpe ratio of the daily P&L;
- accuracy of all predictions (P > 0.5 vs actual).

## Periods

- **Development:** predictions dated 2021-01 to 2023-12.
- **Test:** predictions dated 2024-01 to 2026-09; blind until the development decision is
  committed.

## Decision rule (fixed now)

- **Qualifies in development** if, for either model: accuracy on traded signals > 52%, AND
  mean net return per trade > 0, AND day-level t >= 2.0. If both qualify, take the higher t.
- **Test, once:** accuracy on traded signals > 50% AND mean net per trade > 0.
- **Only then:** a shadow forward paper test on live Angel data, before any real use.

## Caveats

- 30-minute direction is a hard target. Published intraday accuracies are typically 51-54%
  before costs.
- Choosing the top 100 by liquidity point in time avoids look-ahead in the universe, but
  today's listed stocks are still the pool (survivorship).
- The 10 bps cost is realistic for liquid stocks with a discount broker, but slippage on
  larger orders would be higher.
