# Pre-registration: CM1, overnight commodity moves and commodity-linked stocks (written 2026-09-29, BEFORE any data and BEFORE any result)

## Question

Stocks whose profits depend directly on crude oil or gold should move with those
commodities. The question is whether a big commodity move made while NSE is shut gets
FULLY priced into the stock at the next open, or whether the stock keeps drifting the same
way during the next session (under-reaction). If it drifts, the gap can be traded intraday.

## Data (Yahoo daily only; runs in the cloud)

- Commodities: Brent crude futures `BZ=F` and gold futures `GC=F`, daily close.
- Stocks, fixed now by business line, before any data is seen:
  - **Crude, positive exposure (producers):** ONGC.NS, OIL.NS.
  - **Crude, negative exposure (fuel marketers and crude-input users):** BPCL.NS,
    HINDPETRO.NS, IOC.NS, ASIANPAINT.NS, BERGEPAINT.NS, MRF.NS, APOLLOTYRE.NS, CEATLTD.NS,
    INDIGO.NS.
  - **Gold, positive exposure (gold-loan lenders):** MUTHOOTFIN.NS, MANAPPURAM.NS.
- NIFTY (^NSEI), daily open and close.

## Timing

- A commodity's daily return on its trading date D (Yahoo date) is known before NSE opens on
  the first NSE trading day after D, called day N.
- Within day N, the stock's open-to-close return, minus NIFTY's open-to-close return, is the
  market-adjusted intraday move.
- (Part of the commodity's day-D move happens while NSE is still open on day D. That overlap
  is accepted, not modelled.)

## Trades

- **Trigger:** a trade happens on day N only if the commodity's day-D return is at least 1.5%
  in absolute value.
- **Direction:** the expected direction for a stock is the sign of the commodity's return
  times the group's exposure (+1 or -1).
- **Position:** at N's open, take a 1-unit position in each group stock in the expected
  direction (long, or short intraday), and close it at N's close.
- **Result per trade:** the direction times the stock's open-to-close return, minus NIFTY's
  open-to-close return in the same direction, minus 0.10% round-trip intraday cost.

## Measure

- The mean net result per trade, and the share of trades that make money (hit rate).
- A day-clustered t-statistic: trades on the same day N are averaged first.

## Periods

- **Development:** days N from 2016-01 to 2020-12.
- **Test:** 2021-01 to 2026-08. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - the mean net result is at least +0.10% per trade;
  - the day-clustered t is at least 2.0;
  - there are at least 100 trades.
- **Test, once:** the mean net result is above 0.
- **Only then:** an observe-mode alert ("crude moved 2% overnight: OMCs likely to drift"),
  before any paper trading on it.

## Also reported (not part of the decision)

- Each group separately (crude producers, crude users, gold lenders), and all three together.
- A 1.0% trigger instead of 1.5%.
- Gap capture: the stock's close-to-open gap on N versus the commodity move. How much of the
  move is priced at the open is reported as a regression slope.
- Jewellers (TITAN.NS, KALYANKJIL.NS) against gold, as an exploratory group, because their
  gold exposure is ambiguous.

## Caveats

- Daily data cannot isolate the commodity's move after NSE's close. The overlap adds noise,
  which makes a real effect harder, not easier, to find.
- Survivorship: the stock list is today's companies.
- A short leg needs intraday shorting (allowed in the cash market for intraday trades). The
  cost assumption covers brokerage, STT, exchange charges and slippage for liquid names.
