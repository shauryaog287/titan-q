# Pre-registration: NJ1, news-backed versus no-news price jumps (written 2026-10-01, BEFORE the announcement archive is complete and BEFORE any result)

## Question

When a stock jumps on heavy volume, does it keep rising over the next days if there was real
news (an NSE filing), and fade if there was none? Chan (2003, "Stock price reaction to news
and no-news") found exactly that in US stocks. If it holds on NSE, buying news-backed jumps at
the close is a candidate strategy, and the live mover scanner (`mover_scanner.py`) already
detects them.

## Data

- Prices and volumes: the Yahoo daily panels of the whole universe
  (`whole_market_research.load_panels`), today's listed stocks (survivorship noted).
- News: the NSE announcement archive (`order_research.py --fetch`, one file per day,
  2017-01 onward).

## Events

- **A jump:** stock s on day t has a close-to-close return of at least +5%, and volume on t of
  at least 3 times its average volume over the 20 days before t.
- **Liquidity, decided on data before t:** median daily traded value over the previous 60
  days of at least Rs. 1 crore, last close at least Rs. 20, and at least 250 days of history.
  This is a lower bar than the Rs. 10 crore used elsewhere, because the question is about
  smaller stocks.
- **News-backed:** at least one NSE announcement by s broadcast between 15:30 on the previous
  trading day and 15:30 on t, excluding routine categories, fixed now:
  - Trading Window;
  - Shareholders meeting;
  - Copy of Newspaper Publication;
  - Analysts/Institutional Investor Meet/Con. Call Updates;
  - Closure of Trading Window;
  - Certificate under SEBI (Depositories and Participants) Regulations, 2018;
  - Compliances-Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018.
- **No-news:** no such announcement in that window.
- **Merging:** a stock's events less than 10 trading days apart count once (the first).

## Measure

- **Entry:** at the close of t. By then the jump and the filing are known.
- **Primary:** the excess return from the close of t to the close of t+5, against the
  equal-weight return of all stocks passing the same liquidity bar on t, minus 0.50%
  round-trip cost. Measured for news-backed jumps.
- **Clustering:** events are averaged by date, and t is computed over dates.

## Periods

- **Development:** event dates 2017-01 to 2020-12.
- **Test:** 2021-01 to 2026-08. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold for news-backed jumps:
  - the mean net 5-day excess is at least +0.5%;
  - the date-clustered t is at least 2.0;
  - there are at least 150 events.
- **Test, once:** the mean net 5-day excess is above 0.
- **Only then:** an observe-mode "buy candidate" tag in the mover scanner, then a shadow paper
  test, never straight to live trading.

## Also reported (not part of the decision)

- No-news jumps, the same measure; Chan's finding predicts a negative value.
- 1 and 20-day horizons.
- News-backed minus no-news, with its t.
- Jumps of at least +10%.
- Downward moves of -5% or worse, with and without news.

## Caveats

- Announcement times come from NSE's broadcast time. A company that filed on BSE first may
  look "no-news" here, which would blur the contrast.
- Upper-circuit stocks can close at the limit, and a real buy at that close may not fill.
  That flatters the measured returns slightly.
- Survivorship: the price panel holds today's listed stocks.
