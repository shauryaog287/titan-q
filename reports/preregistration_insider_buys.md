# Pre-registration: INS1, insider open-market purchases (written 2026-09-28, BEFORE the bulk download and BEFORE any result)

## Question

When a company's promoters, directors or key managers buy the company's shares on the open
market and disclose it, does the stock beat the market over the following three months?
The disclosures are public under SEBI's Prohibition of Insider Trading Regulations 2015
(Regulation 7(2)); this study uses only what was published, and only after it was published.

## Data

- NSE corporate filings, insider trading (PIT) disclosures, one request per calendar month.
  Checked on 2026-09-28 with a few sample months: the API returns records from 2016-01
  to about 2026-01, and nothing before 2016 or after early 2026. The study uses whatever
  it returns. The only records inspected before this was written were field names and
  category counts for 2021-01, with no prices or returns.
- Prices: Yahoo daily closes of the whole NSE universe (the panels of
  `whole_market_research.py`).

## Events

A disclosure counts if all of these hold:
- acquisition mode "Market Purchase" and transaction type "Buy";
- person category "Promoters", "Promoter Group", "Director" or "Key Managerial Personnel";
- security type "Equity Shares";
- value at least Rs. 10 lakh.

Rules for dates and eligibility:
- **Timing:** the event date is the date of NSE's broadcast of the disclosure. The position
  is entered at the close of the first trading day AFTER that date, so the information is
  always public before entry, even when the broadcast came during market hours.
- **Merging:** disclosures for the same stock with the same entry day are one event.
- **Eligibility at entry:** the stock must pass the whole-market rule (`eligible_at`:
  median 60-day traded value at least Rs. 10 crore, minimum price and history), the same
  bar the forward test uses.

## Measure

- For each event, the net excess return is:
  - the stock's return from the entry close to the close 60 trading days later;
  - minus the equal-weight return of all eligible stocks over the same days;
  - minus 0.50% round-trip cost.
  If the stock stops trading inside the window, its last price counts.
- Events are averaged by calendar month of entry, and the t-statistic is computed over
  months, so a burst of events in one month does not count as independent evidence.

## Periods

- **Development:** entries 2016-01 to 2020-12.
- **Test:** entries from 2021-01 to the last entry that has 60 trading days of prices after
  it. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - the mean net excess return is at least +1.0% per event;
  - the month-clustered t is at least 2.0;
  - there are at least 100 events.
- **Test, once:** the mean net excess return is above 0.
- **Only then:** a shadow forward paper test, like the composite's. Never straight to live
  trading.

## Also reported (not part of the decision)

- 20 and 120 trading-day horizons.
- Promoters and promoter group only, versus directors and key managers only.
- Purchases of Rs. 1 crore or more.
- Insider market SALES, as a possible negative signal, measured the same way.
- The hit rate: the share of events with positive net excess.

## Caveats

- **Survivorship:** the price panel holds today's listed stocks. Companies that collapsed
  and were delisted after insiders bought are missing, which flatters the result.
- **Timing:** some companies file on BSE first. Using NSE's broadcast time can only make
  the entry later, never earlier.
- **Symbol changes:** a stock whose NSE symbol changed since the disclosure may not map to
  today's ticker. Unmapped events are counted and reported.
