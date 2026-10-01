# Pre-registration: ORD1, order-win announcements (written 2026-10-01, BEFORE the announcements are downloaded and BEFORE any result)

## Question

Companies, notably government-owned and defence suppliers, announce new orders and
contracts on NSE. Does the stock keep rising after such an announcement beyond the market
(post-announcement drift), or is the news fully priced at once?

## Data

- **Announcements:** NSE corporate announcements (`/api/corporate-announcements`), one
  request per calendar day, 2017-01-01 onward, downloaded slowly (5 s apart). Checked on
  2026-10-01: the archive reaches 2017. In older years order wins were not filed under their
  own category, which is why the event definition below is based on text.
- **Prices:** the Yahoo daily panel of the whole universe (`whole_market_research.load_panels`).
- **Eligibility:** `eligible_at` at entry.

## Events

An announcement is an order win if its category (`desc`) or its text (`attchmntText`)
matches, case-insensitively, this pattern, fixed now:

    bag(ged|ging)? (an |a |the )?(new )?(order|contract)s?
    | receiv(ed|ing|es)? (an |a |the )?(new )?(order|contract|letter of award|work order)s?
    | letter of (award|acceptance|intent)
    | work order
    | (order|contract)s? (worth|valued at|of value|aggregating)
    | award(ed)? (of )?(an |a |the )?(order|contract)

The following are excluded:
- text mentioning "cancel", "terminat", "withdraw" or "rumour";
- the category "Clarification - Financial results".

Timing and eligibility:
- **Event date:** the broadcast date (`an_dt`). Entry is at the close of the first trading day
  after that date, so the news is always public before entry.
- **Merging:** announcements by the same stock with the same entry day are one event.
- **Eligibility:** the stock must pass `eligible_at` on the entry day.

## Measure

- **Primary:** the excess return over 20 trading days from entry, against the equal-weight
  eligible universe over the same days, minus 0.50% round-trip cost.
- **Clustering:** events are averaged by calendar month of entry, and t is computed over
  months.

## Periods

- **Development:** entries 2017-01 to 2020-12.
- **Test:** entries 2021-01 to 2026-08. Blind until the development decision is committed.

## Decision rule (fixed now)

- **Development qualifies** if all three hold:
  - the mean net excess is at least +1.0% per event;
  - the month-clustered t is at least 2.0;
  - there are at least 100 events.
- **Test, once:** the mean net excess is above 0.
- **Only then:** a shadow forward test of buying after order wins, never straight to live
  trading.

## Also reported (not part of the decision)

- 5 and 60-day horizons.
- Events whose text gives a value in Rs. crore, where it can be parsed, split at the median
  value.
- Stocks Yahoo labels "Aerospace & Defense" or utilities, and the names of companies with
  "Bharat", "Hindustan", "Garden Reach", "Mazagon", "Cochin Shipyard" or "BEML", where
  government-owned and defence companies cluster.
- The day-0 reaction: the close of the day before the announcement to the entry close, to
  show how much was priced at once.

## Caveats

- **Text matching** will misclassify some announcements. Misclassification adds noise and
  makes an effect harder, not easier, to find.
- **Survivorship:** the price panel holds today's listed stocks.
- **Timing:** some companies announce on BSE first. NSE's broadcast time can only make the
  entry later.
