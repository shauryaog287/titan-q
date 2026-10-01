# Pre-registration: EV1, filings read by the local AI (written 2026-10-01, BEFORE any filing was labelled for this study and BEFORE any return was computed)

## Question

ORD1 found order wins by keyword matching and they did not drift up. Can a language model
running on the laptop (Qwen 2.5 7B through Ollama, `llm_backfill.py`) read every filing,
decide what kind of event it is and how important it is, and so find two kinds of filings?

- **H1:** filings that move the stock and keep moving it the same way, which would be a buy
  signal.
- **H2:** filings that drag the stock down for weeks, which would be an avoid signal like
  NJ1b.

## Filings in scope (fixed now)

- **Source:** the NSE corporate-announcements archive (`data/announcements/`, every filing
  since 2017; `order_research.py --fetch`).
- **Categories** (`desc`), the ones where business news is filed:
  - "Updates", "Press Release", "General Updates";
  - "Bagging/Receiving of orders/contracts", "Acquisition", "Credit Rating";
  - "Action(s) taken or orders passed", "Action(s) initiated or orders passed";
  - "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company";
  - "Change in Auditors";
  - "News Verification", "Clarification", "Reply to Clarification Sought", "News Clarification".
- **Left out:** results, dividends, record dates, ESOPs, board-meeting outcomes, investor
  presentations, takeover-code disclosures, share-certificate notices and the other
  procedural categories.
  - Quarterly results need numbers, not a one-line summary; they belong to the planned
    fundamentals study.
- **Stocks:** universe stocks that were liquid on the trading day the filing is attached to
  (the NJ1 definition):
  - median traded value of at least Rs. 1 crore over the previous 60 days;
  - previous close of at least Rs. 20;
  - at least 250 days of history.
- **Duplicates:** identical filings (same stock, category and text) are read once.

## Labelling (fixed now)

- **Model:** `qwen2.5:7b`, temperature 0, JSON output.
- **Prompt:** the exact text in `llm_backfill.py` (`PROMPT`), filled with the category and
  the first 1,500 characters of the filing's text as archived.
- **Labels:** event_type (from the `llm_reader.EVENT_TYPES` list), direction (positive,
  negative or neutral) and materiality (high, medium or low). value_cr is kept for the
  secondary tables.
- The labels are written before any return is looked at, and they are not edited afterwards.

## Measure

- **Timing:** a filing broadcast before 15:30 on a trading day enters at that day's close.
  Otherwise it enters at the next trading day's close. It exits 20 trading days later, at
  the close.
- **Excess:** the stock's return minus the equal-weight return of the liquid universe on
  the entry day over the same window (as NJ1).
- **Cost:** 0.50% for H1 (a trade). None for H2 (an avoid flag).
- **Merging:** for each stock and hypothesis, a qualifying filing less than 10 trading days
  after the previous one counts once.
- **t-statistic:** averaged per calendar month of entry, then computed over months.

## Hypotheses and decision rules (fixed now)

Both hypotheses leave out the event types results, routine and market_commentary.

- **H1 (buy candidate):** direction positive and materiality high.
  - Development qualifies if all three hold:
    - at least 200 events;
    - the mean 20-day net excess is at least +1.0%;
    - the month t is at least 2.0.
- **H2 (avoid rule):** direction negative and materiality high or medium.
  - Development qualifies if all three hold:
    - at least 200 events;
    - the mean 20-day excess is -1.0% or lower;
    - the month t is -2.0 or lower.

## Periods

- **Development:** filings from 2017-01 to 2020-12. These are labelled first.
- **Confirmation (sealed):** filings from 2024-10-01 to 2026-08-31.
  - This window starts after the model was released (September 2024). The model cannot
    know what happened to these stocks afterwards.
  - It is labelled only if a hypothesis qualifies in development, and it is run once.
  - **Pass for H1:** mean net excess above 0 and month t of at least 1.0.
  - **Pass for H2:** mean excess below 0 and month t of -1.0 or lower.
- **Not used:** 2021-01 to 2024-09. It falls inside the model's training period, where its
  judgement could reflect what later happened to the price, and it is not needed.

## If a hypothesis passes

- **H1:** first an observe-mode tag in the news and mover alerts ("historically drifts up,
  EV1"). Any paper strategy built on it gets its own pre-registration.
- **H2:** no buys of the stock for 20 trading days after such a filing, enforced the same
  way as NJ1b.

## Also reported (not part of the decision)

- Each event type separately (order_win, capacity_expansion, guidance, acquisition,
  fundraise, rating_change, regulatory_action, litigation, auditor_change, and so on).
- **Size thirds:** the bottom, middle and top third of the eligible stocks by 60-day median
  traded value. The bottom third answers "small stocks that boom on news".
- 5-day and 60-day horizons.
- Filings that state a rupee value against those that do not.

## Caveats

- **Look-ahead in development:** 2017-2020 is inside the model's training data. Its
  materiality judgement there could be coloured by what it knows happened next, which would
  flatter development. This is why only the post-release window can confirm.
- **Survivorship:** the universe is today's listed stocks.
- **Partial text:** the model reads the archived summary text, not the attached PDF.
