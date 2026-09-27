# Running Titan-Q research in the cloud

The trading PC runs live paper trading (scheduled task, Windows). Heavy research and
retraining can run on a cloud machine instead, for example a Claude Code cloud session, while
the PC is off. The two stay in sync through the git repository.

## What runs where

| Job | Where | Why |
|---|---|---|
| Live paper trading (`main_trading_loop.py`) | trading PC | needs the paper account file, the dashboard, and to run through the NSE session |
| Retraining (`train_models.py`) | cloud or PC | CPU-heavy (~9 minutes), no state |
| Walk-forward backtest (`backtest_daily.py`) | cloud or PC | CPU-heavy (~35 minutes), no state |
| Signal research (`alpha_research.py`) | cloud or PC | fast, no state |

## Requirements for a cloud run

1. The repository must be on GitHub; a private repository is recommended. The cloud session
   clones it.
2. The cloud environment needs network access to Yahoo Finance (query1.finance.yahoo.com,
   query2.finance.yahoo.com, fc.yahoo.com) to download prices. `data/` is not in git. If
   Yahoo is blocked, allow those domains in the environment's network settings.
3. Set up and run:

```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python market_data.py --fetch        # downloads 12 years of daily bars + recent 1-minute bars
python train_models.py               # retrains, writes models/ and models/manifest.json
python backtest_daily.py             # writes reports/
python alpha_research.py             # writes reports/ (test period stays blind)
```

4. Commit `models/` and `reports/` to a branch (for example `research/<date>`), review the
   diff, then merge. The trading PC picks up new models with `git pull`.

## Rules that apply in the cloud too

- Pure ASCII in every file, and "Rs." for currency (see CLAUDE.md).
- A model is enabled only if it beats its simple baseline out of sample; the manifest
  records the evidence.
- `alpha_research.py --reveal-test` is used once per decision, after the decision is
  written down.
