@echo off
title Titan-Q sizing-volatility experiment (development period)
cd /d D:\Quant_Project
:: Step 1 of reports\preregistration_sizing_vol.md: three backtests over 2016-2020
:: (about 35 minutes each). Reads the cached daily bars only; it does not touch
:: the live paper account, so it can run overnight next to the scheduled task.
:: Results: reports\backtest_daily_report_dev_*.md, log in logs\sizing_experiment.log
if not exist logs mkdir logs
set PY=D:\trader_env\Scripts\python.exe
set LOG=logs\sizing_experiment.log
echo started %DATE% %TIME% > %LOG%
for %%V in (live ewma_vol_21 ewma_vol_63) do (
    echo === %%V >> %LOG%
    %PY% backtest_daily.py --start 2016-01-01 --end 2020-12-31 --variant %%V --tag _dev_%%V >> %LOG% 2>&1
)
echo finished %DATE% %TIME% >> %LOG%
echo Done. Apply the decision rule in reports\preregistration_sizing_vol.md BEFORE any test-period run.
