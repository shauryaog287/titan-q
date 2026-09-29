@echo off
color 0A
title Titan-Q AI Master Control
cd /d D:\Quant_Project
echo =======================================================
echo          TITAN-Q LIVE MARKET DEPLOYMENT (NSE)
echo =======================================================
echo.
echo Checking data feed and AI sub-systems...
echo.

:: The synthetic GBM feed (nse_whole_market_feed.py) is no longer launched here:
:: the trading loop never read its output. Market data now comes from
:: market_data.py inside the trading loop itself.

:: Launch the main trading loop under its supervisor (run_live.py restarts it after a crash
:: or a requested restart during the session). "cmd /k" keeps the window open if it exits,
:: so an error stays readable. Everything is also logged to logs\titanq_*.log
start "Titan-Q Master Brain" cmd /k "D:\trader_env\Scripts\python.exe D:\Quant_Project\run_live.py"
echo [OK] Master Trading Loop Armed.

:: Launch the Angel One tick + order-book recorder (read-only; whole NSE market).
:: It exits by itself on weekends, after the close, or if angel_credentials.json
:: is not set up yet. Log: logs\angel_record_*.log
start "Titan-Q Tick Recorder" cmd /k "D:\trader_env\Scripts\python.exe D:\Quant_Project\angel_feed.py --record"
echo [OK] Tick Recorder Started.

:: Launch the Web Dashboard server (localhost only)
start "Titan-Q Web Dashboard" cmd /k "D:\trader_env\Scripts\python.exe D:\Quant_Project\dashboard_server.py"
echo [OK] Live Web Dashboard Running on http://localhost:5000

echo.
echo =======================================================
echo SYSTEM LIVE. PAPER TRADING DURING NSE HOURS (09:15-15:30 IST).
echo To safely shut down, close all command windows.
echo =======================================================
:: No "pause" here: the scheduled task counts as running until this script ends, and
:: with "do not start a new instance" a window left waiting at a pause would make
:: Task Scheduler skip the next day's 09:10 start. The three windows above stay open.
timeout /t 20 >nul
