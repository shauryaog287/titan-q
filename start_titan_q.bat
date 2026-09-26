@echo off
color 0A
title Titan-Q AI Master Control
echo =======================================================
echo          TITAN-Q LIVE MARKET DEPLOYMENT (NSE)
echo =======================================================
echo.
echo Checking data feed and AI sub-systems...
echo.

:: Launch the Whole Market Feed in the background
start "NSE Whole Market Feed" cmd /c "D:\trader_env\Scripts\python.exe D:\Quant_Project\nse_whole_market_feed.py"
echo [OK] Market Data Websocket Launched.

:: Wait a couple seconds to ensure live_state.json is seeded
timeout /t 3 /nobreak > nul

:: Launch the main trading loop
start "Titan-Q Master Brain" cmd /c "D:\trader_env\Scripts\python.exe D:\Quant_Project\main_trading_loop.py"
echo [OK] Master Trading Loop Armed.

:: Launch the Web Dashboard server
start "Titan-Q Web Dashboard" cmd /c "D:\trader_env\Scripts\python.exe D:\Quant_Project\dashboard_server.py"
echo [OK] Live Web Dashboard Running on http://localhost:5000

echo.
echo =======================================================
echo SYSTEM LIVE. EXECUTING PAPER TRADES AGAINST LIVE TICKS.
echo To safely shut down, close all command windows.
echo =======================================================
pause
