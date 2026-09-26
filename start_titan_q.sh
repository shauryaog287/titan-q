#!/bin/bash
echo "======================================================="
echo "         TITAN-Q LIVE MARKET DEPLOYMENT (NSE)          "
echo "                 GOOGLE CLOUD EDITION                  "
echo "======================================================="
echo ""
echo "Checking data feed and AI sub-systems..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Launch the Whole Market Feed in the background
nohup python3 -u nse_whole_market_feed.py > market_feed.log 2>&1 &
echo "[OK] Market Data Websocket Launched (PID: $!). Logging to market_feed.log"

# Wait a couple seconds to ensure live_state.json is seeded
sleep 3

# Launch the main trading loop
nohup python3 -u main_trading_loop.py > trading_brain.log 2>&1 &
echo "[OK] Master Trading Loop Armed (PID: $!). Logging to trading_brain.log"

echo ""
echo "======================================================="
echo "SYSTEM LIVE. EXECUTING PAPER TRADES AGAINST LIVE TICKS."
echo "You can safely close this SSH terminal. The bot will"
echo "continue to run in the background 24/7."
echo "To monitor, run: tail -f trading_brain.log"
echo "======================================================="
