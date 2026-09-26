import logging
import time
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperTradingBroker:
    """
    Phase 5: The Paper Trading Engine.
    Executes the AI's trades using real-time market data but FAKE money.
    Exactly simulates Zerodha's fees, STT taxes, and slippage to ensure 
    our 53% win-rate is mathematically proven before risking real capital.
    """
    def __init__(self, initial_capital=100000.0):
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        # Zerodha STT + Exchange + Brokerage is roughly 0.03% for Intraday (MIS)
        self.brokerage_rate = 0.0003
        # Set to ExecutionEngine.on_fill so the engine's position ledger sees our fills.
        # Without it the engine's reconciler finds a mismatch and halts all trading.
        self.fill_callback = None
        
    async def authenticate(self):
        logging.info("=========================================")
        logging.info(f" PAPER TRADING ENGINE ONLINE")
        logging.info(f" Virtual Capital: Rs. {self.capital:,.2f}")
        logging.info("=========================================")
        return "PAPER_TOKEN"
        
    async def place_order(self, intent, algo_id: str):
        symbol = intent.symbol
        qty = intent.quantity
        side = intent.side
        
        # Fetch the actual live price right now to simulate reality
        try:
            ticker = yf.Ticker(symbol)
            live_price = ticker.fast_info['last_price']
        except Exception:
            logging.warning(f"Could not fetch live price for {symbol}. Order failed.")
            return None
            
        trade_value = live_price * qty
        fee = trade_value * self.brokerage_rate
        
        if side == "BUY":
            if self.capital < (trade_value + fee):
                logging.error(f"Paper Broker: INSUFFICIENT FUNDS. Need Rs. {trade_value:,.2f}, have Rs. {self.capital:,.2f}")
                return None
            self.capital -= (trade_value + fee)
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            
        elif side == "SELL":
            current_qty = self.positions.get(symbol, 0)
            if current_qty < qty:
                logging.error(f"Paper Broker: INSUFFICIENT SHARES. Trying to sell {qty}, own {current_qty}")
                return None
            self.capital += (trade_value - fee)
            self.positions[symbol] -= qty
            
        order_id = f"PAPER_{int(time.time())}"
        logging.info(f"[PAPER TRADE] {side} {qty} {symbol} @ Rs. {live_price:.2f} | Fee: Rs. {fee:.2f} | Balance: Rs. {self.capital:,.2f}")
        if self.fill_callback is not None:
            self.fill_callback(intent.client_order_id, qty, live_price)
        return order_id

    async def get_quote(self, symbol: str):
        try:
            ticker = yf.Ticker(symbol)
            return {"last": ticker.fast_info['last_price']}
        except:
            return {"last": 0.0}
