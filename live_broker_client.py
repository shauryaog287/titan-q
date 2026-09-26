import logging
import asyncio

# In a real environment, you run: pip install kiteconnect
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ZerodhaBrokerClient:
    """
    Phase 5: Live Execution API (Zerodha Kite Connect)
    This module replaces the 'MockBroker' and physically connects our AI 
    to the National Stock Exchange (NSE).
    """
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
        if KiteConnect is None:
            logging.error("kiteconnect library is missing. Run: pip install kiteconnect")
            
        self.kite = KiteConnect(api_key=self.api_key) if KiteConnect else None
        self.access_token = None

    async def authenticate(self, request_token: str = None) -> str:
        """
        Zerodha requires a daily physical login. 
        Later, we can use our Stealth Selenium Scraper to automate clicking the 'Login' button.
        """
        if not self.kite:
            return "MISSING_LIBRARY"
            
        if not request_token:
            login_url = self.kite.login_url()
            logging.warning(f"ACTION REQUIRED: Please login manually at 8:50 AM to get request_token: {login_url}")
            return "AUTH_PENDING"
            
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            logging.info("Zerodha Authentication Successful. Live connection established.")
            return self.access_token
        except Exception as e:
            logging.error(f"Zerodha Auth Failed: {e}")
            return "AUTH_FAILED"

    async def logout(self) -> None:
        logging.info("Terminating API Session.")
        # Cleanup code here

    async def place_order(self, intent, algo_id: str) -> str:
        """
        Translates our AI's generic 'OrderIntent' into strict Zerodha API parameters.
        """
        if not self.kite: return "MOCK_ID"
        
        transaction_type = self.kite.TRANSACTION_TYPE_BUY if intent.side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
        
        try:
            order_id = self.kite.place_order(
                tradingsymbol=intent.symbol.replace(".NS", ""), # 'RELIANCE.NS' -> 'RELIANCE'
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type,
                quantity=intent.quantity,
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_MIS, # MIS = Intraday Margin (5x leverage), CNC = Delivery (No leverage)
                variety=self.kite.VARIETY_REGULAR,
                tag=algo_id # Tags the order so we know our AI placed it
            )
            logging.info(f"LIVE ORDER PLACED ON NSE! ID: {order_id}")
            return order_id
        except Exception as e:
            logging.error(f"Zerodha Order Failed: {e}")
            return None

    async def get_quote(self, symbol: str):
        """Fetches the live Level 1 quote for the asset."""
        if not self.kite: 
            return {"last": 100.0}
            
        try:
            clean_symbol = symbol.replace(".NS", "")
            instrument = f"NSE:{clean_symbol}"
            quote_data = self.kite.quote(instrument)
            last_price = quote_data[instrument]["last_price"]
            return {"last": last_price}
        except Exception as e:
            logging.error(f"Failed to fetch live quote: {e}")
            return {"last": 0.0}

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND CORE: ZERODHA API ADAPTER")
    print("=============================================")
    print("Adapter is ready. Awaiting API Keys.")
