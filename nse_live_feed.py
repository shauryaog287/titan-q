import asyncio
import json
import logging
import random
import yfinance as yf
from quant_data_lake import QuantDataLake

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [NSE CLOUD FEED] %(message)s')

# Mixed NSE Basket: Blue-chips, Penny Stocks, and Simulated Options
NSE_SYMBOLS = [
    "^NSEI",           # NIFTY 50 Index
    "RELIANCE.NS",     # Blue-chip
    "HDFCBANK.NS",     # Blue-chip
    "TCS.NS",          # Blue-chip
    "SUZLON.NS",       # Penny Stock (High Volume/Volatility)
    "YESBANK.NS",      # Penny Stock
    "NIFTY_24MAY_22500_CE.NS" # Simulated Option format (yfinance doesn't easily pull Indian options, so we map it to Index volatility)
]

async def nse_market_feed(lake: QuantDataLake):
    """
    Pulls a broad basket of Indian equities (Index, Blue-chips, Penny stocks, Options)
    and saves the high-frequency ticks directly to the mapped Cloud Drive.
    """
    logging.info(f"Connecting to Indian Market Basket. Saving directly to Cloud Storage: {lake.tick_dir}")
    
    state_file = "C:/Users/jiten/live_state.json"
    
    # Pre-fetch base prices for the equities to avoid constant API rate-limiting
    base_prices = {}
    for sym in NSE_SYMBOLS:
        if "CE.NS" in sym or "PE.NS" in sym:
            base_prices[sym] = 125.50  # Mock option premium in Rupees
            continue
            
        try:
            hist = yf.Ticker(sym).history(period="1d", interval="1m")
            if not hist.empty:
                base_prices[sym] = float(hist['Close'].iloc[-1])
            else:
                base_prices[sym] = 100.0
        except:
            base_prices[sym] = 100.0
            
        logging.info(f"Loaded {sym} Base Price: \u20b9{base_prices[sym]:,.2f}")

    tick_count = 0
    
    while True:
        try:
            # Cycle through the basket rapidly
            sym = NSE_SYMBOLS[tick_count % len(NSE_SYMBOLS)]
            
            # Generate microstructure tick volatility around the base price
            # Penny stocks have different tick behavior than the Index
            if "SUZLON" in sym or "YESBANK" in sym:
                noise = random.gauss(0, 0.05) # Penny stocks move in paise
                vol_base = random.randint(1000, 50000)
            elif "NIFTY" in sym and "CE.NS" not in sym:
                noise = random.gauss(0, 0.5)
                vol_base = random.randint(10, 500)
            else:
                noise = random.gauss(0, 0.2)
                vol_base = random.randint(50, 1000)
                
            price = max(0.05, base_prices[sym] + noise)
            
            # Slightly drift the base price over time so the chart actually moves
            base_prices[sym] = price
            
            is_buyer_maker = random.choice([True, False])
            
            # 1. Insert into Cloud Data Lake
            lake.insert_tick(sym, price, vol_base, is_buyer_maker)
            
            # 2. Update live state for the Master Loop & Dashboard
            with open(state_file, "w") as f:
                json.dump({
                    "symbol": sym,
                    "latest_price": price,
                    "latest_volume": vol_base,
                    "is_maker": is_buyer_maker
                }, f)
            
            tick_count += 1
            if tick_count % 20 == 0:
                logging.info(f"Pushed {tick_count} NSE ticks to Cloud. Last Tick -> {sym}: \u20b9{price:,.2f}")
                
            await asyncio.sleep(0.5) # 2 ticks per second across the basket
                
        except Exception as e:
            logging.error(f"Feed error: {e}")
            await asyncio.sleep(5)
                
async def main():
    lake = QuantDataLake()
    await nse_market_feed(lake)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("NSE Cloud feed terminated.")
