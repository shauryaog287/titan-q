import asyncio
import json
import websockets
import logging
from quant_data_lake import QuantDataLake

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [LIVE FEED] %(message)s')

async def live_market_feed(lake: QuantDataLake):
    """
    Connects to a live WebSocket (Binance BTC/USDT as a high-volatility proxy)
    to stream real-world millisecond tick data into our Data Lake.
    No API keys required. Perfect for feeding the Hawkes Toxicity Engine.
    """
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    
    logging.info(f"Connecting to Live Market Feed: {uri}")
    
    # We maintain a shared state file so the Dashboard and Main Loop can read the live price
    state_file = "C:/Users/jiten/live_state.json"
    
    async with websockets.connect(uri) as websocket:
        logging.info("Connection established. Receiving high-frequency ticks...")
        
        tick_count = 0
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Binance trade payload: p=price, q=quantity, m=is_buyer_maker
                price = float(data['p'])
                volume = float(data['q'])
                is_buyer_maker = data['m']
                
                # 1. Insert into Data Lake (Option B)
                lake.insert_tick("BTC-USDT", price, volume, is_buyer_maker)
                
                # 2. Update live state for the Master Loop & Dashboard (Option C)
                with open(state_file, "w") as f:
                    json.dump({
                        "symbol": "BTC-USDT",
                        "latest_price": price,
                        "latest_volume": volume,
                        "is_maker": is_buyer_maker
                    }, f)
                
                tick_count += 1
                if tick_count % 100 == 0:
                    logging.info(f"Processed {tick_count} live ticks. Current Price: ${price:,.2f}")
                    
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await asyncio.sleep(2)
                
async def main():
    lake = QuantDataLake()
    await live_market_feed(lake)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Live feed terminated.")
