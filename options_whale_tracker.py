import yfinance as yf
import pandas as pd
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_options_chain(ticker_symbol):
    """
    Phase 6: Options Tracking (The Whales)
    Pulls Open Interest (OI) and Put-Call Ratios to detect institutional bets.
    """
    logging.info(f"Connecting to Exchange for {ticker_symbol} Options Data...")
    stock = yf.Ticker(ticker_symbol)
    
    try:
        # Get all available expiration dates
        expirations = stock.options
        if not expirations:
            logging.warning(f"No options data available for {ticker_symbol}.")
            return
            
        logging.info(f"Found {len(expirations)} expiration dates. Pulling near-term chain...")
        
        # Pull the closest expiration
        opt = stock.option_chain(expirations[0])
        calls = opt.calls
        puts = opt.puts
        
        # Calculate PCR (Put-Call Ratio) by Volume and Open Interest
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        logging.info(f"[{ticker_symbol}] Put-Call Ratio (Volume): {pcr_vol:.2f}")
        logging.info(f"[{ticker_symbol}] Put-Call Ratio (Open Interest): {pcr_oi:.2f}")
        
        if pcr_oi > 1.2:
            logging.info("[!] WHALE ALERT: Extreme Put buying detected (Bearish/Hedging)")
        elif pcr_oi < 0.7:
            logging.info("[!] WHALE ALERT: Heavy Call buying detected (Bullish Breakout expected)")
            
        # Save to Data Lake
        os.makedirs("C:/Users/jiten/QuantData/Options", exist_ok=True)
        filename = f"C:/Users/jiten/QuantData/Options/{ticker_symbol}_chain_{datetime.now().strftime('%Y%m%d')}.csv"
        
        calls['type'] = 'Call'
        puts['type'] = 'Put'
        chain = pd.concat([calls, puts])
        chain.to_csv(filename, index=False)
        logging.info(f"Saved complete options chain to {filename}")
        
    except Exception as e:
        logging.error(f"Failed to pull options for {ticker_symbol}: {e}")

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: OPTIONS & WHALE TRACKER")
    print("=============================================")
    
    # We use SPY (S&P 500) to demonstrate since Yahoo Finance lacks NSE Options chains
    download_options_chain("SPY")
