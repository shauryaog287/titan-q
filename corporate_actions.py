import yfinance as yf
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_corporate_actions(ticker_symbol: str):
    """
    Phase 4: Structural Alpha (Corporate Actions).
    Extracts historical dividends and stock splits. 
    This structural data is normalized and fed into the AI's observation space
    because sudden price drops due to a 2:1 stock split will confuse the AI 
    if it doesn't know a corporate action occurred.
    """
    logging.info(f"Extracting Corporate Actions for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    
    try:
        dividends = stock.dividends
        splits = stock.splits
        
        actions = []
        if not dividends.empty:
            logging.info(f"Found {len(dividends)} historical dividend payouts.")
            
        if not splits.empty:
            logging.info(f"Found {len(splits)} historical stock splits.")
            for date, ratio in splits.items():
                logging.info(f"Split on {date.date()}: Ratio {ratio}:1")
                
        # Save to Data Lake
        os.makedirs("C:/Users/jiten/QuantData/Corporate", exist_ok=True)
        if not dividends.empty:
            dividends.to_csv(f"C:/Users/jiten/QuantData/Corporate/{ticker_symbol}_dividends.csv")
        if not splits.empty:
            splits.to_csv(f"C:/Users/jiten/QuantData/Corporate/{ticker_symbol}_splits.csv")
            
        logging.info("Corporate actions successfully saved to Data Lake.")
            
    except Exception as e:
        logging.error(f"Failed to extract corporate actions: {e}")

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: CORPORATE ACTIONS")
    print("=============================================")
    extract_corporate_actions("RELIANCE.NS")
