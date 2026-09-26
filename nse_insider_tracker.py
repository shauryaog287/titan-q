import requests
import json
import logging
import time
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [INSIDER TRACKER] %(message)s')

class NSEInsiderTracker:
    """
    Legally fetches public 'Prohibition of Insider Trading' (PIT) and 
    Promoter buying disclosures from the official NSE India website.
    """
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.api_url = "https://www.nseindia.com/api/corporates-pit?index=equities"
        self.session = requests.Session()
        # Modern websites require standard browser headers to process requests
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })

    def establish_session(self):
        """Hit the homepage first to acquire necessary routing cookies."""
        logging.info("Establishing legal session with NSE India...")
        try:
            self.session.get(self.base_url, timeout=10)
            time.sleep(2) # Be polite to the server
            return True
        except Exception as e:
            logging.error(f"Failed to connect to NSE: {e}")
            return False

    def fetch_insider_trades(self):
        """Fetch the latest JSON payload of insider buying/selling."""
        if not self.establish_session():
            return None
            
        logging.info("Requesting latest Promoter/Insider Disclosures (PIT)...")
        try:
            response = self.session.get(self.api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self.parse_data(data)
            else:
                logging.warning(f"NSE returned status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch API data: {e}")
            return None

    def parse_data(self, raw_json):
        """Filter the raw data for massive promoter BUY orders."""
        if 'data' not in raw_json:
            return None
            
        df = pd.DataFrame(raw_json['data'])
        if df.empty:
            return None
            
        # Filter for actual Promoters buying stock (Acquisition)
        try:
            # Column names on NSE API can vary, but generally contain 'personCategory' and 'secAcq'
            promoters = df[df['personCategory'].str.contains("Promoter", case=False, na=False)]
            buys = promoters[promoters['secAcq'] > 0] # secAcq = Securities Acquired
            
            if not buys.empty:
                logging.info(f"Detected {len(buys)} recent Insider Buy transactions!")
                for _, row in buys.head(5).iterrows():
                    symbol = row.get('symbol', 'UNKNOWN')
                    person = row.get('acqName', 'UNKNOWN')
                    shares = row.get('secAcq', 0)
                    logging.info(f"ALPHA ALERT: Promoter '{person}' just bought {shares} shares of {symbol}!")
            else:
                logging.info("No massive promoter acquisitions detected today.")
                
            return buys
        except Exception as e:
            logging.error(f"Error parsing dataframe: {e}")
            return df

if __name__ == "__main__":
    tracker = NSEInsiderTracker()
    tracker.fetch_insider_trades()
