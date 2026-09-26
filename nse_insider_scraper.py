import requests
import time
import random
import warnings
warnings.filterwarnings("ignore")

print("==========================================")
print("STEALTH SCRAPER: INSIDER DEALS INITIALIZED")
print("==========================================")
print("1. Masking IP and generating human-like Chrome browser headers...")

# The disguise
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

session = requests.Session()
session.headers.update(headers)

print("2. Navigating to NSE homepage to steal session cookies...")
try:
    # Step 1: Visit homepage to get cookies (bypasses basic firewalls)
    homepage_response = session.get("https://www.nseindia.com", timeout=15)
    
    # Wait like a human
    sleep_time = random.uniform(1.5, 3.5)
    print(f"   -> Cookies secured. Pausing for {sleep_time:.1f} seconds to mimic human reading...")
    time.sleep(sleep_time)
    
    print("3. Breaching Bulk Deals API...")
    # Step 2: Hit the hidden API for bulk deals
    api_url = "https://www.nseindia.com/api/snapshot-capital-market-bulk-deals"
    response = session.get(api_url, timeout=15)
    
    if response.status_code == 200:
        data = response.json()
        print("\n🚨 INSIDER ACTIVITY DETECTED TODAY:")
        print("------------------------------------------------")
        
        if 'data' in data and len(data['data']) > 0:
            for deal in data['data'][:7]: # Show top 7 deals
                symbol = deal.get('symbol', 'UNKNOWN')
                client = deal.get('clientName', 'UNKNOWN')
                action = deal.get('buyOrSell', 'UNKNOWN')
                qty = deal.get('quantity', 0)
                
                # Format nicely
                if action == "BUY":
                    print(f"[BUY]  {symbol} : {client} bought {qty} shares")
                else:
                    print(f"[SELL] {symbol} : {client} sold {qty} shares")
        else:
            print("No massive bulk deals reported by NSE yet today.")
    else:
        print(f"\n[!] NSE Firewall blocked the request (Status Code: {response.status_code}).")
        print("To bypass this permanently in Phase 5, we will add Rotating Residential Proxies.")
        
except Exception as e:
    print(f"\n[X] Scraper encountered network/firewall error: {e}")

print("\n==========================================")
