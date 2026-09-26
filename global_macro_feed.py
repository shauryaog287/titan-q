import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("==========================================")
print("[PHASE 6: GLOBAL MACRO WEATHER RADAR]")
print("==========================================")
print("Fetching live data for the 3 Pillars of Global Finance...")

# Tickers:
# CL=F : Crude Oil Futures
# INR=X : USD/INR Exchange Rate
# ^TNX : US 10-Year Treasury Yield
tickers = {"Crude Oil": "CL=F", "USD/INR": "INR=X", "US 10Y Bond": "^TNX"}

macro_data = {}
storm_score = 0 # 0 means sunny, higher means stormy (bearish for Indian equities)

for name, symbol in tickers.items():
    try:
        data = yf.download(symbol, period="5d", progress=False)
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
            
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        pct_change = ((current_price - previous_price) / previous_price) * 100
        
        macro_data[name] = {"price": current_price, "change": pct_change}
        
        # If Oil goes up, bad for India (imports oil)
        # If USD goes up, bad for Rupee
        # If US yields go up, bad for emerging markets (FIIs pull money out)
        if pct_change > 0:
            storm_score += 1
            
    except Exception as e:
        pass

print("\n[LIVE MACRO READINGS]")
print("------------------------------------------")
for name, stats in macro_data.items():
    trend = "[+] UP" if stats["change"] > 0 else "[-] DOWN"
    print(f"{name}: {stats['price']:.2f} ({stats['change']:+.2f}%) -> {trend}")

print("\n[AI WEATHER ANALYSIS FOR INDIAN MARKETS]")
print("------------------------------------------")
if storm_score == 3:
    print("[CRITICAL] RED ALERT: All three macro pillars are rising.")
    print("Foreign money is likely leaving India. Tell the Trading AI to go heavily into SHORT/LOW RISK mode.")
elif storm_score == 2:
    print("[WARNING] YELLOW ALERT: Global pressure is increasing.")
    print("Market might be volatile. Tell the Trading AI to be cautious.")
elif storm_score == 1:
    print("[CLEAR] GREEN: Macro conditions are relatively stable.")
    print("Tell the Trading AI to proceed with standard operations.")
else:
    print("[OPTIMAL] PERFECT WEATHER: Oil, USD, and US Yields are all falling.")
    print("This is incredibly bullish for Indian equities. Tell the Trading AI to go MAX RISK LONG.")

print("==========================================")
