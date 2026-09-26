import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("==========================================")
print("THE SCOUT: MARKET SCREENER INITIALIZED")
print("==========================================")

tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "TATAMOTORS.NS", 
    "ZOMATO.NS", "PAYTM.NS", "SUZLON.NS", "IRFC.NS", "ITC.NS", "SBIN.NS"
]

print(f"Scanning {len(tickers)} stocks for today's trading opportunities...\n")

results = []

for ticker in tickers:
    try:
        data = yf.download(ticker, period="60d", progress=False)
        if data.empty:
            continue
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
            
        current_price = data['Close'].iloc[-1]
        daily_volatility = ((data['High'] - data['Low']) / data['Close']).mean() * 100
        past_price = data['Close'].iloc[-20]
        trend_percent = ((current_price - past_price) / past_price) * 100
        
        results.append({
            "Ticker": ticker.replace(".NS", ""),
            "Price (Rs)": round(current_price, 2),
            "Volatility (%)": round(daily_volatility, 2),
            "Trend (20d)": round(trend_percent, 2)
        })
    except Exception as e:
        pass

df_results = pd.DataFrame(results)

print("\n[HIGH RISK / DAY TRADING TARGETS] (Most Volatile):")
print("The AI will use 5-minute charts to aggressively day-trade these.")
high_risk = df_results.sort_values(by="Volatility (%)", ascending=False).head(3)
print(high_risk.to_string(index=False))

print("\n[LOW RISK / SWING TRADING TARGETS] (Stable Uptrend):")
print("The AI will use Daily charts to slowly hold these for safe profit.")
low_risk = df_results[df_results["Trend (20d)"] > 0].sort_values(by="Volatility (%)", ascending=True).head(3)
print(low_risk.to_string(index=False))
print("\n==========================================")
