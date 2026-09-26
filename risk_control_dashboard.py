import logging
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class RiskControlDashboard:
    """
    Phase 3: Risk Control Dashboard.
    Splits the universe of stocks into High Risk (Volatile) and Low Risk (Stable).
    Routes them to the appropriate AI Brain (Brain A vs Brain B).
    """
    def __init__(self):
        # High Beta / Highly Volatile Stocks (Requires Intraday 5-min monitoring)
        self.high_risk_universe = ["ADANIENT.NS", "ZOMATO.NS", "PAYTM.NS", "YESBANK.NS"]
        
        # Low Beta / Stable Blue-Chips (Requires Daily Swing-Trade monitoring)
        self.low_risk_universe = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]
        
    def analyze_portfolio_risk(self, positions: dict):
        logging.info("Analyzing Portfolio Risk Exposure...")
        high_risk_exposure = 0
        low_risk_exposure = 0
        
        for ticker, value in positions.items():
            if ticker in self.high_risk_universe:
                high_risk_exposure += value
            else:
                low_risk_exposure += value
                
        total = high_risk_exposure + low_risk_exposure
        if total > 0:
            logging.info(f"High Risk Exposure: {(high_risk_exposure/total)*100:.1f}%")
            logging.info(f"Low Risk Exposure: {(low_risk_exposure/total)*100:.1f}%")
            
            if (high_risk_exposure/total) > 0.4:
                logging.warning("⚠️ RISK BREACH: High Risk exposure exceeds 40% maximum allowable threshold.")
        return True

    def route_to_brain(self, ticker: str) -> str:
        """Determines which AI Model should handle this specific stock."""
        if ticker in self.high_risk_universe:
            logging.info(f"Routing {ticker} to BRAIN_A (5-Minute Intraday PPO Model)")
            return "BRAIN_A_INTRADAY"
        else:
            logging.info(f"Routing {ticker} to BRAIN_B (Daily Swing-Trade PPO Model)")
            return "BRAIN_B_SWING"

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND CORE: RISK CONTROL DASHBOARD")
    print("=============================================")
    dashboard = RiskControlDashboard()
    dashboard.route_to_brain("ADANIENT.NS")
    dashboard.route_to_brain("RELIANCE.NS")
    
    mock_portfolio = {"ADANIENT.NS": 50000, "RELIANCE.NS": 20000, "HDFCBANK.NS": 30000}
    dashboard.analyze_portfolio_risk(mock_portfolio)
