import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class OrderBookMicrostructure:
    """
    Phase 7: Market Microstructure (Level 2 Data)
    Analyzes the Exchange Limit Order Book (LOB) to detect institutional 
    spoofing, iceberg orders, and high-frequency liquidity imbalances.
    """
    
    @staticmethod
    def calculate_imbalance(bids: list, asks: list, depth: int = 5) -> float:
        """
        Calculates the Volume Order Imbalance (VOI) at the top N levels.
        bids/asks format: [(price, volume), (price, volume), ...]
        Returns a score from -1.0 (Heavy Ask Pressure) to 1.0 (Heavy Bid Pressure)
        """
        bid_vol = sum(vol for price, vol in bids[:depth])
        ask_vol = sum(vol for price, vol in asks[:depth])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
    @staticmethod
    def calculate_microprice(best_bid: tuple, best_ask: tuple) -> float:
        """
        Calculates the volume-weighted mid-price.
        If there are 10,000 shares on the bid and 10 shares on the ask, 
        the true price is much closer to the ask (imminent upward sweep).
        """
        bid_price, bid_vol = best_bid
        ask_price, ask_vol = best_ask
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (bid_price + ask_price) / 2
            
        # Cross-weighted: Large bid volume pulls microprice closer to ask price
        microprice = (bid_price * ask_vol + ask_price * bid_vol) / total_vol
        return microprice
        
    @staticmethod
    def detect_spoofing(bids: list, asks: list) -> str:
        """
        Scans deep into the order book (levels 3-10) for massive limit orders
        that are disproportionately larger than the Top of Book (levels 1-2).
        Institutions place these to manipulate retail bots, then cancel them.
        """
        top_bid_vol = sum(vol for p, vol in bids[:2])
        deep_bid_vol = sum(vol for p, vol in bids[2:10])
        
        top_ask_vol = sum(vol for p, vol in asks[:2])
        deep_ask_vol = sum(vol for p, vol in asks[2:10])
        
        # If the deep book has 10x the volume of the top book, it's a spoof wall
        if deep_bid_vol > top_bid_vol * 10 and deep_bid_vol > 10000:
            return "SPOOFING DETECTED: Fake Buy Wall (Bearish trap)"
        elif deep_ask_vol > top_ask_vol * 10 and deep_ask_vol > 10000:
            return "SPOOFING DETECTED: Fake Sell Wall (Bullish trap)"
        
        return "CLEAN"

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: LEVEL 2 ORDER BOOK")
    print("=============================================")
    
    # Mocking an institutional spoofing scenario
    # A massive fake buy wall is placed at 99.50 to trick retail into buying,
    # but the institution actually wants to sell at 100.05.
    mock_bids = [
        (100.00, 500), (99.95, 400),   # Top of book (Normal Retail)
        (99.50, 50000), (99.40, 20000) # Deep book (Institutional Spoof)
    ]
    
    mock_asks = [
        (100.05, 600), (100.10, 450), 
        (100.20, 800), (100.30, 900)
    ]
    
    analyzer = OrderBookMicrostructure()
    imb = analyzer.calculate_imbalance(mock_bids, mock_asks, depth=4)
    micro = analyzer.calculate_microprice(mock_bids[0], mock_asks[0])
    spoof = analyzer.detect_spoofing(mock_bids, mock_asks)
    
    print(f"Top Bid: {mock_bids[0][0]} | Top Ask: {mock_asks[0][0]}")
    print(f"Standard Mid-Price: {(mock_bids[0][0] + mock_asks[0][0]) / 2:.2f}")
    print(f"Micro-Price (True Value): {micro:.2f}")
    print(f"Order Imbalance Ratio: {imb:.2f} (+ is Bullish, - is Bearish)")
    print(f"Institutional Algorithm Status: {spoof}")
