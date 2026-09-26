import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HFTMicrostructure:
    """
    Phase 7 (Deep): High-Frequency Trading (HFT) Microstructure Models.
    These algorithms are used by proprietary trading firms to detect 
    insider activity and predict micro-second price movements.
    """
    
    @staticmethod
    def lee_ready_tick_rule(trades: list) -> list:
        """
        The Lee-Ready Algorithm.
        Exchanges don't tell us if a trade was a "Buy" or a "Sell", they just report the execution.
        This algorithm determines if a trade was initiated by an aggressive buyer or aggressive seller
        by comparing the execution price to the previous tick (up-tick vs down-tick).
        """
        classifications = []
        last_price = trades[0]['price']
        last_tick = 1 # 1 for Buy, -1 for Sell
        
        for trade in trades:
            price = trade['price']
            if price > last_price:
                last_tick = 1
            elif price < last_price:
                last_tick = -1
                
            classifications.append({
                'price': price,
                'volume': trade['volume'],
                'side': 'BUY' if last_tick == 1 else 'SELL'
            })
            last_price = price
            
        return classifications

    @staticmethod
    def calculate_vpin(trades: list, volume_bucket_size: int) -> float:
        """
        VPIN (Volume-Synchronized Probability of Informed Trading).
        Measures 'Order Flow Toxicity'. When insiders know a stock will crash, 
        they quietly dump shares. VPIN detects this hidden institutional dumping 
        before the price actually drops.
        """
        buy_volume = 0
        sell_volume = 0
        total_volume = 0
        
        for trade in trades:
            if trade['side'] == 'BUY':
                buy_volume += trade['volume']
            else:
                sell_volume += trade['volume']
                
            total_volume += trade['volume']
            
            # In a real HFT model, we calculate this over rolling volume buckets
            if total_volume >= volume_bucket_size:
                break
                
        if total_volume == 0:
            return 0.0
            
        # VPIN = Absolute difference between buy/sell volume divided by total volume
        vpin = abs(buy_volume - sell_volume) / total_volume
        return vpin

    @staticmethod
    def hawkes_process_intensity(trade_timestamps: list, current_time: float, mu=0.5, alpha=1.2, beta=2.0) -> float:
        """
        Hawkes Processes (Self-Exciting Point Processes).
        In HFT, trades arrive in "bursts". A massive buy order usually triggers a cascade 
        of algorithmic buy orders milliseconds later. 
        This models the 'Intensity' (probability) of the next trade happening right now.
        """
        intensity = mu # Baseline background trading rate
        for t in trade_timestamps:
            if t < current_time:
                # The impact of past trades decays exponentially over time (beta)
                intensity += alpha * np.exp(-beta * (current_time - t))
        return intensity

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: DEEP MICROSTRUCTURE (HFT)")
    print("=============================================")
    
    # Mocking a sequence of rapid-fire exchange executions (ticks)
    # Notice the price is slowly creeping down from 150.00
    mock_executions = [
        {'price': 150.00, 'volume': 100},
        {'price': 149.95, 'volume': 5000}, # Massive drop
        {'price': 149.95, 'volume': 2000}, # Same price, but follows a down-tick
        {'price': 149.90, 'volume': 8000},
        {'price': 149.92, 'volume': 100}   # Tiny retail buy
    ]
    
    hft = HFTMicrostructure()
    
    # 1. Run Lee-Ready to expose the aggressive sellers
    classified_flow = hft.lee_ready_tick_rule(mock_executions)
    
    # 2. Calculate VPIN (Toxicity)
    vpin_score = hft.calculate_vpin(classified_flow, volume_bucket_size=15000)
    
    print("\n[1] LEE-READY TICK CLASSIFICATION:")
    for trade in classified_flow:
        print(f"Price: {trade['price']} | Vol: {trade['volume']:<5} | Aggressor: {trade['side']}")
        
    print(f"\n[2] VPIN TOXICITY SCORE: {vpin_score:.3f} (Range 0.0 to 1.0)")
    if vpin_score > 0.7:
        print("CRITICAL TOXICITY: Informed Traders (Insiders) are aggressively dumping. MARKET MAKERS WILL PULL LIQUIDITY.")
    
    # 3. Hawkes Process Cascade Prediction
    # Trades happened at 0.1s, 0.2s, 0.25s, 0.3s. Current time is 0.31s
    mock_times = [0.1, 0.2, 0.25, 0.3]
    cascade_intensity = hft.hawkes_process_intensity(mock_times, current_time=0.31)
    
    print(f"\n[3] HAWKES CASCADE INTENSITY: {cascade_intensity:.2f}")
    if cascade_intensity > 3.0:
        print("ALGO CASCADE IMMINENT: Extreme probability of another massive trade in the next 100 milliseconds.")
