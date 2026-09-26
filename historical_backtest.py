"""
historical_backtest.py
======================
PIPELINE SMOKE TEST -- NOT A PERFORMANCE ESTIMATE.

This script checks that the modules import and run together end to end. It
reads no historical data (every input is random), the models are untrained,
and each trade's PnL is a random draw with a hard-coded positive mean (see the
pnl_pct lines below). Its win rate and Sharpe ratio are therefore set by those
constants, not by any model: win rate ~= Phi(mean/sd), Sharpe ~= mean/sd times
an annualisation factor. See Titan-Q_Whitepaper.md, Section 6.

Do not quote its numbers as results. The real, walk-forward backtest is
backtest_daily.py.
"""

import os
import pandas as pd
import numpy as np
import time
import logging
import torch

from fusion_and_sizing import KellySizer, CostModel
from portfolio_hrp_allocator import HRPAllocator
from transformer_lob_forecaster import TemporalFusionLOB
from quantum_portfolio_annealer import anneal_swap
from neural_sde_discovery import NeuralSDE
from transfer_entropy_causality import transfer_entropy, discretize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTESTER] - %(message)s')

DATA_LAKE_PATH = "G:\\ My Drive\\NIFTY_Data"

def run_historical_backtest(start_date: str, end_date: str, initial_capital: float = 500000.0):
    logging.info(f"Initializing Titan-Q Offline Backtest Engine...")
    logging.info(f"Target Window: {start_date} to {end_date}")
    
    # Check if offline data exists, otherwise fallback to synthetic historical walk
    has_real_data = os.path.exists(DATA_LAKE_PATH)
    if not has_real_data:
        logging.warning(f"Real data lake not found at {DATA_LAKE_PATH}. Generating rigorous synthetic historical dataset for backtest...")
        n_ticks = 2000
    else:
        logging.info(f"Connected to local data lake at {DATA_LAKE_PATH}.")
        n_ticks = 2000 # Fast check
        
    # ---------------------------------------------------------
    # 1. BOOT THE BRAIN (Load all Phase 1-25 modules)
    # ---------------------------------------------------------
    logging.info("Waking up AI modules for offline evaluation...")
    
    sizer = KellySizer(max_leverage=1.5, kelly_fraction=0.5)
    hrp = HRPAllocator()
    
    lob_transformer = TemporalFusionLOB(n_levels=10, window=50)
    lob_transformer.eval()
    
    sde_model = NeuralSDE(hidden=32, depth=2)
    sde_model.eval()
    
    logging.info("All engines armed. Commencing Historical Walk.")
    
    # ---------------------------------------------------------
    # 2. THE HISTORICAL WALK
    # ---------------------------------------------------------
    capital = initial_capital
    equity_curve = [capital]
    
    start_time = time.time()
    
    for tick in range(1, n_ticks + 1):
        if tick % 1000 == 0:
            logging.info(f"Processed {tick}/{n_ticks} ticks. Current Equity: Rs. {capital:,.2f}")
            
        # --- Simulate reading a row from the historical DataFrame ---
        # Feature: LOB window (Phase 18)
        lob_window = torch.randn(1, 50, 4, 10, dtype=torch.float32)
        
        # Feature: Transfer Entropy Causality (Phase 24)
        options_vol = np.random.randn(50)
        underlying_px = 0.8 * np.roll(options_vol, 1) + np.random.randn(50) * 0.2
        te_bits = transfer_entropy(discretize(options_vol, 8), discretize(underlying_px, 8), k=1, l=1, lag=1)
        
        # --- Model Inferences ---
        with torch.no_grad():
            # Predict direction
            forecast_logits = lob_transformer(lob_window)
            forecast_probs = torch.softmax(forecast_logits, dim=-1)[0].numpy()
            
            # Discover Drift/Diffusion for this exact historical tick
            idx_state = torch.tensor([[0.5]], dtype=torch.float64)
            dyn_mu = sde_model.mu(idx_state).item() * 252.0
            dyn_sigma = sde_model.sigma(idx_state).item() * np.sqrt(252.0)
            
        # --- Strategy Logic ---
        # If AI predicts strong upside (prob > 0.4) and TE causality is high
        if forecast_probs[2] > 0.4 and te_bits > 1.0:
            costs = CostModel(impact_coefficient=0.75, half_spread=0.0004)
            try:
                res = sizer.size(
                    mu_hat=max(0.01, dyn_mu),
                    sigma_annual=max(0.05, dyn_sigma),
                    sample_years=10.0,
                    equity=capital,
                    adv_notional=850_000_000.0,
                    daily_vol=dyn_sigma / np.sqrt(252),
                    cost_model=costs
                )
                fraction = res.f_final
            except:
                fraction = 0.1
                
            # Simulate EVT Tail Risk (Phase 23) overriding the size
            evt_mock_es = np.random.uniform(0.01, 0.15)
            if evt_mock_es > 0.12:
                fraction *= 0.5
                
            trade_amount = capital * fraction
            
            # Simulate a historical PnL outcome based on rough drift
            pnl_pct = np.random.normal(0.0005, 0.002) # 5 bps mean return on winning trades
            capital += trade_amount * pnl_pct
            
        elif forecast_probs[0] > 0.4 and te_bits > 1.0:
            # Short signal logic (similar but inverted)
            fraction = 0.05
            trade_amount = capital * fraction
            pnl_pct = np.random.normal(0.0002, 0.002) 
            capital += trade_amount * pnl_pct
            
        equity_curve.append(capital)
        
    end_time = time.time()
    
    # ---------------------------------------------------------
    # 3. TEAR SHEET / RESULTS
    # ---------------------------------------------------------
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_return = (capital - initial_capital) / initial_capital
    sharpe = np.sqrt(252 * 75) * (returns.mean() / (returns.std() + 1e-9)) # approx 75 ticks/day
    
    # Calculate trade accuracy
    active_trades = returns[returns != 0]
    win_rate = (active_trades > 0).mean() if len(active_trades) > 0 else 0.0
    
    logging.info("==================================================")
    logging.info("SMOKE TEST COMPLETE (random PnL -- NOT a performance result)")
    logging.info("==================================================")
    logging.info(f"Time Elapsed:   {end_time - start_time:.2f} seconds")
    logging.info(f"Ticks Replayed: {n_ticks:,}")
    logging.info(f"Starting Eq:    Rs. {initial_capital:,.2f}")
    logging.info(f"Ending Eq:      Rs. {capital:,.2f}")
    logging.info(f"Total Return:   {total_return:.2%}")
    logging.info(f"Est. Sharpe:    {sharpe:.2f}")
    logging.info(f"Trade Accuracy: {win_rate:.2%} (Win Rate)")
    logging.info("==================================================")
    
    # Save the equity curve for plotting
    df_results = pd.DataFrame({
        "Tick": range(len(equity_curve)),
        "Equity": equity_curve
    })
    df_results.to_csv("D:\\Quant_Project\\equity_curve_results.csv", index=False)
    logging.info("Saved equity curve to D:\\Quant_Project\\equity_curve_results.csv")
    
if __name__ == "__main__":
    # Standard 5-year offline backtest run
    run_historical_backtest(start_date="2020-01-01", end_date="2025-01-01")
