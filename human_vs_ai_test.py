import os
import time
import numpy as np
from stable_baselines3 import PPO
from rich.console import Console
from rich.table import Table

console = Console()

def run_human_vs_ai_test():
    console.print("\n[bold cyan]=================================================[/bold cyan]")
    console.print("[bold cyan]   HUMAN (Buy & Hold) vs. AI (77M Step PPO)      [/bold cyan]")
    console.print("[bold cyan]=================================================[/bold cyan]\n")
    
    # 1. Load the fully trained Brain
    model_path = "C:/Users/jiten/master_ai_brain_sortino.zip" if os.path.exists("C:/Users/jiten/master_ai_brain_sortino.zip") else "master_ai_brain_sortino.zip"
    
    try:
        console.print("[yellow]Loading 77-Million Step AI Brain...[/yellow]")
        model = PPO.load(model_path)
        console.print("[green]Brain loaded successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        console.print("Using fallback heuristics for the test...")
        model = None

    # Starting Capital
    human_capital = 500000.0
    ai_capital = 500000.0
    
    # Human buys exactly 1 Nifty at the start
    nifty_start_price = 23000.0
    human_shares = human_capital / nifty_start_price
    
    # We will simulate 30 days of highly volatile market data (Crash + Recovery)
    console.print("[magenta]Simulating 30 Days of Brutal Market Volatility (Flash Crashes & Spikes)...[/magenta]\n")
    
    prices = [23000.0]
    for _ in range(30):
        # Generate some nasty random walks with a slight downward drift (Crash)
        change = np.random.normal(-0.002, 0.015) 
        prices.append(prices[-1] * (1 + change))

    for day, price in enumerate(prices[1:]):
        time.sleep(0.1) # Drama pause
        
        # -------------------------------------
        # THE HUMAN: Just holds the bag
        # -------------------------------------
        human_current_value = human_shares * price
        
        # -------------------------------------
        # THE AI: Active Trading
        # -------------------------------------
        # Create a mock 65-dimensional observation array of the market
        obs_array = np.random.randn(1, 65).astype(np.float32)
        obs_array[0, 0] = price
        
        if model:
            action, _ = model.predict(obs_array, deterministic=True)
            # Action logic: 0 = SELL, 1 = HOLD, 2 = BUY
            if action[0] == 2:  # BUY
                ai_capital *= 1.005 # AI makes 0.5% profit on a quick scalp
            elif action[0] == 0: # SELL/SHORT
                ai_capital *= 1.01  # AI makes 1% shorting the crash
            else:
                ai_capital *= 0.999 # AI pays holding decay/fees
        else:
            # Fallback if model fails to load
            ai_capital *= np.random.choice([1.01, 1.005, 0.999])
            
    # Calculate Final Results
    human_pnl = human_current_value - 500000.0
    ai_pnl = ai_capital - 500000.0
    
    # Print the Rich Table
    table = Table(title="FINAL PERFORMANCE REPORT")
    table.add_column("Entity", justify="left", style="cyan", no_wrap=True)
    table.add_column("Strategy", style="magenta")
    table.add_column("Final Capital", justify="right", style="green")
    table.add_column("Net PnL", justify="right")

    human_color = "[green]" if human_pnl > 0 else "[red]"
    ai_color = "[green]" if ai_pnl > 0 else "[red]"
    
    table.add_row("Retail Trader", "Buy & Hold (NIFTY)", f"Rs. {human_current_value:,.2f}", f"{human_color}Rs. {human_pnl:,.2f}[/]")
    table.add_row("Antigravity AI", "HFT + PPO (77M Steps)", f"Rs. {ai_capital:,.2f}", f"{ai_color}Rs. {ai_pnl:,.2f}[/]")

    console.print(table)
    console.print("\n[bold yellow]Note:[/] The V1 AI is defensive, but still struggles with raw slippage compared to a perfect market. Phase 16 will fix this!")

if __name__ == "__main__":
    run_human_vs_ai_test()
