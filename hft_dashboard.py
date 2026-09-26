import os
import json
import time
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

def generate_dashboard():
    """Generates the Rich layout for the current tick."""
    layout = Layout()
    
    # Read live state
    state_file = "C:/Users/jiten/live_state.json"
    price = 0.0
    symbol = "WAITING FOR DATA..."
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                price = data.get("latest_price", 0.0)
                symbol = data.get("symbol", "N/A")
        except:
            pass
            
    # Top Panel: Market Overview
    market_text = Text()
    market_text.append(f"SYMBOL: ", style="bold cyan")
    market_text.append(f"{symbol}\n", style="bold white")
    market_text.append(f"LAST PRICE: ", style="bold cyan")
    market_text.append(f"\u20b9{price:,.2f}\n", style="bold green" if price > 0 else "bold red")
    market_panel = Panel(Align.center(market_text, vertical="middle"), title="[bold yellow]LIVE MARKET FEED", border_style="yellow")
    
    # Middle Panel: Models (Mocked for dashboard display, in reality would read from loop)
    model_table = Table(show_header=True, header_style="bold magenta", expand=True)
    model_table.add_column("AI Engine")
    model_table.add_column("Status")
    model_table.add_column("Output")
    
    model_table.add_row("PPO Core Brain", "[green]ONLINE", "[blue]HOLD")
    model_table.add_row("Hawkes Toxicity", "[green]ARMED", f"Score: 0.12 [green](SAFE)")
    model_table.add_row("HMM Regime", "[green]TRACKING", "[yellow]BULL REGIME")
    model_table.add_row("Copula Stat-Arb", "[green]SCANNING", "No Pairs Found")
    
    model_panel = Panel(model_table, title="[bold cyan]QUANTITATIVE MODELS", border_style="cyan")
    
    # Bottom Panel: Portfolio
    port_text = Text("Virtual Capital: \u20b9500,000.00\nActive Positions: 0\nUnrealized PnL: \u20b90.00", style="bold green", justify="center")
    port_panel = Panel(Align.center(port_text, vertical="middle"), title="[bold green]PAPER TRADING BROKER", border_style="green")
    
    layout.split_column(
        Layout(market_panel, size=6),
        Layout(model_panel, size=10),
        Layout(port_panel, size=6)
    )
    return layout

if __name__ == "__main__":
    print("Initializing Antigravity Quant Dashboard...")
    time.sleep(1)
    
    try:
        with Live(generate_dashboard(), refresh_per_second=4, screen=True) as live:
            while True:
                live.update(generate_dashboard())
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("Dashboard closed.")
