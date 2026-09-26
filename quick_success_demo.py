import time
from rich.console import Console
from rich.table import Table

console = Console()

def run_fake_success_scenario():
    console.print("\n[bold yellow]=================================================[/bold yellow]")
    console.print("[bold green]  INITIATING FAKE SCENARIO: 'THE PERFECT STORM'  [/bold green]")
    console.print("[bold yellow]=================================================[/bold yellow]\n")
    
    console.print("[cyan]Setting Environment Variables...[/cyan]")
    console.print(" -> Exchange Fees: [bold green]Rs. 0.00 (Bypassed)[/bold green]")
    console.print(" -> Slippage: [bold green]0.0% (Perfect Liquidity)[/bold green]")
    console.print(" -> Market Condition: [bold green]Clean Mean-Reversion Channel[/bold green]\n")
    
    capital = 500000.0
    price = 100.0
    shares = 0
    
    # We simulate a perfect sine-wave market (easy for any math engine to exploit)
    # The AI buys the dips and shorts the peaks
    
    console.print("[bold magenta]Starting High-Frequency Execution...[/bold magenta]\n")
    
    # 5 Ticks of Perfect Trading
    scenario = [
        {"tick": 1, "action": "BUY",  "price": 95.0,  "reason": "Copula detects oversold lower bound"},
        {"tick": 2, "action": "HOLD", "price": 100.0, "reason": "Riding momentum up"},
        {"tick": 3, "action": "SELL", "price": 110.0, "reason": "HMM detects peak. Liquidating & Shorting"},
        {"tick": 4, "action": "HOLD", "price": 98.0,  "reason": "Riding short momentum down"},
        {"tick": 5, "action": "BUY",  "price": 90.0,  "reason": "Covering short. Massive profit locked in."}
    ]
    
    for step in scenario:
        time.sleep(1) # Fake processing delay
        console.print(f"[white]Tick {step['tick']} | Price: Rs. {step['price']:,.2f}[/white]")
        
        if step['action'] == "BUY":
            if shares < 0: # Covering short
                profit = (price - step['price']) * abs(shares)
                capital += profit
                shares = 0
                console.print(f"  -> [bold green]COVER SHORT[/bold green] | Reason: {step['reason']}")
                console.print(f"  -> [bold yellow]Profit Locked: +Rs. {profit:,.2f}[/bold yellow]")
            else:
                shares_to_buy = int(capital / step['price'])
                shares += shares_to_buy
                capital -= shares_to_buy * step['price']
                console.print(f"  -> [bold cyan]BUY {shares_to_buy} SHARES[/bold cyan] | Reason: {step['reason']}")
                
        elif step['action'] == "SELL":
            # Liquidate longs
            if shares > 0:
                capital += shares * step['price']
                shares = 0
            # Enter Short
            shares = -5000 # Borrow 5000 shares
            console.print(f"  -> [bold red]SELL / SHORT 5000 SHARES[/bold red] | Reason: {step['reason']}")
            
        elif step['action'] == "HOLD":
            console.print(f"  -> [bold white]HOLD POSITION[/bold white] | Reason: {step['reason']}")
            
        price = step['price']
        
    console.print("\n[bold green]=================================================[/bold green]")
    table = Table(title="QUICK SUCCESS RESULTS")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Starting Capital", "Rs. 500,000.00")
    table.add_row("Ending Capital", f"Rs. {capital:,.2f}")
    table.add_row("Net Profit", f"+Rs. {(capital - 500000.0):,.2f}")
    table.add_row("Win Rate", "100.0%")
    
    console.print(table)
    console.print("[bold green]=================================================[/bold green]")

if __name__ == "__main__":
    run_fake_success_scenario()
