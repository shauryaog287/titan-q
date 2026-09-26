import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_equity_curve(csv_path: str, output_path: str):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['Tick'], df['Equity'], color='cyan', linewidth=2, label="Portfolio Equity (₹)")
    
    # Fill under the curve
    plt.fill_between(df['Tick'], df['Equity'], df['Equity'].min(), color='cyan', alpha=0.1)

    plt.title('Titan-Q AI: Simulated Equity Curve (NIFTY 500 Data Lake)', fontsize=16, pad=15)
    plt.xlabel('Simulated Market Ticks', fontsize=12)
    plt.ylabel('Virtual Equity (₹)', fontsize=12)
    
    # Format Y axis with commas
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['₹{:,.0f}'.format(x) for x in current_values])
    
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Success! Equity curve plotted and saved to {output_path}")

if __name__ == "__main__":
    csv_file = "D:\\Quant_Project\\equity_curve_results.csv"
    png_file = "D:\\Quant_Project\\equity_curve_results.png"
    plot_equity_curve(csv_file, png_file)
