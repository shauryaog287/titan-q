import os
import glob
from datetime import datetime

def build_claude_audit():
    """
    Crawls the entire Quant Infrastructure and bundles it into a single highly-structured 
    markdown file. You can upload this single file to Claude Opus 5 to perform a God-Tier 
    architectural code review and error hunt.
    """
    directory = "C:/Users/jiten"
    output_file = os.path.join(directory, "CLAUDE_MASTER_AUDIT_BUNDLE.txt")
    
    # Grab all python phase scripts and infrastructure modules
    py_files = glob.glob(os.path.join(directory, "*.py"))
    
    # Filter out anything irrelevant or temporary
    quant_files = [f for f in py_files if "env" not in f and "scratch" not in f]
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("========================================================================\n")
        out.write("SYSTEM DIRECTIVE: ELITE HFT CODE AUDITOR\n")
        out.write("========================================================================\n")
        out.write("You are an L10 Quantitative Developer and PyTorch Architect.\n")
        out.write("Attached is the entire source code for a 2026-era HFT system.\n")
        out.write("It includes PPO RL, Hawkes Toxicity, HMM Regime Filters, Copulas, and PINNs.\n")
        out.write("Your task is to scan every line of code for the following errors:\n")
        out.write("1. PyTorch memory leaks (un-detached tensors in loops).\n")
        out.write("2. Mathematical calculation errors in the stochastic calculus implementations.\n")
        out.write("3. Multi-threading / Asyncio race conditions in the execution engine.\n")
        out.write("4. Data-type collisions (Float32 vs Float64) in model predictions.\n\n")
        out.write("Do not summarize the code. Find the bugs and provide the corrected code blocks.\n")
        out.write("========================================================================\n\n")
        
        for file_path in quant_files:
            file_name = os.path.basename(file_path)
            out.write(f"\n\n{'='*80}\n")
            out.write(f"FILE: {file_name}\n")
            out.write(f"{'='*80}\n\n")
            
            with open(file_path, "r", encoding="utf-8") as infile:
                out.write(infile.read())
                
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Claude Audit Bundle successfully generated at: {output_file}")
    
if __name__ == "__main__":
    build_claude_audit()
