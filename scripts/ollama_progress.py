"""Progress of an Ollama model pull: sums the chunk records next to the -partial blob."""
import glob
import json
import os

total = done = 0
for f in glob.glob(r"D:\OllamaModels\blobs\*-partial-*"):
    try:
        with open(f, encoding="ascii") as fh:
            p = json.load(fh)
        total += int(p.get("Size", 0))
        done += int(p.get("Completed", 0))
    except (OSError, ValueError):
        pass
complete = [os.path.basename(f) for f in glob.glob(r"D:\OllamaModels\blobs\sha256-*") if "partial" not in f]
print(f"pull: {done / 2**20:,.0f} of {total / 2**20:,.0f} MB in chunk records; complete blobs: {len(complete)}")
