import os
import time
import subprocess
import json

def run_bench(mode, filepath):
    bin_path = "temp.bin"
    out_path = "temp.out"
    subprocess.run(["./arithm", str(mode), "c", filepath, bin_path], check=True)
    size = os.path.getsize(bin_path)
    subprocess.run(["./arithm", str(mode), "d", bin_path, out_path], check=True)
    subprocess.run(["diff", filepath, out_path], check=True)
    return size

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
books = books[:4] # Fewer books for 32-way sweep

# mode calculation: (order << 2) | (m & 3) [BWT/MTF irrelevant here] | MILP(32) | FUZZY(64)
# We will sweep Order 0..7 and MILP/Fuzzy combinations
modes = []
for order in range(8):
    for milp in [0, 32]:
        for fuzzy in [0, 64]:
            modes.append((order << 2) | milp | fuzzy)

results = {}
for book in books:
    bname = os.path.basename(book)
    print(f"Sweep for {bname}...")
    orig = os.path.getsize(book)
    bz2 = len(subprocess.run(["bzip2", "-c", book], capture_output=True).stdout)
    book_res = {"orig": orig, "bz2": bz2, "m": {}}
    for m in modes:
        try:
            sz = run_bench(m, book)
            book_res["m"][m] = sz
            print(f"  Mode {m:<3}: {sz:>8} bytes ({(sz/orig):.4f}) {'*' if sz < bz2 else ''}")
        except:
            pass
    results[bname] = book_res

with open("sweep_results.json", "w") as f:
    json.dump(results, f)
