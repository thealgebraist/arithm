import os
import time
import subprocess
import json

def run_arithm_sweep_real(mode, ctx_bits, filepath):
    bin_path = f"temp_{mode}_{ctx_bits}.bin"
    out_path = f"temp_{mode}_{ctx_bits}.out"
    
    # Compress
    start_c = time.time()
    subprocess.run(["./arithm", str(mode), "c", filepath, bin_path, str(ctx_bits)], check=True)
    t_c = time.time() - start_c
    size = os.path.getsize(bin_path)
    
    # Decompress
    start_d = time.time()
    subprocess.run(["./arithm", str(mode), "d", bin_path, out_path, str(ctx_bits)], check=True)
    t_d = time.time() - start_d
    
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(out_path): os.remove(out_path)
    
    return size, t_c, t_d

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
sweep_books = books[:8] # Subset for speed

# Modes: 0 (1-Way), 2 (2-Way), 4 (4-Way)
# These represent increasing opcode complexity / parallelism utilization
real_modes = [0, 2, 4]
mode_labels = {0: "1-Way", 2: "2-Way", 4: "4-Way"}

# Table Sizes
table_configs = [8, 7, 6, 5, 4, 3]

results = {}
print(f"Starting Real Kernel Sweep on {len(sweep_books)} books...")

for book in sweep_books:
    bname = os.path.basename(book)
    bsize = os.path.getsize(book)
    # print(f"Processing {bname}...")
    
    for ctx in table_configs:
        for m in real_modes:
            key = f"{ctx}_{m}"
            try:
                sz, tc, td = run_arithm_sweep_real(m, ctx, book)
                if key not in results: results[key] = {"total_size":0, "total_orig":0, "time_c":0, "time_d":0}
                results[key]["total_size"] += sz
                results[key]["total_orig"] += bsize
                results[key]["time_c"] += tc
                results[key]["time_d"] += td
            except Exception as e:
                print(f"Failed {key}: {e}")

print("\n" + "="*80)
print(f"{'Ctx':<8} | {'Mode':<8} | {'Ratio':<6} | {'Enc (MB/s)':<12} | {'Dec (MB/s)':<12}")
print("-" * 80)

unique_keys = sorted(results.keys(), key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1])))
for k in unique_keys:
    ctx, m = k.split('_')
    d = results[k]
    ratio = d["total_size"] / d["total_orig"]
    enc_speed = (d["total_orig"] / d["time_c"]) / 1024 / 1024
    dec_speed = (d["total_orig"] / d["time_d"]) / 1024 / 1024
    
    dim = 1 << int(ctx)
    dim_str = f"{dim}x{dim}"
    if int(ctx) == 8: dim_str = "Full"
    
    print(f"{dim_str:<8} | {mode_labels[int(m)]:<8} | {ratio:<6.3f} | {enc_speed:<12.2f} | {dec_speed:<12.2f}")
