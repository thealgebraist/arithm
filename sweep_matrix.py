import os
import time
import subprocess
import json

def get_stats(cmd_c, cmd_d, filepath):
    start_c = time.time()
    res_c = subprocess.run(cmd_c, capture_output=True)
    t_c = time.time() - start_c
    size = len(res_c.stdout)
    start_d = time.time()
    subprocess.run(cmd_d, input=res_c.stdout, capture_output=True)
    t_d = time.time() - start_d
    return size, t_c, t_d

def run_arithm_sweep(padding, ctx_bits, filepath):
    bin_path = f"temp_{padding}_{ctx_bits}.bin"
    out_path = f"temp_{padding}_{ctx_bits}.out"
    
    # Compress
    start_c = time.time()
    # Usage: ./arithm <pad> c <in> <out> <ctx_bits>
    subprocess.run(["./arithm", str(padding), "c", filepath, bin_path, str(ctx_bits)], check=True)
    t_c = time.time() - start_c
    size = os.path.getsize(bin_path)
    
    # Decompress
    start_d = time.time()
    subprocess.run(["./arithm", str(padding), "d", bin_path, out_path, str(ctx_bits)], check=True)
    t_d = time.time() - start_d
    
    # Cleanup to save space
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(out_path): os.remove(out_path)
    
    return size, t_c, t_d

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
# Use a subset of books for the matrix sweep to keep time reasonable (e.g. 8 books)
# If fast enough, we can do 64.
sweep_books = books[:8] 

# Parameters
paddings = [0, 16, 32, 33, 34, 48, 64, 128]
# Context bits: 8 (256x256), 7 (128x128), 6 (64x64), ..., 3 (8x8)
# The user asked for specific sizes: 128x128 (7 bits), 64x64 (6), 32x32 (5), 16x16 (4), 8x8 (3)
table_configs = [8, 7, 6, 5, 4, 3] # 8=256(Full), 7=128, etc.

results = {}

print(f"Starting Sweep on {len(sweep_books)} books...")
print(f"Paddings: {paddings}")
print(f"Context Bits: {table_configs}")

for book in sweep_books:
    bname = os.path.basename(book)
    bsize = os.path.getsize(book)
    print(f"\nProcessing {bname} ({bsize} bytes)...")
    
    for ctx in table_configs:
        for pad in paddings:
            key = f"{ctx}_{pad}"
            try:
                sz, tc, td = run_arithm_sweep(pad, ctx, book)
                if key not in results: results[key] = {"total_size":0, "total_orig":0, "time_c":0, "time_d":0}
                results[key]["total_size"] += sz
                results[key]["total_orig"] += bsize
                results[key]["time_c"] += tc
                results[key]["time_d"] += td
            except Exception as e:
                print(f"Failed {key}: {e}")

print("\n" + "="*80)
print(f"{'Ctx':<5} | {'Pad':<5} | {'Ratio':<6} | {'Enc (MB/s)':<12} | {'Dec (MB/s)':<12}")
print("-" * 80)

# Aggregated Results
unique_keys = sorted(results.keys(), key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1])))
for k in unique_keys:
    ctx, pad = k.split('_')
    d = results[k]
    ratio = d["total_size"] / d["total_orig"]
    enc_speed = (d["total_orig"] / d["time_c"]) / 1024 / 1024
    dec_speed = (d["total_orig"] / d["time_d"]) / 1024 / 1024
    
    # Format Context "Size" string (e.g. 7 -> 128x128)
    # The user asked for "128x128" etc.
    dim = 1 << int(ctx)
    dim_str = f"{dim}x{dim}"
    if int(ctx) == 8: dim_str = "Full"
    
    print(f"{dim_str:<5} | {pad:<5} | {ratio:<6.3f} | {enc_speed:<12.2f} | {dec_speed:<12.2f}")
