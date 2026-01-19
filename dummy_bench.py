import os
import time
import subprocess
import json

def get_stats(cmd_c, cmd_d, filepath):
    # Compression
    start_c = time.time()
    res_c = subprocess.run(cmd_c, capture_output=True)
    t_c = time.time() - start_c
    size = len(res_c.stdout)
    
    # Decompression (writing to /dev/null)
    # We pipe the compressed result to the decompressor
    start_d = time.time()
    subprocess.run(cmd_d, input=res_c.stdout, capture_output=True)
    t_d = time.time() - start_d
    
    return size, t_c, t_d

def run_arithm(mode, filepath):
    bin_path = "temp.bin"
    out_path = "temp.out"
    # Compression
    start_c = time.time()
    subprocess.run(["./arithm", str(mode), "c", filepath, bin_path], check=True)
    t_c = time.time() - start_c
    size = os.path.getsize(bin_path)
    # Decompression
    start_d = time.time()
    subprocess.run(["./arithm", str(mode), "d", bin_path, out_path], check=True)
    t_d = time.time() - start_d
    return size, t_c, t_d

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
books.sort()

results = []
for book in books:
    bname = os.path.basename(book)
    bsize = os.path.getsize(book)
    if bsize < 1000: continue # Skip empty/tiny files
    
    print(f"Benchmarking {bname} ({bsize/1024:.1f} KB)...")
    
    try:
        gz_size, gz_c, gz_d = get_stats(["gzip", "-c"], ["gzip", "-d", "-c"], book)
        bz_size, bz_c, bz_d = get_stats(["bzip2", "-c"], ["bzip2", "-d", "-c"], book)
        ar_size, ar_c, ar_d = run_arithm(0, book)
        
        results.append({
            "book": bname,
            "size": bsize,
            "gzip": {"ratio": gz_size/bsize, "c_speed": bsize/gz_c, "d_speed": bsize/gz_d},
            "bzip2": {"ratio": bz_size/bsize, "c_speed": bsize/bz_c, "d_speed": bsize/bz_d},
            "arithm": {"ratio": ar_size/bsize, "c_speed": bsize/ar_c, "d_speed": bsize/ar_d}
        })
    except Exception as e:
        print(f"  Error benchmarking {bname}: {e}")

# Calculate averages
def avg(key, subkey):
    vals = [r[key][subkey] for r in results]
    return sum(vals) / len(vals)

print("\n" + "="*110)
print(f"{'Tool':<15} | {'Avg Ratio':<10} | {'Avg Enc Speed':<15} | {'Avg Dec Speed':<15}")
print("-" * 110)

for tool in ["gzip", "bzip2", "arithm"]:
    r = avg(tool, "ratio")
    c = avg(tool, "c_speed") / 1024 / 1024
    d = avg(tool, "d_speed") / 1024 / 1024
    name = "L1-Bayes-rANS" if tool == "arithm" else tool
    print(f"{name:<15} | {r:<10.3f} | {c:<11.2f} MB/s | {d:<11.2f} MB/s")

print("="*110)
print(f"Total Books Benchmarked: {len(results)}")
