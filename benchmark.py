import os
import time
import subprocess
import json

books = [f for f in os.listdir("books") if f.endswith(".txt")]
books = [os.path.join("books", f) for f in books]

results = []

for book in books:
    print(f"Benchmarking {book}...")
    original_size = os.path.getsize(book)
    
    # My Arithmetic Coder
    start = time.time()
    subprocess.run(["./arithm", "c", book, "temp.arith"], check=True)
    arith_c_time = time.time() - start
    arith_size = os.path.getsize("temp.arith")
    
    start = time.time()
    subprocess.run(["./arithm", "d", "temp.arith", "temp.out"], check=True)
    arith_d_time = time.time() - start
    
    # Verification
    subprocess.run(["diff", book, "temp.out"], check=True)
    
    # Gzip
    start = time.time()
    subprocess.run(["gzip", "-c", book], capture_output=True, check=True) # just measure time
    gzip_out = subprocess.run(["gzip", "-c", book], capture_output=True).stdout
    gzip_c_time = time.time() - start
    gzip_size = len(gzip_out)
    
    # Bzip2
    start = time.time()
    subprocess.run(["bzip2", "-c", book], capture_output=True, check=True)
    bzip2_out = subprocess.run(["bzip2", "-c", book], capture_output=True).stdout
    bzip2_c_time = time.time() - start
    bzip2_size = len(bzip2_out)
    
    results.append({
        "book": os.path.basename(book),
        "original": original_size,
        "arith": {"size": arith_size, "c_time": arith_c_time, "ratio": arith_size/original_size},
        "gzip": {"size": gzip_size, "c_time": gzip_c_time, "ratio": gzip_size/original_size},
        "bzip2": {"size": bzip2_size, "c_time": bzip2_c_time, "ratio": bzip2_size/original_size}
    })

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print Summary Table
print(f"{'Book':<20} | {'Orig':<10} | {'Arith':<10} | {'Gzip':<10} | {'Bzip2':<10}")
print("-" * 70)
for r in results:
    print(f"{r['book'][:20]:<20} | {r['original']:<10} | {r['arith']['size']:<10} | {r['gzip']['size']:<10} | {r['bzip2']['size']:<10}")

avg_arith = sum(r['arith']['ratio'] for r in results) / len(results)
avg_gzip = sum(r['gzip']['ratio'] for r in results) / len(results)
avg_bzip2 = sum(r['bzip2']['ratio'] for r in results) / len(results)

print("\nAverage Compression Ratio:")
print(f"Arith: {avg_arith:.4f}")
print(f"Gzip:  {avg_gzip:.4f}")
print(f"Bzip2: {avg_bzip2:.4f}")
