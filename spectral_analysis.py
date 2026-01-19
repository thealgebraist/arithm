import numpy as np
import os
import json

def get_entropy(data):
    if len(data) == 0: return 0
    # Rescale/Quantize to 256 levels to compare with raw bytes
    d_min, d_max = np.min(data), np.max(data)
    if d_max > d_min:
        quantized = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        quantized = np.zeros_like(data).astype(np.uint8)
    
    counts = np.bincount(quantized, minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))

def manual_haar(data):
    n = len(data)
    if n % 2 != 0: data = np.pad(data, (0, 1))
    d = data.reshape(-1, 2)
    avgs = (d[:, 0] + d[:, 1]) / np.sqrt(2)
    diffs = (d[:, 0] - d[:, 1]) / np.sqrt(2)
    return np.concatenate([avgs, diffs])

def pca_basis(data, n=16):
    pad_len = (n - (len(data) % n)) % n
    padded = np.pad(data, (0, pad_len), mode='constant')
    matrix = padded.reshape(-1, n)
    if matrix.shape[0] < n: return None, None
    
    # Center the data
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    
    # SVD for PCA
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    # vh rows are the basis vectors (principal components)
    transformed = centered @ vh.T
    evals = (s ** 2) / (matrix.shape[0] - 1)
    return evals, transformed

def run_suite(filepath):
    print(f"Analyzing {os.path.basename(filepath)}...")
    with open(filepath, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    if len(raw) < 1024: return None
    
    limit = 50000
    data = raw[:limit].astype(float)
    
    res = {
        "orig": get_entropy(data),
        "fft": get_entropy(np.abs(np.fft.fft(data))),
        "haar": get_entropy(manual_haar(data)),
        "matrices": {}
    }
    
    for n in [16, 32, 64]:
        evals, trans = pca_basis(data, n=n)
        if evals is not None:
            res["matrices"][f"{n}x{n}"] = {
                "top_eval_ratio": float(evals[0] / np.sum(evals)),
                "entropy": get_entropy(trans.flatten())
            }
    return res

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
all_res = {}
for book in books[:8]:
    r = run_suite(book)
    if r: all_res[os.path.basename(book)] = r

print("\nSummary of Entropy Comparison:")
header = f"{'Book':<20} | {'Raw':<6} | {'FFT':<6} | {'Haar':<6}"
print(header)
print("-" * len(header))
for name, r in all_res.items():
    print(f"{name[:20]:<20} | {r['orig']:.2f} | {r['fft']:.2f} | {r['haar']:.2f}")

print("\nMatrix PCA Entropy (Basis Search):")
header = f"{'Book':<20} | {'16x16':<6} | {'32x32':<6} | {'64x64':<6}"
print(header)
print("-" * len(header))
for name, r in all_res.items():
    m = r['matrices']
    v16 = m.get('16x16', {}).get('entropy', 0)
    v32 = m.get('32x32', {}).get('entropy', 0)
    v64 = m.get('64x64', {}).get('entropy', 0)
    print(f"{name[:20]:<20} | {v16:.2f} | {v32:.2f} | {v64:.2f}")

with open("spectral_results.json", "w") as f:
    json.dump(all_res, f, indent=2)
