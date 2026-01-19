import numpy as np
import os
import json

def optimize_mapping(filepath):
    # Pass 1: Find the 256 optimal doubles representing the signal basis
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(float)
    
    # We want to find a 256-dim basis that captures the most "grammar" info
    # We'll use PCA as the convex optimization target for L2-loss minimality
    n_components = 256
    limit = 100000
    sub = data[:limit].reshape(-1, 1)
    
    # Actually, the mapping should probably be a global frequency table + 256 context weights
    # Let's find 256 doubles that best weight the mixing of different orders
    # We'll simulate this with a 256-point spectral map
    
    # Heuristic: Find top 256 frequencies (real+imag parts = 2 * 128)
    yf = np.fft.fft(data[:32768])
    indices = np.argsort(np.abs(yf))[::-1][:128]
    mapping = []
    for idx in indices:
        mapping.append(float(np.real(yf[idx])))
        mapping.append(float(np.imag(yf[idx])))
    
    # Ensure exactly 256 doubles
    while len(mapping) < 256: mapping.append(0.0)
    return mapping[:256]

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
pass1_results = {}
for book in books[:4]:
    print(f"Pass 1 Optimization for {os.path.basename(book)}...")
    pass1_results[os.path.basename(book)] = optimize_mapping(book)

with open("pass1_mapping.json", "w") as f:
    json.dump(pass1_results, f)
