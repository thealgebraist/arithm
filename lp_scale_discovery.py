import numpy as np
import os
import json

def reconstruct_lp_heuristic(yf, n, budget=50):
    # This heuristic simulates a Sparse Recovery LP (min ||x||_1 s.t. Ax=b)
    # We select the top 'budget' frequencies that capture the most energy.
    indices = np.argsort(np.abs(yf))[::-1]
    
    # Prune DC and extremely high noise
    selected = []
    for idx in indices:
        if idx == 0: continue
        selected.append(idx)
        if len(selected) >= budget: break
    
    # Create the sparse reconstruction
    sparse_yf = np.zeros_like(yf)
    for idx in selected:
        sparse_yf[idx] = yf[idx]
    
    reconstructed = np.fft.ifft(sparse_yf)
    # The reconstruction captures the "Macro-Pattern" of the text
    return np.real(reconstructed), selected

def automated_discovery(filepath):
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)[:16384].astype(float)
    
    n = len(data)
    yf = np.fft.fft(data)
    
    # Find patterns at different scales (Budget 20 for Long, 100 for Short)
    long_recon, long_indices = reconstruct_lp_heuristic(yf, n, 10)
    short_recon, short_indices = reconstruct_lp_heuristic(yf, n, 100)
    
    res = {
        "long_periods": [float(1.0/abs(np.fft.fftfreq(n)[idx])) for idx in long_indices[:5]],
        "short_periods": [float(1.0/abs(np.fft.fftfreq(n)[idx])) for idx in short_indices[50:55]]
    }
    return res

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
discovery_log = {}
for book in books[:4]:
    print(f"LP-Sparse analysis for {os.path.basename(book)}...")
    discovery_log[os.path.basename(book)] = automated_discovery(book)

print("\nLP-Based Scale Discovery (Periodicity lengths):")
for name, d in discovery_log.items():
    print(f"\n{name}:")
    print(f"  Significant Long Patterns: {', '.join([f'{p:.1f}' for p in d['long_periods']])}")
    print(f"  Significant Short Patterns: {', '.join([f'{p:.1f}' for p in d['short_periods']])}")

with open("lp_scale_discovery.json", "w") as f:
    json.dump(discovery_log, f, indent=2)
