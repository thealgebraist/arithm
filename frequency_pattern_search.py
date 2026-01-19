import numpy as np
import os
import json

def find_patterns_via_fft_lp(filepath, window_size=512):
    with open(filepath, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Take a representative slice
    data = raw[:20000].astype(float)
    
    # We sliding window or take global FFT
    # Global FFT identifies persistent periodicities (long patterns)
    n = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(n)
    
    # LP Heuristic: L1-regularized Sparse Recovery of the signal
    # We want to find the top Frequencies that explain the "shape" of the book
    # This identifies "rhythms" in the text (sentence lengths, repeated headers)
    
    magnitudes = np.abs(yf)
    # Filter for low frequencies (macro-patterns)
    indices = np.argsort(magnitudes)[::-1] # Significant peaks
    
    discovered_patterns = []
    # Identify the Top 10 periodicities
    for i in range(1, 20): # ignore DC at 0
        idx = indices[i]
        freq = np.abs(xf[idx])
        if freq == 0: continue
        period = 1.0 / freq
        # Magnitude represents the "strength" of this periodicity
        magnitude = magnitudes[idx]
        
        # Heuristic "LP" thresholding for significance
        if magnitude > np.mean(magnitudes) * 5:
            discovered_patterns.append({
                "frequency": float(freq),
                "period_length": float(period),
                "strength": float(magnitude)
            })
            
    return discovered_patterns

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
results = {}

for book in books[:4]:
    name = os.path.basename(book)
    print(f"Searching frequency patterns in {name}...")
    patterns = find_patterns_via_fft_lp(book)
    # Sort by strength and keep meaningful ones
    patterns = sorted(patterns, key=lambda x: x['strength'], reverse=True)[:10]
    results[name] = patterns

print("\nFrequency Domain Pattern Results (Periodicity):")
for name, p_list in results.items():
    print(f"\n{name}:")
    for p in p_list:
        print(f"  Period: {p['period_length']:>8.2f} chars | Strength: {p['strength']:>12.2f}")

with open("frequency_patterns.json", "w") as f:
    json.dump(results, f, indent=2)
