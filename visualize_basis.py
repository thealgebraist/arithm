import numpy as np
import os
import json

def analyze_patterns_spectral(filepath, n=64):
    with open(filepath, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    data = raw[:100000].astype(float)
    pad = (n - (len(data) % n)) % n
    mat = np.pad(data, (0, pad)).reshape(-1, n)
    
    # SVD for optimal basis
    mean = np.mean(mat, axis=0)
    u, s, vh = np.linalg.svd(mat - mean, full_matrices=False)
    
    top_bases = []
    for i in range(5):
        basis = vh[i]
        # Map basis vector to visible character space for inspection
        basis_shifted = ((basis - np.min(basis)) / (np.max(basis) - np.min(basis)) * 94 + 32).astype(int)
        char_repr = "".join([chr(c) for c in basis_shifted])
        top_bases.append(char_repr)
    
    return top_bases, s.tolist()

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
basis_discovery = {}
for book in books[:4]:
    bases, singular_vals = analyze_patterns_spectral(book)
    basis_discovery[os.path.basename(book)] = {
        "top_5_bases": bases,
        "singular_values": singular_vals[:10]
    }

print("\nOptimal Basis Discovery (64-char patterns):")
for name, res in basis_discovery.items():
    print(f"\nBook: {name}")
    for i, b in enumerate(res['top_5_bases']):
        print(f"  Basis {i+1}: {b}")

with open("basis_discovery.json", "w") as f:
    json.dump(basis_discovery, f, indent=2)
