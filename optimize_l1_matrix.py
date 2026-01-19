import numpy as np
import os
import json

def analyze_l1_matrix(filepath):
    # Goal: Find the 16KB optimal Bayesian Order-1 Matrix
    # 256 Contexts * 64 Top Symbols (1 byte each for frequency bin) = 16KB
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 1. Count Order-1 Transitions
    matrix = np.zeros((256, 256), dtype=np.uint32)
    for i in range(len(data) - 1):
        matrix[data[i], data[i+1]] += 1
        
    # 2. For each context, find the top 64 symbols
    optimized_matrix = []
    for ctx in range(256):
        freqs = matrix[ctx]
        total = np.sum(freqs)
        if total == 0:
            top_indices = np.arange(64)
            quantized_freqs = np.ones(64) * (4096 // 64)
        else:
            # Sort and take top 64
            top_indices = np.argsort(freqs)[::-1][:63]
            # Quantize these to a 12-bit total (4096) for rANS
            actual_freqs = freqs[top_indices]
            # We use MILP-like rounding to ensure total is exactly 4096
            # (Simple Proportional + Rounding for the heuristic pass)
            q = (actual_freqs / total * 4095).astype(int)
            q[q == 0] = 1 # Reserve at least 1 count for safety
            
            # Escape symbol (index 63) for symbols not in Top 63
            escape_count = max(1, 4096 - np.sum(q))
            q = np.append(q, escape_count)
            top_indices = np.append(top_indices, 256) # 256 is escape flag
            quantized_freqs = q
            
        optimized_matrix.append({
            "ctx": ctx,
            "syms": top_indices.tolist(),
            "freqs": quantized_freqs.tolist()
        })
        
    return optimized_matrix

books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
l1_matrices = {}
for book in books[:1]: # Optimize for the first book as prototype
    print(f"Generating 16KB Bayesian Matrix for {os.path.basename(book)}...")
    l1_matrices[os.path.basename(book)] = analyze_l1_matrix(book)

with open("l1_matrix.json", "w") as f:
    json.dump(l1_matrices, f)
