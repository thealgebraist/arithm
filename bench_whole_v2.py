import os
import gzip
import time
import random
import numpy as np

# Increase limit for "Whole Book Heuristic"
# Processing 20KB is enough to show the effect without waiting hours for Python loops
DATA_LIMIT = 100000 

def get_bwt_cost(text, perm):
    # Calculate sum of adjacent row distances
    # Distance = Hamming distance of first 64 chars
    # We sample for speed
    
    n = len(perm)
    total_cost = 0.0
    
    # We can vectorize this cost calculation for the whole block?
    # No, permutation is arbitrary.
    # Just sample random pairs to estimate or iterate full?
    # For 20KB, iterating full 20,000 * 64 is 1.2M ops. Fast.
    
    # Convert text to integers for fast XOR/Diff
    t_ints = np.frombuffer(text.encode('latin1'), dtype=np.uint8)
    
    cost = 0
    # Window for distance
    W = 32 # Only compare first 32 bytes for the metric
    
    for k in range(n - 1):
        idx1 = perm[k]
        idx2 = perm[k+1]
        
        # Row 1: t_ints rolled by idx1 (actually suffix starting at idx1)
        # Slicing with numpy:
        # We need cyclic shift.
        # Construct slices:
        # Part A: t[i : i+W] (if i+W < n)
        # Part B: t[0 : W - (n-i)] (wrap)
        
        # Optimize: Pre-construct the 20K x 32 feature matrix?
        # 20000 * 32 bytes = 600KB. Easy.
        pass

def solve_fast_heuristic(filename):
    try:
        with open(filename, 'rb') as f:
            raw = f.read(DATA_LIMIT)
    except: return
    
    n = len(raw)
    if n < 100: return
    
    print(f"\nProcessing {os.path.basename(filename)} ({n} bytes)...")
    
    # 1. Feature Matrix (Cyclic Window)
    # Rows = cyclic shifts. We only store first W=32 bytes for metric calc.
    W = 16 # Small window for speed
    features = np.zeros((n, W), dtype=np.uint8)
    t_bytes = np.frombuffer(raw, dtype=np.uint8)
    for i in range(n):
        if i + W <= n:
            features[i] = t_bytes[i:i+W]
        else:
            # Wrap
            part1 = t_bytes[i:]
            part2 = t_bytes[:W - len(part1)]
            features[i] = np.concatenate([part1, part2])
            
    # 2. Initial Solution: Standard Suffix Array (Lexicographical)
    # We treat 'raw' as byte string
    # Python's sort is fast enough for 20KB
    print("  Generating Standard SA...")
    offsets = list(range(n))
    # Sort key: Cyclic shift
    # Optimization: Use just the features for sort? No, need precise BWT.
    # Standard SA uses full suffix.
    # Using 'features' is approximate BWT, but let's stick to true BWT as start.
    # Actually for 20KB, raw[i:] sort is fast.
    offsets.sort(key=lambda i: raw[i:] + raw[:i])
    
    # Evaluate Standard BWT cost
    # Cost = Sum of L1 distance between adjacent feature vectors
    # We use vectorization
    perm = np.array(offsets)
    ordered_feats = features[perm] # Reordered rows
    
    # Diff between Row i and Row i+1
    # diffs = sum(|row_i - row_{i+1}|)
    diff_matrix = np.abs(ordered_feats[:-1].astype(int) - ordered_feats[1:].astype(int))
    # Weighted? Standard L1 is fine for now
    current_cost = np.sum(diff_matrix)
    
    # Compressibility (Gzip)
    bwt_std = bytes([raw[(i-1)%n] for i in perm])
    gz_std = len(gzip.compress(bwt_std))
    print(f"  Std-BWT: Cost={current_cost}, Gzip={gz_std}")
    
    # 3. Local Optimization (TSP Heuristic)
    # We try to 'Greedy Insert' or 'Swap' blocks to lower cost.
    # Simple Heuristic: Look at high-cost jumps and try to find better neighbors.
    # Or just "Sort by Feature" using a different metric?
    # Let's try "Sorting by Features" using Euclidean distance (Space Filling Curve proxy)
    # Actually, pure sorting of the metric vectors might be a good initialization?
    
    # Metric Sort (Sort purely by the W-byte window values)
    # This ignores long-range context, effectively locally clustering.
    # This is equivalent to BWT with only depth W.
    print("  Generating Metric-Sort (Depth-16)...")
    # Sort integer vectors? View as void?
    # Lex sort of features is same as BWT-16.
    # But TSP wants *Hamming* closeness, not Lex closeness.
    # (0, 255) is far in Hamming, but close in Lex if (0,0) exists?
    # No, (0, 255) is far.
    
    # Let's try Linearizing via a pseudo-Hilbert index or just use the Feature rows directly
    # and run a quick Traveling Salesperson approximation on them.
    # Approximating TSP on 20,000 points in 16-D?
    # Use "Farthest Insertion" or "Nearest Neighbor" from random start?
    
    # Nearest Neighbor Construction
    print("  Running NN-TSP construction...")
    visited = np.zeros(n, dtype=bool)
    path = np.zeros(n, dtype=int)
    
    curr = np.random.randint(0, n)
    path[0] = curr
    visited[curr] = True
    
    # To do this for 20,000 points is O(N^2) = 400M comparisons. Too slow for pure Py.
    # We need a partitioned approach.
    # Cluster points first -> K-Means?
    # Or simpler: Sort by Feature (BWT-like) then locally optimize.
    
    # Optimization Strategy:
    # Take Std Permutation.
    # Iterate i from 0 to N-1:
    #   Look at window [i-K, i+K].
    #   Find best Swap(i, j) that minimizes local cost.
    
    optimized_perm = perm.copy()
    K = 50 # Local window lookahead
    
    # Vectorized Local Swap Check is hard.
    # Let's do a simple stochastic swap.
    # 100,000 random swaps.
    
    improved = False
    print("  Running 50k Stochastic Swaps...")
    
    # We'll just randomly pick indices and see if swapping helps.
    # Focus on "High Energy" boundaries.
    
    # Precompute costs array
    row_costs = np.sum(np.abs(features[optimized_perm[:-1]].astype(int) - features[optimized_perm[1:]].astype(int)), axis=1)
    
    for _ in range(50000):
        # Pick a high cost edge
        # idx = np.random.choice(n-1, p=probs) # expensive
        idx = np.random.randint(0, n-1)
        if row_costs[idx] < 10: continue # Skip if already good
        
        # Try to swap row (idx+1) with random other row that might match row (idx) better?
        # Or swap (idx) and (idx+1)? usually swapping adjacent doesn't help if they are far.
        # We need to find a 'target' that is close to optimized_perm[idx].
        
        # This allows us to jump out of Lex order.
        pass
        
    # Actually, let's try the pure "Metric Sort" (Sort by the raw byte values of the window)
    # This is effectively BWT-16.
    # Is BWT-16 better than BWT-Full for gzip?
    # Usually Full is better.
    # But let's check BWT-16 for the "Heuristic" check.
    
    # Metric Sort
    # We define a "Gray Code" score?
    # Simple: Lexicographical Sort of Features (which is what BWT-16 is)
    # Let's try "Sort by Sum of Features" (Intensity sort)
    
    # feature_sums = np.sum(features, axis=1)
    # sum_perm = np.argsort(feature_sums)
    # bwt_sum = bytes([raw[(i-1)%n] for i in sum_perm])
    # gz_sum = len(gzip.compress(bwt_sum))
    # print(f"  Sum-Sort: Gzip={gz_sum}") # Likely bad
    
    # Let's go back to the prompt: "TSP-BWT with heuristic tsp solver".
    # Since we established "Nearest Neighbor" is good in the small bench,
    # let's try a randomized NN for the big block.
    # Instead of full NN, we do "K-Nearest Neighbors" search?
    # Too heavy.
    
    # Approximated NN:
    # 1. Sort by features (Std BWT).
    # 2. This puts similar items close.
    # 3. "Locally Jitter":
    #    For i in range(n):
    #       Look at window [i, i+10].
    #       Find the permutation of these 10 that minimizes path cost.
    #       (Local Exhaustive TSP).
    
    print("  Running Windowed-TSP Smoothing (W=8)...")
    ordered_feats = features[perm]
    new_perm = perm.copy()
    
    WINDOW = 8
    # Sliding window
    for i in range(0, n - WINDOW, 4): # Step 4
        # Get subset of rows
        local_indices = new_perm[i : i+WINDOW]
        local_feats = features[local_indices]
        
        # Solve TSP exactly for these 8
        # Brute force 8! = 40k. A bit slow if we do it 5000 times (5000 * 40k = 200M ops).
        # Use greedy NN for the window.
        
        # Greedy NN for local window
        curr_local = 0 # Indices into local_feats, start with first
        path_l = [0]
        used_l = {0}
        
        for k in range(WINDOW - 1):
            last = path_l[-1]
            # Find closest in unused
            best_d = 999999
            best_idx = -1
            
            for cand in range(WINDOW):
                if cand in used_l: continue
                # Dist
                d = np.sum(np.abs(local_feats[last].astype(int) - local_feats[cand].astype(int)))
                if d < best_d:
                    best_d = d
                    best_idx = cand
            
            path_l.append(best_idx)
            used_l.add(best_idx)
            
        # Reorder perms
        new_perm[i : i+WINDOW] = local_indices[path_l]

    # Recalculate cost
    ordered_feats2 = features[new_perm]
    diff_matrix2 = np.abs(ordered_feats2[:-1].astype(int) - ordered_feats2[1:].astype(int))
    cost2 = np.sum(diff_matrix2)
    
    bwt_tsp = bytes([raw[(i-1)%n] for i in new_perm])
    gz_tsp = len(gzip.compress(bwt_tsp))
    
    print(f"  TSP-Smoothed: Cost={cost2} ({(cost2/current_cost):.3f}x), Gzip={gz_tsp}")

# Execute
books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
books = books[:1] # Just one for deep analysis
for b in books:
    solve_fast_heuristic(b)
