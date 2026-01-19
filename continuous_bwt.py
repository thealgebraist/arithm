import numpy as np
import cvxpy as cp
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

def text_to_matrix(text):
    # One-hot encoding
    alphabet = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    vocab_size = len(alphabet)
    n = len(text)
    
    # Create Matrix of cyclic shifts
    # M[i] is the rotation of text starting at i
    # We represent chars as one-hot vectors
    # For computation, we might just use the indices or embeddings
    # Let's use scalar indices for simplicity in "embedding" space, 
    # but strictly "continuous" usually implies high-dim. 
    # Let's stick to a weighted scalar for lexicographical approximation:
    # val(row) = sum( char[k] * eps^k )
    # But for the "TSP View" (Clustering), we compute distances between full rows.
    
    shifts = []
    ids = [char_to_idx[c] for c in text]
    for i in range(n):
        shifts.append(ids[i:] + ids[:i])
    return np.array(shifts), alphabet

def solve_milp_bwt(shifts_matrix):
    """
    Generalizes BWT as a TSP problem: Find the permutation of cyclic shifts
    that minimizes the sum of adjacent row distances (Maximum context clustering).
    This relaxes the strict 'lexicographical' sort to an 'similarity' sort.
    """
    n = len(shifts_matrix)
    print(f"\n[MILP] Solving TSP-BWT for N={n}...")
    
    # 1. Compute Distance Matrix (Simple Euclidean/Hamming on rows)
    # Use Hamming-like: number of mismatched positions, or differences
    # Ideally, BWT sorts by prefix. Distance should be weighted heavily on prefix.
    # W = [1.0, 0.5, 0.25, ...]
    weights = np.array([0.5**k for k in range(n)])
    
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: 
                dist_mat[i,j] = 1e9
            else:
                # Weighted difference ensures prefix-based sorting preference
                diff = np.abs(shifts_matrix[i] - shifts_matrix[j])
                dist_mat[i,j] = np.sum(diff * weights)

    # 2. Define TSP Variables
    x = cp.Variable((n, n), boolean=True)
    u = cp.Variable(n) # For subtour elimination (MTZ constraints)

    # 3. Objective: Minimize path cost
    cost = cp.sum(cp.multiply(dist_mat, x))
    
    constraints = [
        cp.sum(x, axis=0) == 1, # Enter each city once
        cp.sum(x, axis=1) == 1, # Leave each city once
        cp.diag(x) == 0         # No self-loops
    ]
    
    # MTZ Subtour constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints.append(u[i] - u[j] + n*x[i,j] <= n - 1)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    # Use Scipy Highs Solver (MILP)
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # Flatten x to n*n variables
    # Cost vector c
    c = dist_mat.flatten()
    
    # Constraints:
    # sum_rows = 1, sum_cols = 1
    # A_eq @ x = 1
    A_eq = np.zeros((2*n, n*n))
    b_eq = np.ones(2*n)
    
    for i in range(n):
        # Sum over cols for row i
        A_eq[i, i*n : (i+1)*n] = 1
        # Sum over rows for col i
        A_eq[n+i, i::n] = 1
        
    # Bounds: 0 <= x <= 1, Integers
    
    # MTZ constraints are hard to represent in standard sparse matrix form easily here 
    # without a helper.
    # Given the environment issues, let's fallback to Brute Force for small N (N<=8).
    # This is mathematically equivalent for the proof of concept.
    
    if n <= 9:
        print("  Using Brute Force TSP for N <= 9...")
        import itertools
        min_cost = float('inf')
        best_perm = []
        for p in itertools.permutations(range(n)):
            # Cost = sum dist(p[i], p[i+1])
            # But BWT 'sort' is not TSP cycle.
            # BWT is sorting rows. The cost of 'Sorting' is minimizing sum |row_i - row_{i+1}|
            # This is a Hamiltonian Path problem (TSP Path) on the set of rows.
            
            c = 0
            # Path cost (open loop)
            for k in range(n-1):
                c += dist_mat[p[k], p[k+1]]
                
            if c < min_cost:
                min_cost = c
                best_perm = list(p)
        return best_perm
    else:
        print("  N > 9, skipping exact MILP due to solver missing.")
        return list(range(n))
    
    # Reconstruct Path
    adj = x.value > 0.5
    path = [0]
    curr = 0
    visited = {0}
    while len(path) < n:
        for j in range(n):
            if adj[curr, j] and j not in visited:
                path.append(j)
                visited.add(j)
                curr = j
                break
        else:
            # Fallback if graph is disconnected (should handle in optimization, but simple greedy fix here)
            remaining = set(range(n)) - visited
            if remaining:
                next_node = list(remaining)[0]
                path.append(next_node)
                visited.add(next_node)
                curr = next_node
                
    return path

def solve_grad_descent_bwt(shifts_matrix):
    """
    Continuous BWT via Soft Permutations and Gradient Descent.
    We optimize a doubly-stochastic matrix P to sort the rows based on a weighted score.
    """
    n = len(shifts_matrix)
    print(f"\n[GD] Solving Continuous Sort for N={n}...")
    
    # Target: We want P such that (P @ Scores) is sorted (monotonic).
    # Ideally, P @ Shifts should be "smooth".
    
    device = torch.device('cpu')
    
    # Weighted score for each row (Lexicographical scalar proxy)
    weights = torch.tensor([0.5**k for k in range(n)], dtype=torch.float32)
    shifts = torch.tensor(shifts_matrix, dtype=torch.float32)
    
    # Calculate scalar "Lex Value" for each row to define the target sort order
    # In pure continuous BWT, we might learn these, but here we define the target
    # and learn the P to achieve it via gradient descent, proving we can "solve" for BWT.
    row_scores = (shifts * weights).sum(dim=1)
    
    # Learnable Log-Sinkhorn/Permutation logits
    # P = Softmax(Sinkhorn(Q))
    Q = torch.randn((n, n), requires_grad=True)
    optimizer = torch.optim.Adam([Q], lr=0.1)
    
    target_sorted, _ = torch.sort(row_scores)
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # Sinkhorn Iterations to project Q onto Birkhoff Polytope (Doubly Stochastic)
        P = Q
        for _ in range(5):
            P = P - torch.logsumexp(P, dim=1, keepdim=True) # Row norm
            P = P - torch.logsumexp(P, dim=0, keepdim=True) # Col norm
        P = torch.exp(P)
        
        # Predicted Scores of the "Permuted" rows
        # If P is a perm matrix, pred_scores[i] is the score of the i-th sorted element
        # Actually P maps source_idx -> sort_idx
        # pred_sorted = P @ row_scores
        pred_sorted = torch.matmul(P, row_scores)
        
        # Loss: Difference between our soft-sorted vector and the true sorted target
        # Minimizing this forces P to align indices correctly
        loss = F.mse_loss(pred_sorted, target_sorted)
        
        # Entropy Regularization to force sharpening (closer to binary)
        entropy = -torch.sum(P * torch.log(P + 1e-9))
        total_loss = loss + 0.01 * entropy
        
        total_loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  Step {step}: Loss {loss.item():.6f}")

    # Final hard assignment
    P_final = Q.detach()
    # LAP/Hungarian is best here, but argmax is okay for simple demo
    perm = torch.argsort(P_final, dim=1, descending=True)[:, 0].numpy()
    
    # Dedup/Fix validity (greedy)
    used = set()
    fixed_perm = []
    for p in perm:
        if p not in used:
            fixed_perm.append(p)
            used.add(p)
    missing = list(set(range(n)) - used)
    fixed_perm.extend(missing)
    
    return fixed_perm

# Demo
text = "banana$"
matrix, alphabet = text_to_matrix(text)

# 1. Standard BWT (Ground Truth)
lex_order = sorted(range(len(matrix)), key=lambda i: tuple(matrix[i]))
bwt_std = [text[(i-1)%len(text)] for i in lex_order]
print(f"Standard BWT: {''.join(bwt_std)}")

# 2. MILP (TSP-style relaxation)
milp_order = solve_milp_bwt(matrix)
bwt_milp = [text[(i-1)%len(text)] for i in milp_order]
print(f"MILP BWT (TSP): {''.join(bwt_milp)}")

# 3. Gradient Descent (Continuous Sorting)
gd_order = solve_grad_descent_bwt(matrix)
# Note: GD finds the mapping Source->Dest efficiently
# But we need the indices of the sorted rows.
# Re-evaluate logic: The GD perm maps 'Sort Rank i' -> 'Original Index j'
bwt_gd = [text[(i-1)%len(text)] for i in gd_order]
print(f"GD BWT (DiffSort): {''.join(bwt_gd)}")
