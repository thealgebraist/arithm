# 1 Hour Optimization Challenge: 64-Opcode + 1024-Double Static Data

## Objective
Find the optimal compression kernel defined by:
1.  **Code**: Exactly 64 Instructions/Opcodes (The "Algorithm").
2.  **Data**: Exactly 1024 `doubles` (8KB) of pre-trained constants/weights/tables.
3.  **Goal**: Maximize Compression Ratio (or minimize Loss) on a target corpus.

## Constraints
- **Total Compute**: 1 GPU Hour / 1 CPU Hour.
- **Search Space**: Discrete (Code) $\times$ Continuous (Data).
- **Methods**: MILP, LP, Convex, FFNN, Heuristics.

---

## The 16 Search Strategies

We execute these strategies in parallel (or time-sliced) to explore different inductive biases.

### Cluster A: Differentiable Programming (The smooth approach)
**1. Soft-CPU / Differentiable Neural Computer**
*   **Concept**: Represent the 64 opcodes as a sequence of "Soft Gates" (weighted sum of all possible opcodes). The 1024 doubles are the learnable mixing weights and memory slots.
*   **Optimization**: Gradient Descent (Adam) end-to-end.
*   **Finalization**: `argmax` the soft gates to discrete 64 instructions.

**2. Constrained Neural Compression (Distillation)**
*   **Concept**: Train a standard dense FFNN with a total parameter count constraint of 1024.
*   **Structure**: A highly specific architecture (e.g., Factorized Linear layers) where the weights map to the 1024 doubles and the architecture logic fits in 64 ops.
*   **Optimization**: Standard Backprop.

**3. Manifold Optimization (Riemannian)**
*   **Concept**: Treat the 1024 doubles as a point on a specific manifold (e.g., Orthogonal group for rotation matrices) to preserve energy/entropy.
*   **Optimization**: Riemannian Gradient Descent (Geodesic steps).
*   **Code**: Fixed linear transform (Matrix Mul).

**4. Convex Dictionary Learning (Sparse Coding)**
*   **Concept**: Learn 1024 "atoms" (basis vectors) such that text chunks can be reconstructed sparsely.
*   **Optimization**: Block-Coordinate Descent (Convex optimization steps like FISTA/Lasso).
*   **Code**: "Find nearest atoms and subtract".

### Cluster B: Discrete & combinatorial (The hard approach)
**5. MILP Code Scheduling**
*   **Concept**: Formulate the data-dependency graph of the compression arithmetic.
*   **Optimization**: Use a Mixed Integer Linear Solver (SCIP/Gurobi) to find the optimal arrangement of 64 instructions to minimize register spilling and maximize throughput (Speed optimization).
*   **Data**: Fixed to standard constants (e.g. primes) initially.

**6. Genetic Algorithm (CMA-ES Hybrid)**
*   **Concept**: Two-loop optimization.
*   **Outer Loop**: Genetic Algorithm evolves the 64 discrete opcodes (mutation/crossover).
*   **Inner Loop**: CMA-ES (Covariance Matrix Adaptation) optimizes the 1024 doubles for the fitness of that specific code.

**7. Monte Carlo Tree Search (AlphaCoder-style)**
*   **Concept**: Treat code generation as a game tree. State = current program lines. Action = Append Opcode.
*   **Optimization**: MCTS guided by a lightweight value function (quick loss check on 1KB data).

**8. Symbolic Regression (Genetic Programming)**
*   **Concept**: Represent the algorithm as an expression tree.
*   **Optimization**: Evolve trees to predict the next byte ($P(x_{t+1} | x_t)$).
*   **Constraint**: Prune trees > 64 nodes. Constants are optimized via local BFGS.

### Cluster C: Spectral & Mathematical
**9. Generalized BWT (Continuous Permutation)**
*   **Concept**: The 1024 doubles form a learnable Permutation Matrix ($32 \times 32$) to reorder inputs before RLE/Entropy coding.
*   **Optimization**: Sinkhorn Sorting (Differentiable) then annealing to hard permutation.

**10. Basis Discovery (SVD/PCA/ICA)**
*   **Concept**: 1024 doubles = Eigenvectors of the text's sliding window covariance matrix.
*   **Optimization**: Exact SVD (linalg.svd).
*   **Code**: Projection ($X \cdot W$).

**11. Fourier/Wavelet Search (FFT)**
*   **Concept**: 1024 doubles = Optimal coefficients to keep in a Fourier Transform.
*   **Optimization**: "Keep Largest" (Thresholding) or "Keep most information-dense" (Entropy minimization).

**12. Linear Programming Relaxation (Control Flow)**
*   **Concept**: Model the program as a Probabilistic Finite Automaton.
*   **Optimization**: LP to solve for steady-state probabilities (the 1024 doubles) that maximize prediction likelihood.

### Cluster D: Heuristic & Hybrid
**13. Quantized Look-Up Table (LUT)**
*   **Concept**: The 1024 doubles are actually $K$ centroids of a Vector Quantizer.
*   **Optimization**: K-Means Clustering on the dataset context vectors.
*   **Code**: "Find nearest centroid".

**14. Simulated Annealing (Program Space)**
*   **Concept**: Random walk in the space of 64-opcode assembly. Swap instructions, change operands.
*   **Temperature**: Cooling schedule determines acceptance of worse solutions.

**15. Meta-Learning (MAML)**
*   **Concept**: Learn the 1024 doubles such that they act as a "Fast Weight" initialization.
*   **Optimization**: Second-order gradient optimization.

**16. Gradient-Free Pattern Search (Nelder-Mead)**
*   **Concept**: Treat the entire system as a Black Box $F(Data, Code)$.
*   **Optimization**: Direct search on the 1024-dimensional hypercube. Only viable if the function is smooth.

---

## Execution Plan (1 Hour Budget)

We cannot run all 16 to convergence. We use a **Multi-Armed Bandit** approach (Successive Halving).

1.  **Phase 1 (0-10 mins)**: **Exploration**.
    *   Launch all 16 strategies on a tiny subset of data (e.g., 50KB).
    *   Run each for ~30 seconds.
2.  **Phase 2 (10-20 mins)**: **Selection**.
    *   Pick Top 4 performing strategies.
    *   Run each for ~2.5 minutes on medium data (1MB).
3.  **Phase 3 (20-60 mins)**: **Exploitation**.
    *   Pick the **Winner**.
    *   Dedicate remaining ~40 minutes to full convergence on the Winner (Full Compute).
    *   Example: If "Soft-CPU" wins, run full Adam training for 1000 epochs.

## Recommended Implementation
We will implement a `meta_search.py` harness that:
1.  Defines a common interface `Strategy.optimize(time_budget)`.
2.  Implements the 4 most promising diverse strategies (Soft-CPU, Genetic, K-Means, SVD).
3.  Runs the Bandit allocation.
