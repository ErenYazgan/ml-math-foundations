import numpy as np
import time

def benchmark_dynamic_system(matrix_size=300, time_steps=200):
    """
    Real-World Benchmark: Time-Series / Dynamical Systems
    Computes matrix states for sequential time steps (A^1, A^2, ..., A^n)
    demonstrating where Diagonalization mathematically destroys Brute Force.
    """
    print(f"--- Real-World Benchmark: Computing {time_steps} sequential states for a {matrix_size}x{matrix_size} Matrix ---\n")
    
    np.random.seed(42)
    A = np.random.randn(matrix_size, matrix_size)
    A = (A + A.T) / 2 
    A = A / np.max(np.abs(np.linalg.eigvals(A))) # Prevent Overflow
    
    # ==========================================
    # METHOD 1: Brute Force Loop (Naive Approach)
    # ==========================================
    start_time = time.time()
    for t in range(1, time_steps + 1):
        _ = np.linalg.matrix_power(A, t)
    brute_time = time.time() - start_time
    print(f"[Method 1] Brute Force Loop Time: {brute_time:.4f} seconds")
    
    # ==========================================
    # METHOD 2: Engineering Approach (Diagonalization)
    # ==========================================
    start_time = time.time()
    # Pay the heavy 'Setup Tax' only ONCE
    evals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    
    # Run the loop lightning fast
    for t in range(1, time_steps + 1):
        _ = np.dot(V, np.dot(np.diag(evals ** t), V_inv))
    diag_time = time.time() - start_time
    print(f"[Method 2] Diagonalization Loop Time: {diag_time:.4f} seconds")
    
    print(f"\nTime Multiplier: Diagonalization is {brute_time / diag_time:.1f}x times faster in sequential operations.")

if __name__ == "__main__":
    benchmark_dynamic_system()