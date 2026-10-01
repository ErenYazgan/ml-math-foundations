import numpy as np

def power_iteration(A, num_iterations=1000, tolerance=1e-10):
    """
    Extracts the dominant eigenvalue and eigenvector of a square matrix
    using the Power Iteration method.
    
    Engineering Note: 
    Instead of solving the characteristic polynomial det(A - lambda*I) = 0 
    (which causes floating-point overflow and hardware instability for large matrices), 
    this brute-force iterative approach leverages CPU matrix multiplication speed 
    to isolate the most dominant variance axis (PCA foundation).
    """
    n = A.shape[0]
    
    # Step 1: Initialize a random vector (the "soup" containing all hidden eigenvectors)
    x = np.random.rand(n)
    
    for _ in range(num_iterations):
        # Step 2: Multiply to bend the vector towards the dominant axis
        x_new = np.dot(A, x)
        
        # Step 3: Normalize to prevent RAM/Floating-point overflow (The Guillotine)
        x_new = x_new / np.linalg.norm(x_new)
        
        # CPU Optimization: If the vector stops changing, break the loop early
        if np.linalg.norm(x_new - x) < tolerance:
            x = x_new
            break
            
        x = x_new
        
    # Step 4: Extract the eigenvalue using the Rayleigh Quotient
    # Formula: lambda = (x^T * A * x) / (x^T * x). Since length is 1, denominator is 1.
    eigenvalue = np.dot(x.T, np.dot(A, x))
    
    return eigenvalue, x


if __name__ == "__main__":
    # --- TEST ARENA ---
    # Simulating a symmetric covariance matrix (typical in PCA)
    np.random.seed(42) # For reproducible test results
    
    print("--- Power Iteration (From Scratch) vs Black-Box (NumPy) ---")
    
    A = np.array([[4.0, 1.0, 1.0],
                  [1.0, 3.0, 2.0],
                  [1.0, 2.0, 5.0]])
                  
    # 1. Execute Custom Engine
    scratch_val, scratch_vec = power_iteration(A)
    print(f"[From Scratch] Dominant Eigenvalue : {scratch_val:.6f}")
    print(f"[From Scratch] Dominant Eigenvector: {scratch_vec}")
    
    print("-" * 60)
    
    # 2. Execute NumPy Black-Box
    np_vals, np_vecs = np.linalg.eig(A)
    
    # NumPy returns all values; we must extract the maximum one for comparison
    max_idx = np.argmax(np_vals)
    print(f"[NumPy] Dominant Eigenvalue        : {np_vals[max_idx]:.6f}")
    
    # Engineering Gotcha: Eigenvectors can be pointing in exact opposite directions (e.g., [1, 2] vs [-1, -2]). 
    # Both are mathematically correct as they represent the same line/axis in space.
    print(f"[NumPy] Dominant Eigenvector       : {np_vecs[:, max_idx]}")