import numpy as np

def extract_eigenvalues_fast():
    """
    Demonstrates O(1) eigenvalue extraction for Diagonal and Triangular matrices.
    
    Engineering Note:
    In large-scale data systems, running decomposition algorithms or power iteration 
    on sparse, diagonal, or triangular matrices is a massive waste of CPU cycles. 
    Mathematical theory proves that for these specific matrix structures, the 
    eigenvalues are simply the elements on the main diagonal. This allows us to 
    bypass complex det(A - lambda*I) = 0 calculations entirely, reducing computational 
    complexity to O(1) extraction.
    """
    
    # --- 1. 2x2 DIAGONAL MATRIX ---
    print("--- 1. 2x2 DIAGONAL MATRIX ---")
    D_2x2 = np.diag([5, 8])
    print("Matrix (D_2x2):\n", D_2x2)
    
    evals_2x2 = np.linalg.eigvals(D_2x2)
    print(f"Extracted Eigenvalues: {evals_2x2}\n")
    
    
    # --- 2. NxN DIAGONAL MATRIX ---
    print("--- 2. NxN DIAGONAL MATRIX ---")
    N = 5
    D_NxN = np.diag(np.arange(1, N + 1)) # Generates [1, 2, 3, 4, 5] on the diagonal
    print(f"Matrix (D_{N}x{N}):\n", D_NxN)
    
    evals_NxN = np.linalg.eigvals(D_NxN)
    print(f"Extracted Eigenvalues: {evals_NxN}\n")
    
    
    # --- 3. TRIANGULAR MATRICES (LOWER & UPPER) ---
    print("--- 3. TRIANGULAR MATRICES ---")
    # Generate a random 4x4 matrix
    np.random.seed(42) # For reproducible pipeline testing
    A = np.random.randint(-5, 5, (4, 4))
    
    # Lower Triangular (Zeros above main diagonal)
    lower_A = np.tril(A)
    print("Lower Triangular Matrix:\n", lower_A)
    print("Eigenvalues (Computed) :", np.linalg.eigvals(lower_A))
    print("Diagonal Elements      :", np.diag(lower_A), "\n")
    
    # Upper Triangular (Zeros below main diagonal)
    upper_A = np.triu(A)
    print("Upper Triangular Matrix:\n", upper_A)
    print("Eigenvalues (Computed) :", np.linalg.eigvals(upper_A))
    print("Diagonal Elements      :", np.diag(upper_A))

if __name__ == "__main__":
    extract_eigenvalues_fast()