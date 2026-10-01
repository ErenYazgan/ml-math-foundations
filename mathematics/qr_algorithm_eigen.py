import numpy as np

def extract_eigenvalues_via_qr(A, num_iterations=1000):
    """
    Extracts all eigenvalues of a matrix using the Iterative QR Algorithm.
    
    Engineering Note:
    Unlike Power Iteration which only finds the dominant eigenvalue, 
    the QR Algorithm extracts ALL eigenvalues. By continuously decomposing 
    a matrix into Orthogonal (Q) and Upper-Triangular (R) components, 
    and then multiplying them in reverse (R * Q), the matrix iteratively 
    converges into an Upper-Triangular form (Schur form). 
    As proven mathematically, the eigenvalues of a triangular matrix 
    rest exactly on its main diagonal, allowing for O(1) extraction 
    after convergence. This is the foundation of modern eigenvalue solvers 
    used in industry libraries (e.g., LAPACK, NumPy, SciPy).
    """
    
    Ak = A.copy()
    
    for _ in range(num_iterations):
       
        Q, R = np.linalg.qr(Ak)
        
        Ak = np.dot(R, Q)
        
    eigenvalues = np.diag(Ak)
    
    return eigenvalues

if __name__ == "__main__":
    # We use a symmetric matrix for clean, real-number eigenvalues
    np.random.seed(42)
    A = np.array([[4.0, 1.0, -2.0],
                  [1.0, 5.0, 3.0],
                  [-2.0, 3.0, 6.0]])
                  
    print("--- Eigenvalue Extraction: Custom QR Engine vs NumPy Black-Box ---")
    
    custom_evals = extract_eigenvalues_via_qr(A)

    custom_evals_sorted = np.sort(custom_evals)[::-1]
    print(f"\n[From Scratch QR] Extracted Eigenvalues: \n{custom_evals_sorted}")
    
    numpy_evals = np.linalg.eigvals(A)
    numpy_evals_sorted = np.sort(numpy_evals)[::-1]
    print(f"\n[NumPy eigvals]   Extracted Eigenvalues: \n{numpy_evals_sorted}")
    
    difference = np.abs(custom_evals_sorted - numpy_evals_sorted)
    print(f"\nAbsolute Hardware Error Margin: {difference.max():.2e}")