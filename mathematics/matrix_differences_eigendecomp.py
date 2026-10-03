import numpy as np

def prove_matrix_polynomial_eigens():
    """
    Bulletproof Hardware Verification:
    Instead of relying on sorting algorithms (which fail when negative eigenvalues 
    are squared and change their absolute order), we test the pure definition:
    (A^2 - AB - BA + B^2) * v = (lambda^2) * v
    """
    size = 4
    np.random.seed(42)
    A = np.random.randn(size, size)
    A = (A + A.T) / 2 
    B = np.random.randn(size, size)
    B = (B + B.T) / 2
    
    C_base = A - B
    evals_base, evecs_base = np.linalg.eig(C_base)
    
    A_squared = np.dot(A, A)
    AB = np.dot(A, B)
    BA = np.dot(B, A)
    B_squared = np.dot(B, B)
    
    C_expanded = A_squared - AB - BA + B_squared
    
    left_side = np.dot(C_expanded, evecs_base)
    
    right_side = evecs_base * (evals_base ** 2)
    
    error = np.max(np.abs(left_side - right_side))
    
    print("--- Hardware Proof: Matrix Polynomial Eigenspace (Fixed) ---\n")
    print(f"Absolute Preservation Error (Should be ~0): {error:.2e}\n")
    
    if error < 1e-10:
        print("Verdict: PROVED. Eigenvectors are completely preserved, eigenvalues are perfectly squared.")

if __name__ == "__main__":
    prove_matrix_polynomial_eigens()