import numpy as np
import matplotlib.pyplot as plt

def visualize_random_eigenvalues(matrix_size=40, iterations=200):
    """
    Visualizes the eigenvalue distribution of random matrices on a complex plane.
    
    Engineering Note:
    When initializing weights in Deep Neural Networks (DNNs), matrices are populated 
    with random noise. According to Girko's Circular Law, the eigenvalues of such 
    random matrices form a uniform disc in the complex plane. If the spectral radius 
    (the absolute maximum eigenvalue) of this disc is strictly > 1, the network suffers 
    from Exploding Gradients. If it is < 1, it suffers from Vanishing Gradients. 
    This algorithm visually demonstrates the structural boundary of pure computational noise,
    which is the foundational theory behind modern weight initialization techniques 
    (e.g., Xavier/Glorot, He Initialization).
    """
    
    all_eigenvalues = np.zeros(matrix_size * iterations, dtype=complex)
    
    print(f"Generating {iterations} random {matrix_size}x{matrix_size} matrices...")
    
    for i in range(iterations):
       
        A = np.random.randn(matrix_size, matrix_size)
        
        evals = np.linalg.eigvals(A)
        
        start_idx = i * matrix_size
        end_idx = start_idx + matrix_size
        all_eigenvalues[start_idx:end_idx] = evals

    plt.figure(figsize=(8, 8))
    
    plt.scatter(np.real(all_eigenvalues), np.imag(all_eigenvalues), 
                s=5, alpha=0.5, color='teal', edgecolors='black', linewidth=0.2)
    
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.title(f"Eigenvalue Distribution of Random Matrices\n(Girko's Circular Law)", fontsize=14, pad=15)
    plt.xlabel("Real Part", fontsize=12)
    plt.ylabel("Imaginary Part", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal') 
    
    print("Computation complete. Rendering plot...")
    plt.show()

if __name__ == "__main__":
    visualize_random_eigenvalues()