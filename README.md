# Machine Learning & Linear Algebra Foundations
> *From Math to Deep Learning*

This repository contains from-scratch Python (NumPy) implementations of the core linear algebra algorithms powering modern Machine Learning and Data Science. 

Instead of relying solely on black-box libraries, this project bridges the gap between theoretical mathematics and computational execution. The primary focus is on algorithmic stability, hardware optimization, and understanding how data is transformed in high-dimensional spaces.

##  Core Implementations & Engineering Concepts

*   **Stable Least-Squares Fitting:** Bypassing the computational instability of classic matrix inversion by implementing Least-Squares models via QR Decomposition (`least_squares_qr.py`, `matrix_inverse_qr.py`).
*   **Algorithmic Optimization for Online Learning:** Utilizing the Sherman-Morrison formula to update inverse matrices dynamically, reducing computational complexity from $O(n^3)$ to $O(n^2)$ (`sherman_morrison_online_learning.py`).
*   **Computational Limits & Floating-Point Precision:** Demonstrating hardware-level floating-point zero errors in singular matrices and comparing pseudo-inverse (`pinv`) mechanics against classical inverse (`inv`) (`matrix_determinant_zero.py`, `matrix_pinv_vs_inv.py`).
*   **Vector Space Mechanics:** Orthogonal projection and decomposing vectors into prediction and residual components (`vector_decomposition.py`).

##  Tech Stack
*   **Language:** Python
*   **Core Library:** NumPy (for low-level matrix operations and broadcasting mechanics)

---
*Built to understand the hardware and mathematical reality behind AI models.*
