import numpy as np
import matplotlib.pyplot as plt

# Matrix A
A = np.array([
    [3, 1],
    [1, 3]
], dtype=np.float32)

# SVD decomposition
U, S, Vt = np.linalg.svd(A)

# Build Σ (Sigma) with the singular values
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, S)

# Display matrices in the console
np.set_printoptions(precision=3, suppress=True)
print("✅ Matrix A:\n", A)
print("\n🔹 Matrix U (u₁ and u₂ vectors as columns):\n", U)
print("\n🔹 Singular values Σ (diagonal form):\n", Sigma)
print("\n🔹 Matrix Vᵀ:\n", Vt)

# Create the unit circle (input)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape (2, 100)

# Steps of the transformation via SVD
step1 = Vt @ circle         # initial rotation (Vᵀ)
step2 = Sigma @ step1       # stretching (Σ)
step3 = U @ step2           # final rotation (U) → transformed ellipse

# Plotting
plt.figure(figsize=(7, 7))
plt.plot(circle[0], circle[1], 'gray', label="Unit circle (input)")
plt.plot(step3[0], step3[1], 'red', label="Transformed ellipse (A · circle)")

# Singular vectors (U)
plt.quiver(0, 0, U[0, 0], U[1, 0], scale=1, scale_units='xy', angles='xy',
           color='blue', label='u₁ (column 1 of U)')
plt.quiver(0, 0, U[0, 1], U[1, 1], scale=1, scale_units='xy', angles='xy',
           color='green', label='u₂ (column 2 of U)')

# Also plot the original columns of A
plt.quiver(0, 0, A[0, 0], A[1, 0], color='purple', angles='xy', scale_units='xy', scale=1, label='column 1 of A')
plt.quiver(0, 0, A[0, 1], A[1, 1], color='orange', angles='xy', scale_units='xy', scale=1, label='column 2 of A')

# Formatting
plt.gca().set_aspect('equal')
plt.title("SVD Decomposition: A = U · Σ · Vᵀ")
plt.grid(True)
plt.legend()
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Annotate matrix A in the corner of the plot
text_A = f"A =\n[{A[0,0]:.0f}  {A[0,1]:.0f}]\n[{A[1,0]:.0f}  {A[1,1]:.0f}]"
plt.text(-3.8, 3.5, text_A, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))

plt.show()
