import numpy as np
import matplotlib.pyplot as plt

# Matrix A: symmetric positive definite
A = np.array([
    [4, 2],
    [2, 3]
], dtype=np.float32)

# Cholesky decomposition: A = L · Lᵀ
L = np.linalg.cholesky(A)

# Create a unit circle (standard input)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape (2, 100)

# Transformation by A
ellipse = A @ circle

# Cholesky steps
step1 = L @ circle         # oblique stretching
step2 = L.T @ step1        # full reconstruction

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Base circle
ax.plot(circle[0], circle[1], color='gray', label="Unit circle (input)")

# Full transformation: A @ circle
ax.plot(ellipse[0], ellipse[1], color='red', label="A · circle (final ellipse)")

# Cholesky steps
ax.plot(step2[0], step2[1], '--', color='green', label="L·Lᵀ · circle (via Cholesky)")

# Column vectors of L
ax.quiver(0, 0, L[0, 0], L[1, 0], color='blue', angles='xy', scale_units='xy', scale=1, label='Column 1 of L')
ax.quiver(0, 0, L[0, 1], L[1, 1], color='orange', angles='xy', scale_units='xy', scale=1, label='Column 2 of L')

# Column vectors of A (to see the direct transformation)
ax.quiver(0, 0, A[0, 0], A[1, 0], color='purple', angles='xy', scale_units='xy', scale=1, label='Column 1 of A', width=0.01)
ax.quiver(0, 0, A[0, 1], A[1, 1], color='brown', angles='xy', scale_units='xy', scale=1, label='Column 2 of A', width=0.01)

# Formatting
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Cholesky Decomposition: A = L · Lᵀ (visual)")
ax.legend()
plt.show()
