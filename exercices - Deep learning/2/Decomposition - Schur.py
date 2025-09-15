import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur

# Real square matrix (non-symmetric example)
A = np.array([
    [5, 4],
    [1, 2]
], dtype=np.float32)

# Schur decomposition: A = Q · T · Qᵀ
T, Q = schur(A)

# Unit circle as input
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # (2, 100)

# Steps: A @ circle = Q @ T @ Qᵀ @ circle
step1 = Q.T @ circle
step2 = T @ step1
step3 = Q @ step2

# Direct result of A @ circle (should be equal to step3)
ellipse = A @ circle

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(circle[0], circle[1], color='gray', label="Unit circle (input)")
ax.plot(ellipse[0], ellipse[1], color='red', label="A · circle (final ellipse)")
ax.plot(step3[0], step3[1], '--', color='green', label="Q·T·Qᵀ · circle (via Schur)")

# Columns of Q (orthonormal basis)
ax.quiver(0, 0, Q[0, 0], Q[1, 0], color='blue', angles='xy', scale_units='xy', scale=1, label='Column 1 of Q')
ax.quiver(0, 0, Q[0, 1], Q[1, 1], color='orange', angles='xy', scale_units='xy', scale=1, label='Column 2 of Q')

# Columns of A (to understand its transformation)
ax.quiver(0, 0, A[0, 0], A[1, 0], color='purple', angles='xy', scale_units='xy', scale=1, label='Column 1 of A', width=0.01)
ax.quiver(0, 0, A[0, 1], A[1, 1], color='brown', angles='xy', scale_units='xy', scale=1, label='Column 2 of A', width=0.01)

# Formatting
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Schur decomposition: A = Q · T · Qᵀ")
ax.legend()
plt.show()
