import numpy as np
import matplotlib.pyplot as plt

# 🔷 Choose a NON-orthogonal matrix A
A = np.array([
    [2, 1],
    [1, 2]
], dtype=np.float64)

# QR decomposition
Q, R = np.linalg.qr(A)

# Reconstruction of A
A_rec = Q @ R

# Origin
origin = np.zeros(2)

# Create the plot
plt.figure(figsize=(6, 6))

# Vectors of A (original vectors)
plt.quiver(*origin, *A[:,0], color='red', angles='xy', scale_units='xy', scale=1, label='A1 (original)')
plt.quiver(*origin, *A[:,1], color='orange', angles='xy', scale_units='xy', scale=1, label='A2 (original)')

# Vectors of Q (orthonormal basis)
plt.quiver(*origin, *Q[:,0], color='green', angles='xy', scale_units='xy', scale=1, label='Q1 (orthonormal)')
plt.quiver(*origin, *Q[:,1], color='blue', angles='xy', scale_units='xy', scale=1, label='Q2 (orthonormal)')

# Reconstructed vectors A1', A2' via Q·R
plt.quiver(*origin, *A_rec[:,0], color='cyan', angles='xy', scale_units='xy', scale=1, label="A1 (reconstructed)")
plt.quiver(*origin, *A_rec[:,1], color='magenta', angles='xy', scale_units='xy', scale=1, label="A2 (reconstructed)")

# Set up the plane
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.gca().set_aspect('equal')
plt.title('Visual QR Decomposition: A = Q · R')
plt.legend()
plt.show()
