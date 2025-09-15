import numpy as np
import matplotlib.pyplot as plt

# Non-diagonalizable matrix (Jordan block)
A = np.array([
    [5, 1],
    [0, 5]
], dtype=np.float32)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

# Unit circle (input)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape (2, N)

# Transformation by matrix A
ellipse = A @ circle

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Jordan Decomposition: Non-diagonalizable Matrix")

# Plot the unit circle
ax.plot(circle[0], circle[1], 'gray', label='Unit circle (input)')

# Plot the transformed ellipse
ax.plot(ellipse[0], ellipse[1], 'red', label='A · circle (ellipse)')

# Main eigenvector
ax.quiver(0, 0, eigvecs[0, 0], eigvecs[1, 0],
          angles='xy', scale_units='xy', scale=1,
          color='blue', label='Real eigenvector')

# Columns of A: effect on canonical basis
ax.quiver(0, 0, A[0, 0], A[1, 0],
          color='orange', angles='xy', scale_units='xy', scale=1, label='A·[1, 0]')
ax.quiver(0, 0, A[0, 1], A[1, 1],
          color='purple', angles='xy', scale_units='xy', scale=1, label='A·[0, 1]')

# Legend
ax.legend()
plt.show()
