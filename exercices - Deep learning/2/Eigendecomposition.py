import numpy as np
import matplotlib.pyplot as plt

# Matrix A (must be diagonalizable)
A = np.array([
    [4, 1],
    [2, 3]
], dtype=np.float32)

# Compute eigenvalues (λ) and eigenvectors (v)
eigvals, eigvecs = np.linalg.eig(A)

# Generate a unit circle (starting basis)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape: (2, 100)

# Transform the circle by matrix A
ellipse = A @ circle

# Apply A to the canonical basis: [1,0] and [0,1]
origin = np.array([[0], [0]])
x_transformed = A @ np.array([[1], [0]])
y_transformed = A @ np.array([[0], [1]])

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Unit circle (input)
ax.plot(circle[0], circle[1], 'gray', label="Unit circle (input)")

# Ellipse transformed by A
ax.plot(ellipse[0], ellipse[1], 'r', label="A · circle")

# Eigenvectors (in blue)
for i in range(len(eigvals)):
    vec = eigvecs[:, i]
    ax.quiver(0, 0, vec[0], vec[1],
              angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.015, label=f'eigenvector v{i+1}')

# Effect of A on the canonical basis
ax.quiver(0, 0, x_transformed[0], x_transformed[1],
          angles='xy', scale_units='xy', scale=1,
          color='orange', width=0.015, label='A·[1, 0]')
ax.quiver(0, 0, y_transformed[0], y_transformed[1],
          angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.015, label='A·[0, 1]')

# Formatting
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Eigendecomposition: A = V · Λ · V⁻¹")
ax.legend()
plt.show()
