import numpy as np
from scipy.linalg import lu
import matplotlib.pyplot as plt

# Square matrix A
A = np.array([
    [2, 3, 1],
    [4, 7, 7],
    [6, 18, 22]
], dtype=np.float32)

# LU decomposition via SciPy
P, L, U = lu(A)

# Function to display a matrix as a heatmap
def plot_matrix(M, title, ax):
    cax = ax.matshow(M, cmap='coolwarm')
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(M):
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

# Reconstruction for verification
PA = P @ A
LU = L @ U

# Graphical display
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("LU Decomposition: P·A = L·U", fontsize=16)

plot_matrix(A, "Matrix A", axes[0, 0])
plot_matrix(P, "Matrix P (Permutation)", axes[0, 1])
plot_matrix(L, "Matrix L (Lower)", axes[1, 0])
plot_matrix(U, "Matrix U (Upper)", axes[1, 1])
plot_matrix(PA, "Product P·A", axes[2, 0])
plot_matrix(LU, "Product L·U", axes[2, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
