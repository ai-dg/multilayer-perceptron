import numpy as np
import matplotlib.pyplot as plt

# Générer 100 points 2D autour d'une droite
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = 2 * X[:, 0] + 0.5 * np.random.randn(100)  # Introduire corrélation

# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal')
# plt.title("Nuage de points (avant PCA)")
# plt.show()

def main():

    # Step 1 - Center data
    mean_cols = np.mean(X, axis=0)
    X_centered = X - mean_cols

    # Step 2 - Calculate covariance matrix
    n = X_centered.shape[0]
    X_covariance = (1/n) *  X_centered.T @ X_centered

    # Step 3 - Own vectors decompositions
    eigvals, eigvects = np.linalg.eigh(X_covariance)

    index_desc = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[index_desc]
    eigvects_desc = eigvects[:, index_desc]

    # Step 4 - Dimension reduction (1D)
    u1 = eigvects_desc[:, 0]
    X_proj = X_centered @ u1

    # Step 5 - Visualisation
    X_reconstructed = np.outer(X_proj, u1) + mean_cols

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Données originales")
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], color='red', label="Projection PCA (1D)")
    
    scale = 3
    origin = mean_cols
    v = u1 * scale
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='green', label="u₁ (1ère composante)")

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("PCA manuelle — Réduction à 1D")
    plt.show()

    

    


if __name__ == "__main__":
    main()