import numpy as np

def main():

    # Exercice 1
    A = [
        [2, 4],
        [3, 5],
    ]

    A = np.asarray(A, np.float32)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Exercice 1")
    if np.allclose(A @ eigenvectors, eigenvalues * eigenvectors, atol=1e-6):
        print("Equality holds A @ V ≈ λ v")
    else:
        print("Equality doesn't hold")

    print("-------------------------------------")

    # Exercice 2
    eigen_diag = np.diag(eigenvalues)
    eigen_vector_inverse = np.linalg.inv(eigenvectors)

    print("Exercice 2")
    if np.allclose(eigenvectors @ eigen_diag @ eigen_vector_inverse, A, atol=1e-6):
        print("Equality holds A ≈ V @ Λ @ V⁻¹")
    else:
        print("Equality doesn't hold")

    print("-------------------------------------")
    
    # Exercice 3
    A = [
        [1, 1, 1],
        [1, 2, 0],
        [1, 0, 1]
    ]

    A = np.asarray(A, np.float32)

    eigvals, eigvecs = np.linalg.eigh(A)

    print("Exercice 3")
    if np.allclose(eigvecs.T @ eigvecs, np.eye(A.shape[1]), atol=1e-6):
        print("Real symmetric matrix")
    else:
        print("Real symmetric matrix doesn't hold")

    print("-------------------------------------")

    # Exercice 4
    A = [
        [1, 2],
        [1, 2]
    ]

    A = np.asarray(A, np.float32)

    eigvals, eigvecs = np.linalg.eig(A)

    print("Exercice 4")
    if np.any(np.isclose(eigvals, 0.0)):
        print("Matrix A is singular (non inversible)")
    else:
        print("Matrix A isn't singular (inversible)")

    print("-------------------------------------")

    # Exercice 5
    A = [
        [1, 3],
        [3, 2],
    ]


    eigvals, eigvecs = np.linalg.eigh(A)

    max_eigenvalue = np.max(eigvals)

    max_fx = -np.inf


    # for _ in range(10000):
    #     x = np.random.randn(2).astype(np.float32)
    #     x = x / np.linalg.norm(x)
    #     fx = x.T @ A @ x
    #     if fx > max_fx:
    #         max_fx = fx

    X = np.random.randn(10000, 2).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fx_all = np.einsum("ij,ij->i", X @ A, X)
    max_fx = np.max(fx_all)

    print("Max f(x) ≈", max_fx)

    if np.isclose(max_fx, max_eigenvalue, rtol=1e-2):
        print("max(f(x)) ≈ max(λ) → checked")
    else:
        print("Difference so big")



if __name__ == "__main__":
    main()