import numpy as np
import matplotlib.pyplot as plt


def main():
    # Exercice 1
    A = [
        [2, 1],
        [3, 2],
        [7, 5],
    ]

    A = np.asarray(A, np.float32)

    U, S, Vt = np.linalg.svd(A)

    print("Exercice 1")
    print(f"Shape of U: {U.shape}")
    print(f"Shape of S: {S.shape}")
    print(f"Shape of Vt: {Vt.shape}")

    # S_diag = np.diag(S)

    S_diag = np.zeros(A.shape)
    np.fill_diagonal(S_diag, S)

    A_reconstructed = U @ S_diag @ Vt

    print("Exercice 1")
    if np.allclose(A_reconstructed, A, atol=1e-6):
        print("Decomposition SVD holds")
    else:
        print("Decomposition SVD doesn't hold")

    print("--------------------------------------")


    # Exercice 2
    
    A = np.array([
        [2, 1, 0],
        [0, 1, 3],
        [1, 0, 2]
    ], dtype=np.float32)

    U, S, Vt = np.linalg.svd(A)
    sigma_max = S[0]

    max_stretch = 0
    for _ in range(1000):
        x = np.random.randn(3)
        x /= np.linalg.norm(x)

        stretch = np.linalg.norm(A @ x)

        max_stretch = max(max_stretch, stretch)

    print("Exercice 2")
    print("Singular max value:", sigma_max)
    print("Max strechtching of A·x :", max_stretch)
    print("≈ Equality ?", np.isclose(sigma_max, max_stretch, atol=1e-2))
    print("--------------------------------------")

    # Exercice 3

    k = 1

    # mxk
    U_k = U[:, :k]
    # kxk
    S_k = np.diag(S[:k])
    # kxn
    Vt_k = Vt[:k, :]


    A_k = U_k @ S_k @ Vt_k


    print("Exercice 3")
    print(f"Original matrix: \n{A}")
    print(f"A_k: {A_k}")
    print(f"Error: {np.linalg.norm(A - A_k, ord='fro')}")
    print("--------------------------------------")

    # Exercice 4  
    size_A = np.prod(A.shape)

    print(f"Size A: {size_A}")
    size_Ak = np.prod(U_k.shape) + np.prod(S_k.shape) + np.prod(Vt_k.shape)

    ratio = size_Ak / size_A
    percent = 100 * (1 - ratio)

    print("Exercice 4")
    print(f"Shape of A: \n{size_A}")
    print(f"Shape of A_k: \n{size_Ak}")
    print(f"Ratio compression: \n{percent:.2f} %")
    print("--------------------------------------")


    # Exercice 5
    
    sigma = 0.2
    noise = np.random.normal(scale=sigma, size=A.shape)
    A_noisy = A + noise
    
    U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)
    
    k = 5
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    A_clean = U_k @ S_k @ Vt_k

    error = np.linalg.norm(A_clean - A)
    error_relative = error / np.linalg.norm(A)
    
    print(f"Construction error : {error:.4f}")
    print(f"Relative error: {error_relative}")
    

if __name__ == "__main__":
    main()