import numpy as np


def main():

    # Exercice 1
    A = np.asarray([
        [1, 2],
        [5, 6],
        [7, 9],
    ], np.float32)

    A_pinv = np.linalg.pinv(A)

    A_reconst = A @ A_pinv @ A
    A_reconst2 = A_pinv @ A @ A_pinv

    print("Exercice 1")
    if np.allclose(A_reconst, A, atol=1e-6):
        print("Matrix A reconstructed, ")
    else:
        print("Matrix not reconstructed")

    if np.allclose(A_reconst2, A_pinv, atol=1e-6):
        print("Matrix A_inv reconstructed")
    else:
        print("Matrix A_inv not reconstructed")
    print("------------------------------------------")


    # Exercice 2
    b = np.asarray([4, 3, 9], np.float32)

    x = A_pinv @ b

    print("Exercice 2")
    if np.allclose(np.dot(A, x), b, atol=1.5):
        print("Equality Ax = b holds")
    else:
        print("Equality Ax = b doesn't hold")

    
    print("Soluttion x: \n", x)
    print("Ax: \n", A @ x)
    print("b: \n", b)
    print("------------------------------------------")


    # Exercice 3

    A1 = np.asarray([
        [3, 5, 6],
        [7, 5, 2],
    ], np.float32)

    A2 = np.asarray([
        [3, 2],
        [5, 3],
        [5, 3],
    ])

    b1 = np.asarray([4, 3], np.float32)
    b2 = np.asarray([5, 2, 7], np.float32)

    x1 = np.linalg.pinv(A1) @ b1
    x2 = np.linalg.pinv(A2) @ b2


    residual1 = np.linalg.norm(np.dot(A1, x1) - b1)
    residual2 = np.linalg.norm(np.dot(A2, x2) - b2)

    print("Exercie 3")
    print("Underdeterminated (A1 : 2x3)")
    print("Solution x1: \n", x1)
    print("Residual ||Ax - b||: \n", residual1)

    print("Overdeterminated (A2 : 3x2)")
    print("Solution x2: \n", x2)
    print("Residual ||Ax - b||: \n", residual2)

    print("------------------------------------------")
    

    # Exerice 4

    A = np.asarray([
        [1, 2],
        [2, 4],
    ], np.float32)

    b = np.asarray([2, 2], np.float32)

    x_pinv = np.linalg.pinv(A) @ b
    x_lstsq = np.linalg.lstsq(A, b)[0]

    norm_x_pinv = np.linalg.norm(x_pinv)
    norm_x_lstsq = np.linalg.norm(x_lstsq)

    print("Exercice 4")
    if np.allclose(norm_x_pinv, norm_x_lstsq, atol=1e-6):
        print("Norms equality holds")
    else:
        print("Norms equality doesn't hold")

    print("------------------------------------------")

    # Exercice 5
    A = np.asarray([
        [1, 3, 4],
        [5, 3, 4],
        [4, 5, 4],
    ], np.float32)

        
    U, S, Vt = np.linalg.svd(A)

    S_inv = np.divide(1, S)

    A_inv = Vt.T @ np.diag(S_inv) @ U.T

    A_inv_original = np.linalg.pinv(A)

    print("Exercice 5")
    if np.allclose(A_inv, A_inv_original, atol=1e-6):
        print("Equality pseudoinverse holds")
    else:
        print("Equality pseudoinvese doesn't hold")
    
    print("------------------------------------------")


if __name__ == "__main__":
    main()