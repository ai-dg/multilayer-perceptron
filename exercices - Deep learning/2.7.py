import numpy as np

# Exercice 2.7 — Special Kinds of Matrices and Vectors

def main():

    # Exercice 1
    mA = [
        [1, 2],
        [2, 1],
    ]

    mA = np.asarray(mA, np.float32)
    mA_T = mA.T

    A_sym = (mA + mA_T) / 2
    A_skew = (mA - mA_T) / 2

    print("Exercice 1")
    if np.allclose(mA, A_sym):
        print("Symmetric matrix")
    else:
        print("Not symmetric matrix")

    if np.allclose(mA, A_skew):
        print("Skew symmetric matrix")
    else:
        print("Not skew symmetric matrix")
    
    print("-------------------------------------------")

    # Exercice 2
    d_vector = [1, 1, 1]
    x_vector = [2, 2, 2]

    d_vector = np.asarray(d_vector, np.float32)
    x_vector = np.asarray(x_vector, np.float32)

    D_vector = np.diag(d_vector)

    result1 = D_vector @ x_vector
    result2 = d_vector * x_vector

    print("Exercice 2")
    if np.allclose(result1, result2):
        print("Equality holds")
    else:
        print("Equality doesn't hold")

    print("-------------------------------------------")

    # Exercice 3
    mA = [
        [1, 1],
        [1, -1]
    ]
    mA = np.asarray(mA, np.float32)

    mQ, mR = np.linalg.qr(mA, mode='complete')

    rank = np.linalg.matrix_rank(mA)
    
    if rank >= mA.shape[1]:
        print("Linear independant")
    else:
        print("Linear dependent")

    QTQ = mQ.T @ mQ

    if np.allclose(QTQ, np.eye(mQ.shape[1]), atol=1e-6):
        print("Orthonormal")
    else:
        print("Not orthonormal")
    

    R_lower = np.tril(mR, k=-1)
    print(f"Lower:\n{R_lower}")
    if np.allclose(R_lower, 0):
        print("Upper-triangular")
    else:
        print("Not upper-triangular")

    print("Q =\n", mQ)
    print("R =\n", mR)

    # Exercice 4
    

    # Exercice 5

    [...]

if __name__ == "__main__":
    main()