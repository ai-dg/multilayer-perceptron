import numpy as np

# Exercise 2.7 — Special Kinds of Matrices and Vectors

def main():

    # Exercise 1
    mA = [
        [1, 2],
        [2, 1],
    ]

    mA = np.asarray(mA, np.float32)
    mA_T = mA.T

    A_sym = (mA + mA_T) / 2
    A_skew = (mA - mA_T) / 2

    print("Exercise 1: Matrix Symmetry")
    print("Original matrix:\n", mA)
    print("Symmetric matrix (A + A^T)/2:\n", A_sym)
    print("Antisymmetric matrix (A - A^T)/2:\n", A_skew)
    if np.allclose(mA, A_sym, atol=1e-7):
        print("=> The matrix is symmetric.")
    else:
        print("=> The matrix is not symmetric.")

    if np.allclose(mA, A_skew, atol=1e-7):
        print("=> The matrix is antisymmetric.")
    else:
        print("=> The matrix is not antisymmetric.")
    
    print("-------------------------------------------")

    # Exercise 2
    d_vector = [1, 1, 1]
    x_vector = [2, 2, 2]

    d_vector = np.asarray(d_vector, np.float32)
    x_vector = np.asarray(x_vector, np.float32)

    D_vector = np.diag(d_vector)

    result1 = D_vector @ x_vector
    result2 = d_vector * x_vector

    print("Exercise 2")
    if np.allclose(result1, result2):
        print("Equality holds")
    else:
        print("Equality doesn't hold")

    print("-------------------------------------------")

    # Exercise 3
    mA = [
        [1, 1],
        [1, -1]
    ]
    mA = np.asarray(mA, np.float32)

    mQ, mR = np.linalg.qr(mA)

    rank = np.linalg.matrix_rank(mA)
    
    if rank >= mA.shape[1]:
        print("Linearly independent")
    else:
        print("Linearly dependent")

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

    print("-------------------------------------------")

    # Exercice 4
    QI = np.linalg.inv(mQ)

    print("Exercice 4")
    if np.allclose(QI, mQ.T, atol=1e-6):
        print("Orthogonal")
    else:
        print("Not orthogonal")

    print("-------------------------------------------")

    # Exercice 5
    u = np.array([1, 0], dtype=np.float32)
    v = np.array([0, 1], dtype=np.float32)

    u_n = u / np.linalg.norm(u, 2)
    v_n = v / np.linalg.norm(v, 2)

    is_unit_u = np.isclose(np.linalg.norm(u_n, 2), 1.0, atol=1e-7)
    is_unit_v = np.isclose(np.linalg.norm(v_n, 2), 1.0, atol=1e-7)
    is_orth   = np.isclose(u_n @ v_n, 0.0, atol=1e-7)


    print("Exercice 5")

    print("unit(u):", is_unit_u, "unit(v):", is_unit_v, "orthogonal:", is_orth)




    print("-------------------------------------------")

if __name__ == "__main__":
    main()