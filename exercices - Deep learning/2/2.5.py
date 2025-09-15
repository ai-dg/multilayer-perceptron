import numpy as np

# Exercice 2.5 — Linear Dependence and Span

def main():
    
    # Exercice 1
    mA = [
        [1,2,3],
        [2,4,6],
        [7,4,9],
    ]

    mA = np.asarray(mA, np.float32)
    rank = np.linalg.matrix_rank(mA)
    print("Exercice 1")
    if rank < min(mA.shape[0], mA.shape[1]):
        print("Linear dependence")
    else:
        print("Linear independence")

    print("----------------------------")

    # Exercice 2
    mB = [
        [1, 0],
        [0, 1],
    ]

    v1 = [2, 8]
    mB = np.asarray(mB, np.float32)
    v1  = np.asarray(v1, np.float32)
    x_hat = np.empty(mB.shape[0], dtype=np.float32)

    # B @ x ≈ v
    x_hat, *_ = np.linalg.lstsq(mB, v1, rcond=None)
    resid = np.linalg.norm(mB @ x_hat - v1)

    print("Exercice 2")
    print("Residual:", resid)
    if np.allclose(mB @ x_hat, v1, atol=1e-1):
        print("V inside Matrix Span of B")
    else:
        print("V outise Matrix Span of B")

    print("----------------------------")

    

    # Exercice 3
    v2 = np.array([1,0], np.float32)
    v3 = np.array([0,1], np.float32)
    v4 = np.array([1,1], np.float32)
    print("Exercice 3")
    result = np.column_stack((v2,v3,v4))
    rank_result = np.linalg.matrix_rank(result)
    if rank_result < min(result.shape[0], result.shape[1]):
        print("Linear dependence")
    else:
        print("Linear independence")
    c1 = 1
    c2 = 1
    c3 = -1
    combination = c1 * v2 + c2 * v3 + c3 * v4
    print(f"Result: {combination}")
    print("----------------------------")

    U, S, Vt = np.linalg.svd(result.astype(np.float32))
    null_vector = Vt[-1, :]
    print("Coefficients de dépendance:", null_vector)

    try:
        assert np.allclose(combination, 0.0, atol=1e-7)

    except AssertionError:
        print("Combination not null")
        exit(1)


    test = v2 + v3 - v4
    print(test)



if __name__ == "__main__":
    main()