import numpy as np

def main():
    
    # Exercice 1

    A = np.asarray([
        [3, 4],
        [9, 7],
    ], np.float32)

    A_trace = np.trace(A)
    manual_trace = np.sum(np.diag(A))

    print("Exercice 1")
    if np.allclose(manual_trace, A_trace):
        print("Trace holds")
    else:
        print("Trace doesn't hold")

    print("-----------------------------------")



    # Exercice 2
    A = np.asarray([
        [2, 5 ,9],
        [9, 10, 33],
        [3, 41, 32],
    ], np.float32)

    B = np.asarray([
        [4, 2, 1],
        [4, 9, 1],
        [9, 4, 5],
    ], np.float32)

    result1 = np.trace(A) + np.trace(B)
    result2 = np.trace(A + B)

    print("Exercice 2")
    if np.allclose(result1, result2, atol=1e-6):
        print("Equality holds Tr(A + B) = Tr(A) + Tr(B)")
    else:
        print("Equality doesn't hold Tr(A + B) = Tr(A) + Tr(B)")


    print("-----------------------------------")

    # Exercie 3
    A = np.asarray([
        [1,2,4],
        [5,3,6],
    ], np.float32)

    B = np.asarray([
        [2,3],
        [5,4],
        [3,4],
    ], np.float32)

    result1 = np.trace(A @ B)
    result2 = np.trace(B @ A)

    print("Exercice 3")
    if np.allclose(result1, result2):
        print("Equality holds Tr(AB) = Tr(BA)")
    else:
        print("Equality doesn't hold Tr(AB) = Tr(BA)")

    print("-----------------------------------")


    # Exercice 4

    A = np.asarray([
        [3,2],
        [4,9],
    ], np.float32)

    B = np.asarray([
        [4,6],
        [4,2],
    ], np.float32)

    result1 = np.trace(A.T @ B)    
    result2 = np.sum(A * B)

    print("Exercice 4")
    if np.allclose(result1, result2, atol=1e-6):
        print("Equality holds Tr(ATB)=∑Aij⋅Bij")
    else:
        print("Equality doesn't hold Tr(ATB)=∑Aij⋅Bij")

    print("-----------------------------------")


    # Exercice 5

    Q = np.linalg.qr(A)[0]

    result1 = np.trace(Q.T @ A @ Q)
    result2 = np.trace(A)

    print("Exercice 5")
    if np.allclose(result1, result2):
        print("Equality holds Tr(QTAQ) = Tr(A)")
    else:
        print("Equality doesn't hold Tr(QTAQ) = Tr(A)")

    print("-----------------------------------")


if __name__ == "__main__":
    main()