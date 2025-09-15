import numpy as np


def main():

    # Exercice 1
    A = np.asarray([
        [3, 5],
        [0, 4],
    ], np.float32)

    detA = np.linalg.det(A)
    trilA = np.tril(A)
    triuA = np.triu(A)
    diagA = np.prod(np.diag(A))

    print("Exercice 1")
    print(f"Determinant A:\n{detA}")
    print(f"Lower triangle:\n {trilA}")
    print(f"Upper triangle:\n{triuA}")
    print(f"Diagonal of A: \n {diagA}")
    print("------------------------------------")

    # Exercice 2
    A = np.asarray([
        [1, 2, 3],
        [2, 4, 6],
        [4, 8, 12],
    ], np.float32)

    detA = np.linalg.det(A)

    print("Exercice 2")
    print(f"Determinant of A:\n {detA}")
    if  detA > 0:
        print("Matrix inversible")
    else:
        print("Matrix not inversible")

    print("------------------------------------")

    # Exercice 3

    A = np.asarray([
        [3, 7, 5],
        [5, 9, 2],
        [3, 4, 6],
    ], np.float32)

    A_swapped = np.copy(A)
    A_swapped[[0, 2]] = A_swapped[[2, 0]]
    detA = np.linalg.det(A)
    detA_swapped = np.linalg.det(A_swapped)

    print("Exercice 3")
    print(f"Determinant of A:\n {detA}")
    print(f"Determinant of A swapped:\n {detA_swapped}")
    
    if detA == -detA_swapped:
        print("Determinant sign exchange holds")
    else:
        print("Determinant sign exchange doesn't hold")
    
    print("------------------------------------")

    # Exercice 4

    coefficient = 2

    A_scaled = np.copy(A)

    A_scaled[0] = A_scaled[0] * coefficient

    detA_scaled = np.linalg.det(A_scaled)

    detA_expected = coefficient * np.linalg.det(A)

    print("Exercice 4")
    print(f"Determinant A scaled:\n {detA_scaled}")
    print(f"Determinant A not scaled:\n{detA_expected}")

    if np.equal(detA_scaled, detA_expected):
        print("Determinants have similar values")
    else:
        print("Determinants haven't similar values")

    print("------------------------------------")

    # Exercice 5

    B = np.asarray([
        [3, 4, 2],
        [4, 6, 7],
        [4, 5, 2],
    ], np.float32)

    detAB = np.linalg.det(A) * np.linalg.det(B)
    detA_B = np.linalg.det(A @ B)

    print("Exercice 5")
    if np.allclose(detAB, detA_B):
        print("Equality holds")
    else:
        print("Equality doesn't hold")

    print("------------------------------------")



if __name__ == "__main__":
    main()