import numpy as np

# Exercice 2.3 — Multiplying Matrices and Vectors
def multiply_matrix_vector(A, x):
    A = np.asarray(A, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.ravel()  # (1,n) -> (n,)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.shape[0]:
        raise ValueError("Incompatible shapes for A @ x")
    y = np.empty(A.shape[0], dtype=np.float32)
    for i, row in enumerate(A):
        s = 0.0
        for j, val in enumerate(row):
            s += val * x[j]
        y[i] = s
    return y  # shape (m,)



def main():
    # Exercice 1
    m1 = [
        [1,2,3],
        [3,2,1]
    ]


    v1 = [
        [1,2,3],
    ]

    m1 = np.asarray(m1, np.float32)
    v1 = np.asarray(v1, np.float32)


    if m1.shape[1] != v1.shape[1]:
        print("Shapes not suitable for matrix/vectors products")
        exit(1)

    v2 = multiply_matrix_vector(m1, v1)    
  
    print("Exercice 1")
    print(f"Product result: {v2}")
    print("-------------------------")

    # Exercice 2
    m_A = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    m_B = [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]

    v_x = [
        [1, 1, 1],
    ]

    m_A = np.asarray(m_A, np.float32)
    m_B = np.asarray(m_B, np.float32)
    v_x = np.asarray(v_x, np.float32)


    # (A * B) * x
    result1 = np.matmul(m_A, m_B)
    result1 = multiply_matrix_vector(result1, v_x)

    #  A * (B * x)
    result2 = multiply_matrix_vector(m_B, v_x)
    result2 = np.matmul(m_A, result2)

    print("Exercice 2")
    if np.allclose(result1, result2):
        print("The results are associative")
    else:
        print("The results are not associatives")

    norm_first = np.linalg.norm(result1)
    norm_second = np.linalg.norm(result2)

    

    if np.isclose(norm_first, norm_second) == 0:
        print("The results are equals")
    else:
        print("The results are associatives but not equals")

    print("-------------------------")


    # Exercice 3
    v_y = [
        [2, 2, 2],
    ]

    v_y = np.asarray(v_y, np.float32)

    # A @ (x + y)
    result1 = np.add(v_x, v_y)
    result1 = multiply_matrix_vector(m_A, result1)

    # A @ x + A @ y
    result2 = multiply_matrix_vector(m_A, v_x) + multiply_matrix_vector(m_A, v_y)

    print("Exercice 3")
    if np.allclose(result1, result2):
        print("Distributive property holds")
    else:
        print("Distributive property doesn't holds")

    print("-------------------------")


if __name__ == "__main__":
    main()
