import numpy as np

# Exercice 2.6 — Norms

def lp_norm(x : np.ndarray, p : int):

    if p == np.inf:
        p = np.max(np.abs(x))
        return p
    total_sum = 0.0
    if x.ndim == 1:
        for i in x:
            total_sum += np.abs(i)**p
    elif x.ndim == 2:
        for row in x:
            for i in row:
                total_sum += np.abs(i)**p
    else:
        return
    
    result = total_sum**(1.0/p)

    # decimales = result % 1

    # if decimales >= 0.5:
    #     result = np.round(result, 2)
    # else:
    #     result = np.floor(result)

    # result = np.around(result, 3)

    return result


def normalize_vector(x : np.ndarray):
    
    x_alpha = x

    norm = np.linalg.norm(x, ord=2)

    x_alpha = np.where(norm!=0, x_alpha/norm, x_alpha)

    return x_alpha

def normalize_rows(M: np.ndarray):
    M = np.asarray(M, dtype=float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)   # shape (n,1)
    return np.where(norms != 0, M / norms, M)



def main():

    m1 = np.asarray([ [2,1], [1,2]], np.float32)
    v1 = np.asarray([3,5], np.float32)

    # Exercice 1
    result = np.linalg.norm(m1, ord='fro')
    result2 = lp_norm(m1, 2)

    print("Exercice 1")
    print("Matrix")
    print(f"Result original function {result}")
    print(f"Result copy function {result2}")

    result = np.linalg.norm(v1, 2)
    result2 = lp_norm(v1, 2)

    print("Vector")
    print(f"Result original function {result}")
    print(f"Result copy function {result2}")
    print("---------------------------------------------")

    v2 = np.asarray([1,2,3,4], np.float32)

    # Exercice 2
    l1 = lp_norm(v2, 1)
    l2 = lp_norm(v2, 2)
    l3 = lp_norm(v2, np.inf)

    l1_original = np.linalg.norm(v2, ord=1)
    l2_original = np.linalg.norm(v2, ord=2)
    l3_original = np.linalg.norm(v2, ord=np.inf)
    print("Exercice 2")
    print(f"L1\nOriginal: {l1_original}\n Copy: {l1}")
    print(f"L2\nOriginal: {l2_original}\n Copy: {l2}")
    print(f"L∞\nOriginal: {l3_original}\n Copy: {l3}")
    print("---------------------------------------------")

    # Exercice 3
    m2 = [
        [1, 2, 3],
        [3, 2, 1],
        [5, 4, 2],
    ]

    m2 = np.asarray(m2, np.float32)
    m3 = np.empty((3,3), np.float32)

    for i, row in enumerate(m2):
        m3[i] = normalize_vector(row)

    print("Exercice 3")
    print(f"Matrix normalized: \n{m3}")
    print("---------------------------------------------")

    


if __name__ == "__main__":
    main()