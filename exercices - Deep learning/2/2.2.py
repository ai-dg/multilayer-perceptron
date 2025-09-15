import numpy as np

# Exercice 2.2 — Matrix and Vector Operations


def main():

    # Exercice 1

    v1 = np.array([1, 2, 3, 4], np.float32)
    v2 = np.array([4, 3, 2, 1], np.float32)

    v3 = np.add(v1, v2)

    print("Exercice 1")
    print(
        f"Values {v3}, shape {
            np.shape(v3)} dim {
            np.ndim(v3)} type {
                v3.dtype}")
    print("------------------------")

    # Exercice 2
    m1 = [
        [1, 2, 3],
        [3, 2, 1],
    ]

    m2 = [
        [3, 2, 1],
        [1, 2, 3],
    ]

    m1 = np.asarray(m1, np.float32)
    m2 = np.asarray(m2, np.float32)

    m3 = np.multiply(m1, m2)

    print("Exercice 2")
    print(
        f"Values {m3}, shape {
            np.shape(m3)} dim {
            np.ndim(m3)} type {
                m3.dtype}")
    print("------------------------")

    # Exercice 3

    v4 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)

    mean = np.mean(v4)

    v4_centered = v4 - mean

    print("Exercice 3")
    print(f"Mean {mean}")
    print(f"Centered vector: {v4_centered}")
    print("------------------------")

    # Exercice 4

    sum_centered = np.sum(v4_centered)

    print("Exercice 4")
    print(f"Sum {sum_centered}")
    print("------------------------")

    try:
        assert np.abs(sum_centered) < 1e-6
    except AssertionError:
        print("Vector not centered")


if __name__ == "__main__":
    main()
