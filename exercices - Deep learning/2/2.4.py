import numpy as np

def main():
    mA = [
        [3,0,0],
        [10,3,4],
        [0,0,3],
    ]

    mA = np.asarray(mA, np.float32)
    mI = np.eye(3, dtype=np.float32)

    product_matrix = mA @ mI

    # A*I == A
    if np.allclose(product_matrix, mA):
        print("Identity condition holds")
    else:
        print("Identity condition doesn't holds")

    try:
        mA_inv = np.linalg.inv(mA)
    except np.linalg.LinAlgError:
        print("Matrix doesn't have inverse, linear dependency")
        exit(1)


    product_inverse = mA @ mA_inv
  
    # A @ A_inv = I
    print(f"Inverse Matrix A:\n {mA_inv}")
    print(f"Product A @ A_inv:\n {product_inverse}")
    if np.allclose(product_inverse, mI):
        print("Matrix product equal to Identity matrix")
    else:
        print("Identity matrix product doesn't hold")    

    

if __name__ == "__main__":
    main()