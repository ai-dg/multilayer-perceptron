import numpy as np

# 2.1 Scalars, Vectors, Matrices, and Tensors

def main():
    
    s = np.float32("3.14")

    v = np.array([1,2,3,4,5], np.float32)

    m = np.array([[3,4,5,6], 
                   [5,1,3,4], 
                   [4,5,6,7]], dtype=np.float32)

    t = np.random.rand(2, 3, 4).astype(np.float32)


    print(f"Value: {s} ndim: {np.ndim(s)} shape: {np.shape(s)} type: {np.dtype(s)}")

    print(f"Value: {v} ndim: {np.ndim(v)} shape: {np.shape(v)} type: {v.dtype}")

    print(f"Value: {m} ndim: {np.ndim(m)} shape: {np.shape(m)} type: {m.dtype}")

    print(f"Value: {t} ndim: {np.ndim(t)} shape: {np.shape(t)} type: {t.dtype}")

if __name__ == "__main__":
    main()