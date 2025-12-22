import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = True
        self.params = {}
        self.grads = {}
        self.dimensions = 0

    def ft_build(self, dimensions: int):
        self.dimensions = dimensions
        # raise NotImplementedError

    def ft_forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ft_backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DenseLayer(BaseLayer):
    def __init__(self, units: int, activation: str = "relu", is_output: bool = False):
        super().__init__()
        self.units = units
        self.activation = activation.lower()
        self.is_output = is_output
        self.X = None
        self.Z = None
        self.A = None
        self.is_built = False

    def ft_build(self, dimensions: int):
        # Kaiming initialization
        limit = np.sqrt(2.0 / dimensions)
        self.params["b"] = np.zeros((1, self.units))
        self.params["W"] = np.random.randn(dimensions, self.units) * limit
        self.is_built = True

    def ft_forward(self, X: np.ndarray) -> np.ndarray:
        if not self.is_built:
            self.ft_build(X.shape[1])
        self.X = X
        W = self.params["W"]
        b = self.params["b"]
        Z = X @ W + b
        self.Z = Z
        A = self.ft_activation_forward(Z)
        self.A = A
        return A

    def ft_backward(self, dA: np.ndarray) -> np.ndarray:
        W = self.params["W"]
        X = self.X
        A = self.A

        if self.activation == "softmax":
            if self.is_output:
                dZ = dA
            else:
                dot = np.sum(dA * A, axis=1, keepdims=True)
                dZ = A * (dA - dot)

        elif self.activation == "sigmoid" and self.is_output:
            dZ = dA
        else:
            dZ = dA * self.ft_activation_backward(A)

        dW = X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ W.T

        self.grads["W"] = dW
        self.grads["b"] = db
        return dX


    def ft_sigmoid(self, Z):
        sigmoid = 1.0 / (1.0 + np.exp(-Z))
        return sigmoid

    def ft_d_sigmoid(self, A):
        d_sigmoid = A * (1.0 - A)
        return d_sigmoid

    def ft_relu(self, Z):
        relu = np.maximum(0, Z)
        return relu

    def ft_d_relu(self, A):
        d_relu = np.where(A > 0, 1.0, 0.0)
        return d_relu

    def ft_softmax(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp = np.exp(Z_shift)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return softmax

    def ft_d_softmax(self, A):
        d_softmax = np.ones_like(A)
        return d_softmax

    def ft_activation_forward(self, Z):
        if self.activation == "sigmoid":
            return self.ft_sigmoid(Z)
        elif self.activation == "relu":
            return self.ft_relu(Z)
        elif self.activation == "softmax":
            return self.ft_softmax(Z)
        else:
            raise ValueError(
                f"Activation function {self.activation} not supported")

    def ft_activation_backward(self, A):
        if self.activation == "sigmoid":
            return self.ft_d_sigmoid(A)
        elif self.activation == "relu":
            return self.ft_d_relu(A)
        elif self.activation == "softmax":
            return None
        else:
            raise ValueError(
                f"Activation function {self.activation} not supported")


def main():
    layer = DenseLayer(units=1, activation="relu")
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    layer.ft_build(X.shape[1])
    output = layer.ft_forward(X)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use("TkAgg")
    # plt.plot(output)
    # plt.savefig("output.png")
    # plt.show()
    # plt.close()
    print("Forward:")
    print(output)
    output = layer.ft_backward(output)
    print("Backward relu")
    print(output)

    # plt.plot(output)
    # plt.show()

    


if __name__ == "__main__":
    main()
