import numpy as np


class BaseLayer:
    """BaseLayer class from which all layers inherit
    Callable object to use in types hints. Base object used to
    create complete Layers like DenseLayer:

    * in `__init__()` for initialize:
        * trainable boolean
        * params : dict to store forwards weights and biais
        * grads : dict to store backwards weights and biais

    * in ft_build mandatory function, initialize dimensions variable
    * in ft_forward mandatory function
    * in ft_backward mandatory function
    """

    def __init__(self):
        self.trainable = True
        self.params = {}
        self.grads = {}

    def ft_build(self, dimensions: int):
        self.dimensions = dimensions
        raise NotImplementedError

    def ft_forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ft_backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DenseLayer(BaseLayer):
    """DenseLayer using BaseLayer, to create layers
    Args:
        * units: int -> Quantity of neurones.
        * activation: str -> type of activation `relu`, `sigmoid`, `softmax`.

    Notes:
        * DenseLayer class initialize weights and bias values with Kaiming
        initialization (values limits) and random values - ft_build().
        * Forward method uses linear model as initial params `y = w * x + b`
        but the activation is based of the initial model to transform it in
        `sigmoid`, `softmax` or `relu`.
        * Backward propagation uses dA = 

    """

    def __init__(self, units: int, activation: str = "relu"):
        super().__init__()
        self.units = units
        self.activation = activation.lower()
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
        Z = self.Z
        A = self.A

        d_activation = self.ft_activation_backward(A)

        dZ = dA * d_activation

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
            return self.ft_d_softmax(A)
        else:
            raise ValueError(
                f"Activation function {self.activation} not supported")


def main():
    layer = DenseLayer(units=3, activation="relu")
    layer.ft_build(4)
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    output = layer.ft_forward(X)
    print(output)


if __name__ == "__main__":
    main()
