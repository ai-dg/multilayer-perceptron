import numpy as np
from custom_layer import BaseLayer

class BaseOptimizer:
    def ft_step(self, layers):
        raise NotImplementedError

class SGD(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def ft_step(self, layers : list[BaseLayer]):
        for layer in layers:
            if not getattr(layer, "trainable", True):
                continue

            for name, grad in layer.grads.items():
                param = layer.params[name]
    
                if grad is None:
                    continue
                layer.params[name] = param - self.lr * grad

class Adam(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr : float = learning_rate
        self.beta1 : float = beta1
        self.beta2 : float = beta2
        self.eps : float = eps
        self.m : dict = {}
        self.v : dict = {}
        self.t : int = 0

    def ft_step(self, layers  : list[BaseLayer]):
        self.t += 1

        for layer in layers:
            if not getattr(layer, "trainable", True):
                continue
            
            for name, grad in layer.grads.items():
                
                key = (id(layer), name)

                if key not in self.m:
                    self.m[key] = np.zeros_like(grad)
                    self.v[key] = np.zeros_like(grad)

                m = self.m[key]
                v = self.v[key]

                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * (grad**2)

                self.m[key] = m
                self.v[key] = v

                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)
                
                result = m_hat / (np.sqrt(v_hat) + self.eps)

                w = layer.params[name]
                layer.params[name] = w - self.lr * result


def main():
    print("Hello, World!")

    optimizer = Adam()
    print("m dict:", optimizer.m)
    print("v dict:", optimizer.v)
    print("t:", optimizer.t)


if __name__ == "__main__":
    main()
