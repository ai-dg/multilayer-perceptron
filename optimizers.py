import numpy as np


class BaseOptimizer:
    """
    Optimizer de base.
    Tous les optimizers doivent implémenter ft_step(layers).
    """

    def ft_step(self, layers):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def ft_step(self, layers):
        """
        Met à jour tous les paramètres trainables de tous les layers.
        On suppose que chaque layer a :
          - layer.trainable : bool
          - layer.params : dict{name -> np.ndarray}
          - layer.grads  : dict{name -> np.ndarray}
        """
        for layer in layers:
            if not getattr(layer, "trainable", True):
                continue

            for name, param in layer.params.items():
                grad = layer.grads.get(name)
                if grad is None:
                    continue
                layer.params[name] = param - self.lr * grad


class Adam(BaseOptimizer):
    """
    Implémentation de l'optimizer Adam.

    Formules :
      m_t = β₁ m_{t-1} + (1 - β₁) g_t
      v_t = β₂ v_{t-1} + (1 - β₂) g_t²

      m̂_t = m_t / (1 - β₁ᵗ)
      v̂_t = v_t / (1 - β₂ᵗ)

      θ := θ - lr * m̂_t / (sqrt(v̂_t) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}

        self.t = 0

    def _get_key(self, layer, name: str):
        return (id(layer), name)

    def ft_step(self, layers):
        self.t += 1

        for layer in layers:
            if not getattr(layer, "trainable", True):
                continue

            for name, param in layer.params.items():
                grad = layer.grads.get(name)
                if grad is None:
                    continue

                key = self._get_key(layer, name)

                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                m = self.m[key]
                v = self.v[key]

                
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)


                self.m[key] = m
                self.v[key] = v

                
                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)

                
                layer.params[name] = param - self.lr * m_hat / (
                    np.sqrt(v_hat) + self.eps
                )


def main():
    print("Hello, World!")

    optimizer = Adam()
    print("m dict:", optimizer.m)
    print("v dict:", optimizer.v)
    print("t:", optimizer.t)


if __name__ == "__main__":
    main()
