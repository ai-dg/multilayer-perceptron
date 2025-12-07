import numpy as np

class BaseLoss:
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def ft_gradient(self, y_true, y_pred):
        raise NotImplementedError


class BinaryCrossEntropy(BaseLoss):
    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)

        L = -np.mean(
            y_true * np.log(y_pred)
            + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        return L

    def ft_gradient(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n_samples = y_true.shape[0]

        dL_dy_pred = (y_pred - y_true) / (y_pred * (1.0 - y_pred))
        return dL_dy_pred / n_samples


class CategoricalCrossEntropy(BaseLoss):
    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)

        per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
        L = np.mean(per_sample)
        return L

    def ft_gradient(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n_samples = y_true.shape[0]

        dL_dy_pred = -y_true / y_pred
        return dL_dy_pred / n_samples


class MeanSquaredError(BaseLoss):
    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        L = np.mean((y_pred - y_true) ** 2)
        return L

    def ft_gradient(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        n_samples = y_true.shape[0]
        dL_dy_pred = 2.0 * (y_pred - y_true) / n_samples
        return dL_dy_pred


def main():
    loss = BinaryCrossEntropy()
    y_true = np.array([[0.5], [0.8]])
    y_pred = np.array([[0.4], [0.7]])
    print("Loss:", loss(y_true, y_pred))
    print("Grad:\n", loss.ft_gradient(y_true, y_pred))


if __name__ == "__main__":
    main()
