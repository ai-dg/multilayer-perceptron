import numpy as np
import pickle as pkl
from typing import Optional, Sequence
from custom_layer import DenseLayer
from optimizers import SGD, Adam, BaseOptimizer
from losses import BinaryCrossEntropy, MeanSquaredError
from losses import CategoricalCrossEntropy, BaseLoss
from metrics import Accuracy, Precision, Recall, F1Score, BaseMetric
from callbacks import History, EarlyStopping, Callback


class CustomSequential:
    def __init__(self, layers: Optional[list[DenseLayer]] = None):
        self.layers: list[DenseLayer] = layers or []
        self.isbuilt: bool = False
        self.optimizer: Optional[BaseOptimizer] = None
        self.loss: Optional[BaseLoss] = None
        self.metrics: list[BaseMetric] = []
        self.history: Optional[History] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.callbacks: list[Callback] = []

    def ft_add(self, layer: DenseLayer):
        self.layers.append(layer)

    def ft_build(self, num_features: int):
        for layer in self.layers:
            layer.ft_build(num_features)
            # Hidden and output layers have to be neurons dim
            if hasattr(layer, "units"):
                num_features = layer.units
        self.isbuilt = True

    def ft_choose_optimizer(self, optimizer: str, learning_rate: float):
        optimizer = optimizer.lower()
        if optimizer == "sgd":
            self.optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

    def ft_choose_loss(self, loss: str):
        loss = loss.lower()
        if loss in ("binarycrossentropy", "binary_crossentropy", "bce"):
            self.loss = BinaryCrossEntropy()
        elif loss in ("categoricalcrossentropy",
                      "categorical_crossentropy", "cce"):
            self.loss = CategoricalCrossEntropy()
        elif loss in ("meansquarederror", "mse", "mean_squared_error"):
            self.loss = MeanSquaredError()
        else:
            raise ValueError(f"Loss {loss} not supported")

    def ft_choose_metrics(self, metrics: Optional[list[str]]):
        metrics = metrics or []
        for name in metrics:
            m_name = name.lower()
            if m_name == "accuracy":
                self.metrics.append(Accuracy())
            elif m_name == "precision":
                self.metrics.append(Precision())
            elif m_name == "recall":
                self.metrics.append(Recall())
            elif m_name in ("f1", "f1score", "f1_score"):
                self.metrics.append(F1Score())
            else:
                raise ValueError(f"Metric {name} not supported")

    def ft_compile(
        self,
        optimizer: str,
        loss: str,
        metrics: Optional[list[str]] = None,
        learning_rate: float = 0.001,
    ):

        self.layers[-1].is_output = True
        self.ft_choose_optimizer(optimizer, learning_rate)
        self.ft_choose_loss(loss)
        self.ft_choose_metrics(metrics)

    def ft_forward(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.ft_forward(X)
        return X

    def ft_backward(self, dA: np.ndarray):
        for layer in reversed(self.layers):
            dA = layer.ft_backward(dA)
        return dA

    def ft_callbacks_traitement(self,
                                callbacks: Optional[Sequence["Callback"]]):
        self.callbacks = list(callbacks) if callbacks is not None else []

        histories = [cb for cb in self.callbacks if isinstance(cb, History)]
        early_stops = [
            cb for cb in self.callbacks if isinstance(
                cb, EarlyStopping)]

        if len(histories) > 1:
            raise ValueError("Only one History callback is allowed.")
        if len(early_stops) > 1:
            raise ValueError("Only one EarlyStopping callback is allowed.")

        if histories:
            self.history = histories[0]
        else:
            self.history = History()
            self.callbacks.append(self.history)

        if early_stops:
            self.early_stopping = early_stops[0]
        # else:
        #     self.early_stopping = EarlyStopping()
        #     self.callbacks.append(self.early_stopping)

    def ft_compute_logs(self, X_train, y_train, X_valid=None, y_valid=None):
        y_train_pred = self.ft_forward(X_train)
        logs: dict[str, float] = {"loss": float(
            self.loss(y_train, y_train_pred))}

        for metric in self.metrics:
            name = metric.__class__.__name__.lower()
            logs[name] = float(metric.ft_evaluate(y_train, y_train_pred))

        if X_valid is not None and y_valid is not None:
            y_valid_pred = self.ft_forward(X_valid)
            logs["val_loss"] = float(self.loss(y_valid, y_valid_pred))
            for metric in self.metrics:
                name = metric.__class__.__name__.lower()
                logs["val_" +
                     name] = float(metric.ft_evaluate(y_valid, y_valid_pred))

        return logs

    def ft_format_logs(self, epoch: int, epochs: int, logs: dict[str, float]):
        parts = [f"epoch {epoch + 1:02d}/{epochs:02d}"]
        for k in ["loss", "val_loss"]:
            if k in logs:
                parts.append(f"{k}: {logs[k]:.4f}")

        for k, v in logs.items():
            if k not in ("loss", "val_loss"):
                parts.append(f"{k.lower()}: {v:.4f}")

        return " - ".join(parts)

    def ft_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        callbacks: Optional[Sequence[Callback]] = None,
    ):
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before calling ft_fit.")

        if not self.isbuilt:
            self.ft_build(X_train.shape[1])

        self.ft_callbacks_traitement(callbacks)

        N = X_train.shape[0]
        for cb in self.callbacks:
            cb.ft_on_train_begin(logs={})

        for epoch in range(epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            for cb in self.callbacks:
                cb.ft_on_epoch_begin(epoch, logs={})

            for start in range(0, N, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.ft_forward(X_batch)
                self.loss(y_batch, y_pred)
                dA = self.loss.ft_gradient(y_batch, y_pred)
                self.ft_backward(dA)
                self.optimizer.ft_step(self.layers)

            logs = self.ft_compute_logs(X_train, y_train, X_valid, y_valid)

            for cb in self.callbacks:
                cb.ft_on_epoch_end(epoch, logs)

            print(self.ft_format_logs(epoch, epochs, logs))

            if self.early_stopping is not None and \
                    self.early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        for cb in self.callbacks:
            cb.ft_on_train_end(logs={})

        return self.history

    def ft_evaluate(self, X: np.ndarray, y: np.ndarray):
        if self.loss is None:
            raise RuntimeError(
                "Model must be compiled before calling ft_evaluate.")

        y_pred = self.ft_forward(X)
        loss_value = float(self.loss(y, y_pred))

        metrics_dict = {}
        for metric in self.metrics:
            name = getattr(metric, "name", metric.__class__.__name__)
            metrics_dict[name] = float(metric.ft_evaluate(y, y_pred))

        return loss_value, metrics_dict

    def ft_predict(self, X: np.ndarray):
        y_pred = self.ft_forward(X)
        return y_pred

    def ft_get_weights(self):
        params = []
        for layer in self.layers:
            params.append(layer.params)
        return params

    def ft_set_weights(self, weights_list):
        for layer, weights in zip(self.layers, weights_list):
            layer.params = weights
            layer.grads = {}

    def ft_save(self, path: str):
        with open(path, "wb") as f:
            pkl.dump(self.layers, f)

    def ft_load(self, path: str):
        with open(path, "rb") as f:
            self.layers = pkl.load(f)
        self.isbuilt = True


def main():
    print("Hello, World!")

    model = CustomSequential([
        DenseLayer(units=3, activation="relu"),
        DenseLayer(units=2, activation="sigmoid"),
    ])

    model.ft_build(4)
    print("Model built with 2 layers.")


if __name__ == "__main__":
    main()
