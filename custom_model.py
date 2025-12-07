import numpy as np
from typing import Optional, Sequence

from custom_layer import DenseLayer
from optimizers import SGD, Adam, BaseOptimizer
from losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy, BaseLoss
from metrics import Accuracy, Precision, Recall, F1Score, BaseMetric
from callbacks import History, EarlyStopping, Callback
import pickle as pkl


class CustomSequential:
    def __init__(self, layers: Optional[list[DenseLayer]] = None):
        self.layers: list[DenseLayer] = layers or []
        self.isbuilt: bool = False
        self.optimizer: Optional[BaseOptimizer] = None
        self.loss: Optional[BaseLoss] = None
        self.metrics: list[BaseMetric] = []


    def ft_add(self, layer: DenseLayer) -> None:
        self.layers.append(layer)

    def ft_build(self, input_dim: int) -> None:
        """
        Initialise les poids de chaque DenseLayer à partir de input_dim.
        On propage la dimension de sortie de chaque couche comme dimension
        d'entrée de la suivante.
        """
        current_dim = input_dim
        for layer in self.layers:
            layer.ft_build(current_dim)
            if hasattr(layer, "units"):
                current_dim = layer.units
        self.isbuilt = True

    def ft_compile(
        self,
        optimizer: str,
        loss: str,
        metrics: Optional[list[str]] = None,
        learning_rate: float = 0.001,
    ) -> None:
        
        opt_name = optimizer.lower()
        if opt_name == "sgd":
            self.optimizer = SGD(learning_rate=learning_rate)
        elif opt_name == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

        loss_name = loss.lower()
        if loss_name in ("binarycrossentropy", "binary_crossentropy", "bce"):
            self.loss = BinaryCrossEntropy()
        elif loss_name in ("meansquarederror", "mse", "mean_squared_error"):
            self.loss = MeanSquaredError()
        elif loss_name in ("categoricalcrossentropy", "categorical_crossentropy", "cce"):
            self.loss = CategoricalCrossEntropy()
        else:
            raise ValueError(f"Loss {loss} not supported")

        self.metrics = []
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


    def ft_forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.ft_forward(out)
        return out

    def ft_backward(self, d_out: np.ndarray) -> None:
        grad = d_out
        for layer in reversed(self.layers):
            grad = layer.ft_backward(grad)

    def ft_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> History:
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before calling ft_fit.")

        if not self.isbuilt:
            self.ft_build(X_train.shape[1])

        callbacks = list(callbacks) if callbacks is not None else []
        history_cb: Optional[History] = None
        early_stopping_cb: Optional[EarlyStopping] = None

        for cb in callbacks:
            if isinstance(cb, History):
                history_cb = cb
            if isinstance(cb, EarlyStopping):
                early_stopping_cb = cb

        if history_cb is None:
            history_cb = History()
            callbacks.append(history_cb)
        if early_stopping_cb is None:
            early_stopping_cb = EarlyStopping()
            callbacks.append(early_stopping_cb)

        n_samples = X_train.shape[0]

        for cb in callbacks:
            cb.ft_on_train_begin(logs={})

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for cb in callbacks:
                cb.ft_on_epoch_begin(epoch, logs={})

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.ft_forward(X_batch)

                batch_loss = self.loss(y_batch, y_pred)

                d_out = self.loss.ft_gradient(y_batch, y_pred)

                self.ft_backward(d_out)

                self.optimizer.ft_step(self.layers)

            y_train_pred = self.ft_forward(X_train)
            train_loss = self.loss(y_train, y_train_pred)

            logs: dict[str, float] = {
                "loss": float(train_loss),
            }

            for metric in self.metrics:
                value = metric.ft_evaluate(y_train, y_train_pred)
                metric_name = metric.__class__.__name__
                logs[metric_name] = float(value)

            if X_valid is not None and y_valid is not None:
                y_valid_pred = self.ft_forward(X_valid)
                val_loss = self.loss(y_valid, y_valid_pred)
                logs["val_loss"] = float(val_loss)

                for metric in self.metrics:
                    value = metric.ft_evaluate(y_valid, y_valid_pred)
                    metric_name = metric.__class__.__name__
                    logs["val_" + metric_name] = float(value)

            for cb in callbacks:
                cb.ft_on_epoch_end(epoch, logs)

            
            epoch_str = f"epoch {epoch + 1:02d}/{epochs}"
            log_str = f" - loss: {logs['loss']:.4f}"
            if "val_loss" in logs:
                log_str += f" - val_loss: {logs['val_loss']:.4f}"
            for metric_name, metric_value in logs.items():
                if metric_name not in ["loss", "val_loss"] and not metric_name.startswith("val_"):
                    log_str += f" - {metric_name.lower()}: {metric_value:.4f}"
                elif metric_name.startswith("val_") and metric_name != "val_loss":
                    log_str += f" - val_{metric_name[4:].lower()}: {metric_value:.4f}"
            print(epoch_str + log_str)

            if getattr(early_stopping_cb, "stop_training", False):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        for cb in callbacks:
            cb.ft_on_train_end(logs={})

        return history_cb


    def ft_evaluate(self, X: np.ndarray, y: np.ndarray):
        if self.loss is None:
            raise RuntimeError("Model must be compiled before calling ft_evaluate.")
        y_pred = self.ft_forward(X)
        loss = self.loss(y, y_pred)
        metrics_values = [metric.ft_evaluate(y, y_pred) for metric in self.metrics]
        return loss, metrics_values

    def ft_predict(self, X: np.ndarray) -> np.ndarray:
        return self.ft_forward(X)

    def ft_get_weights(self):
        return [layer.params for layer in self.layers]

    def ft_set_weights(self, weights_list):
        for layer, weights in zip(self.layers, weights_list):
            layer.params = weights
            layer.grads = {}

    def ft_save(self, path: str) -> None:
        """
        Sauvegarde seulement la liste des layers (architecture + poids).
        L'optimizer et les callbacks ne sont pas sauvegardés.
        """
        with open(path, "wb") as f:
            pkl.dump(self.layers, f)

    def ft_load(self, path: str) -> None:
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
