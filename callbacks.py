
class Callback:
    def ft_on_train_begin(self, logs: dict | None = None):
        pass

    def ft_on_train_end(self, logs: dict | None = None):
        pass

    def ft_on_epoch_begin(self, epoch: int, logs: dict | None = None):
        pass

    def ft_on_epoch_end(self, epoch: int, logs: dict | None = None):
        pass


class History(Callback):
    def __init__(self):
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

    def ft_clear_history(self):
        for k in list(self.history.keys()):
            self.history[k].clear()

    def ft_on_train_begin(self, logs: dict | None = None):
        self.ft_clear_history()

    def ft_on_epoch_end(self, epoch: int, logs: dict | None = None):
        if logs is None:
            return
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5,
                 min_delta: float = 0.0, mode: str = "min"):
        if "loss" in monitor and mode != "min":
            raise ValueError(
                f"EarlyStopping: monitor='{monitor} requires mode='min'"
            )
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = None
        self.wait = 0
        self.stop_training = False

    def ft_on_train_begin(self, logs: dict | None = None):
        self.best = None
        self.wait = 0
        self.stop_training = False

    def ft_on_epoch_end(self, epoch: int, logs: dict | None = None):
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None:
            self.best = current
            self.wait = 0
            return

        improved = False
        # Loss
        if self.mode == "min":
            improved = current < self.best - self.min_delta
        # Accuracy
        elif self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


def main():
    history = History()
    print(history.history)
    print(history.ft_on_epoch_end(
        0, {"loss": 0.1, "val_loss": 0.2,
            "accuracy": 0.3, "val_accuracy": 0.4}))
    print(history.history)


if __name__ == "__main__":
    main()
