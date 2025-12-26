import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from callbacks import History


def ft_export_predictions_txt(
        y_pred: np.ndarray,
        y_true=None,
        out_prefix: str = "mlp"):
    os.makedirs("plots", exist_ok=True)

    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        proba_m = y_pred[:, 0].astype(float)
    else:
        proba_m = y_pred.reshape(-1).astype(float)

    pred_label = (proba_m >= 0.5).astype(int)

    rows = []
    if y_true is not None:
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        N = min(len(proba_m), len(y_true))

        acc = float(np.mean(pred_label[:N] == y_true[:N]))

        headers = ["Index", "y_true", "proba_M", "pred_label", "class_pred"]
        rows.append(["-", "SUMMARY", f"acc={acc:.4f}", "-", "-"])

        for i in range(N):
            cls = "Malignant" if pred_label[i] == 1 else "Benign"
            rows.append([i,
                         int(y_true[i]),
                         f"{proba_m[i]:.6f}",
                         int(pred_label[i]),
                         cls])
    else:
        headers = ["Index", "proba_M", "pred_label", "class_pred"]
        for i in range(len(proba_m)):
            cls = "Malignant" if pred_label[i] == 1 else "Benign"
            rows.append([i, f"{proba_m[i]:.6f}", int(pred_label[i]), cls])

    table = tabulate(rows, headers=headers, tablefmt="grid")
    out_path = f"./plots/{out_prefix}_predictions.txt"
    with open(out_path, "w") as f:
        f.write(table)

    print(f"> predictions exported to '{out_path}'")


def ft_export_history_txt(history, out_prefix: str = "mlp"):
    h = history.history

    os.makedirs("plots", exist_ok=True)

    def canonical(k):
        k = k.strip()
        if k.lower().startswith("val_"):
            return "val_" + k[4:].lower()
        return k.lower()

    canon = {}
    for k, v in h.items():
        canon[canonical(k)] = v

    n_epochs = len(next(iter(canon.values())))

    columns = ["epoch"] + sorted(canon.keys())

    table = []
    for i in range(n_epochs):
        row = [i + 1]
        for key in columns[1:]:
            values = canon.get(key, [])
            row.append(f"{values[i]:.6f}" if i < len(values) else "")
        table.append(row)

    txt = tabulate(
        table,
        headers=[c.upper() for c in columns],
        tablefmt="grid",
        floatfmt=".6f"
    )

    out_path = f"./plots/{out_prefix}_history.txt"
    with open(out_path, "w") as f:
        f.write(txt)

    print(f"History exported to {out_path}")


def plot_scatter(X, y, feature_x, feature_y, feature_names):
    ix = feature_names.index(feature_x)
    iy = feature_names.index(feature_y)

    y = y.ravel()

    benign = y == 0
    malignant = y == 1

    plt.figure()
    plt.scatter(X[benign, ix], X[benign, iy], label="Benign", alpha=0.6)
    plt.scatter(X[malignant, ix], X[malignant, iy],
                label="Malignant", alpha=0.6)

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_histogram(X, y, feature, feature_names, bins=30):
    idx = feature_names.index(feature)
    y = y.ravel()

    plt.figure()
    plt.hist(X[y == 0, idx], bins=bins, alpha=0.6, label="Benign")
    plt.hist(X[y == 1, idx], bins=bins, alpha=0.6, label="Malignant")

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_boxplot(X, y, feature, feature_names):
    idx = feature_names.index(feature)
    y = y.ravel()

    data = [
        X[y == 0, idx],
        X[y == 1, idx],
    ]

    plt.figure()
    plt.boxplot(data, labels=["Benign", "Malignant"])
    plt.ylabel(feature)
    plt.title(f"{feature} by class")
    plt.grid(True)
    plt.show()


def ft_has_values(x) -> bool:
    if x is None:
        return False
    try:
        return len(x) > 0
    except TypeError:
        return True


def ft_canonical_key(k: str) -> str:
    k = k.strip()
    if k.lower().startswith("val_"):
        return "val_" + k[4:].lower()
    return k.lower()


def ft_plot_learning_curves(history, out_prefix: str = "mlp"):
    h = history.history

    os.makedirs("plots", exist_ok=True)

    train_loss = h.get("loss", None)
    val_loss = h.get("val_loss", None)

    plt.figure()
    if ft_has_values(train_loss):
        plt.plot(train_loss, label="train_loss")
    if ft_has_values(val_loss):
        plt.plot(val_loss, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve - Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./plots/{out_prefix}_loss.png", dpi=150)
    plt.close()

    canon = {}
    for k, v in h.items():
        ck = ft_canonical_key(k)
        canon[ck] = v

    exclude = {"loss", "val_loss"}

    metric_names = sorted({
        k for k in canon.keys()
        if k not in exclude and not k.startswith("val_")
    })

    if not metric_names:
        return

    plt.figure()
    plotted_any = False

    for name in metric_names:
        train_key = name
        val_key = "val_" + name

        train_vals = canon.get(train_key, None)
        val_vals = canon.get(val_key, None)

        if ft_has_values(train_vals):
            plt.plot(train_vals, label=f"train_{name}")
            plotted_any = True
        if ft_has_values(val_vals):
            plt.plot(val_vals, label=f"val_{name}")
            plotted_any = True

    if plotted_any:
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title("Learning curves - Metrics")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./plots/{out_prefix}_metrics.png", dpi=150)

    plt.close()


def main():

    history = History()
    history.history = {
        "loss": [0.1, 0.2, 0.3],
        "val_loss": [0.1, 0.2, 0.3],
        "Accuracy": [0.1, 0.2, 0.3],
        "val_Accuracy": [0.1, 0.2, 0.3],
    }
    ft_plot_learning_curves(history)


if __name__ == "__main__":
    main()
