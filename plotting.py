import matplotlib.pyplot as plt
from callbacks import History


def ft_plot_loss(history, out_path="loss_curve.png"):
    h = history.history

    plt.figure()
    plt.plot(h.get("loss", []), label="train_loss")
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def ft_plot_accuracy(history, out_path="accuracy_curve.png"):
    h = history.history

    acc = h.get("Accuracy", h.get("accuracy", None))
    val_acc = h.get("val_Accuracy", h.get("val_accuracy", None))

    if acc is None:
        print("⚠️ No accuracy found in history.")
        return

    plt.figure()
    plt.plot(acc, label="train_accuracy")

    if val_acc is not None:
        plt.plot(val_acc, label="val_accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def ft_plot_learning_curves(history: History, out_prefix: str = "mlp"):
    h = history.history

    plt.figure()
    plt.plot(h.get("loss", []), label="train_loss")
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve - Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_loss.png", dpi=150)

    if "Accuracy" in h or "val_Accuracy" in h or "accuracy" in h or "val_accuracy" in h:
        plt.figure()
        acc = h.get("Accuracy", h.get("accuracy", []))
        val_acc = h.get("val_Accuracy", h.get("val_accuracy", []))
        if acc:
            plt.plot(acc, label="train_accuracy")
        if val_acc:
            plt.plot(val_acc, label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Learning curve - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{out_prefix}_accuracy.png", dpi=150)

    plt.close("all")

def ft_plot_all(history, prefix="mlp"):
    ft_plot_loss(history, f"{prefix}_loss.png")
    ft_plot_accuracy(history, f"{prefix}_accuracy.png")


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
