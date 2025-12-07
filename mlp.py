import argparse
import numpy as np

from custom_model import CustomSequential
from custom_layer import DenseLayer
from data_processor import DataProcessor
from callbacks import History, EarlyStopping
from plotting import ft_plot_learning_curves, ft_plot_all


def build_model(input_dim: int, args: argparse.Namespace) -> CustomSequential:
    hidden_layers = args.layers or [24, 24]
    if len(hidden_layers) < 2:
        print("⚠️ Le sujet demande au moins 2 hidden layers. On en ajoute automatiquement.")
        while len(hidden_layers) < 2:
            hidden_layers.append(hidden_layers[-1])

    layers = []
    for units in hidden_layers:
        layers.append(DenseLayer(units=units, activation="relu"))

    layers.append(DenseLayer(units=1, activation="sigmoid"))

    model = CustomSequential(layers)
    model.ft_build(input_dim)

    model.ft_compile(
        optimizer=args.optimizer,
        loss="BinaryCrossEntropy",
        metrics=args.metrics,
        learning_rate=args.learning_rate,
    )

    return model


def run_split(args: argparse.Namespace):
    if args.dataset is None:
        raise ValueError("--dataset est obligatoire en mode split")

    dp = DataProcessor(args.dataset)
    X, y = dp.ft_load_dataset()
    X_train, y_train, X_valid, y_valid = dp.ft_train_valid_split(
        X, y, valid_ratio=args.valid_ratio, seed=args.seed
    )
    X_train, X_valid = dp.ft_normalize(X_train, X_valid)

    np.savez(args.train_out, X=X_train, y=y_train)
    np.savez(args.valid_out, X=X_valid, y=y_valid)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")
    print(f"Train saved to {args.train_out}")
    print(f"Valid saved to {args.valid_out}")


def run_train(args: argparse.Namespace):
    if args.train_data is None or args.valid_data is None:
        raise ValueError("--train_data et --valid_data sont obligatoires en mode train")

    train = np.load(args.train_data)
    valid = np.load(args.valid_data)

    X_train, y_train = train["X"], train["y"]
    X_valid, y_valid = valid["X"], valid["y"]

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    model = build_model(input_dim=X_train.shape[1], args=args)

    callbacks = []
    history_cb = History()
    callbacks.append(history_cb)

    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=args.min_delta,
            mode="min",
        ))

    history = model.ft_fit(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.ft_save(args.model_path)
    print(f"> saving model '{args.model_path}' to disk...")

    ft_plot_learning_curves(history, out_prefix=args.curve_prefix)

    print(f"Learning curves saved with prefix '{args.curve_prefix}_*.png'")


def run_predict(args: argparse.Namespace):
    if args.model_path is None:
        raise ValueError("--model_path est obligatoire en mode predict")
    if args.predict_data is None:
        raise ValueError("--predict_data est obligatoire en mode predict")

    data = np.load(args.predict_data)
    X, y = data["X"], data["y"]

    model = CustomSequential()
    model.ft_load(args.model_path)
    model.ft_compile(
        optimizer="Adam",
        loss="BinaryCrossEntropy",
        metrics=["Accuracy"],
        learning_rate=0.001,
    )

    y_pred = model.ft_predict(X)
    loss = float(model.loss(y, y_pred))

    print(f"Binary cross-entropy on dataset: {loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron - 42 MLP")
    parser.add_argument(
        "--mode",
        choices=["split", "train", "predict"],
        required=True,
        help="Which phase to run: split, train, or predict.",
    )

    parser.add_argument("--dataset", type=str,
                        help="Path to original full CSV dataset (with 'diagnosis' column) [mode=split].")
    parser.add_argument("--train_out", type=str, default="./datasets/train.npz",
                        help="Output path for training npz file [mode=split].")
    parser.add_argument("--valid_out", type=str, default="./datasets/valid.npz",
                        help="Output path for validation npz file [mode=split].")
    parser.add_argument("--valid_ratio", type=float, default=0.2,
                        help="Validation ratio for train/valid split [mode=split].")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting [mode=split].")

    parser.add_argument("--train_data", type=str,
                        help="Path to training npz file (X,y) [mode=train].")
    parser.add_argument("--valid_data", type=str,
                        help="Path to validation npz file (X,y) [mode=train].")
    parser.add_argument("--model_path", type=str, default="./models/model.pkl",
                        help="Path to save/load model [mode=train/predict].")

    parser.add_argument("--layers", type=int, nargs="+",
                        help="Hidden layer sizes, e.g. --layers 24 24 24 [mode=train].")
    parser.add_argument("--epochs", type=int, default=70,
                        help="Number of training epochs [mode=train].")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size [mode=train].")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"],
                        help="Optimizer to use [mode=train].")
    parser.add_argument("--learning_rate", type=float, default=0.0314,
                        help="Learning rate [mode=train].")
    parser.add_argument("--metrics", type=str, nargs="*", default=["Accuracy"],
                        help="List of metrics to evaluate during training [mode=train].")

    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping (bonus).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping [mode=train].")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Min improvement for early stopping [mode=train].")

    parser.add_argument("--curve_prefix", type=str, default="mlp",
                        help="Prefix for saved learning curve images [mode=train].")

    parser.add_argument("--predict_data", type=str,
                        help="Path to dataset npz file (X,y) used for prediction [mode=predict].")

    args = parser.parse_args()

    if args.mode == "split":
        run_split(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
