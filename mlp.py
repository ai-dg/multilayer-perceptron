import os
import argparse
import numpy as np

from custom_model import CustomSequential
from custom_layer import DenseLayer
from data_processor import DataProcessor
from callbacks import History, EarlyStopping
from plotting import ft_plot_learning_curves, plot_scatter
from plotting import plot_histogram, plot_boxplot
from plotting import ft_export_history_txt, ft_export_predictions_txt

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",

    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",

    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def ft_build_model(
        input_dim: int,
        args: argparse.Namespace) -> CustomSequential:
    hidden_layers = args.layers or [24, 24]
    if len(hidden_layers) < 2:
        print("The subject asked 2 hidden layers minimum, adding more layers.")
        while len(hidden_layers) < 2:
            hidden_layers.append(hidden_layers[-1])

    layers = []
    for units in hidden_layers:
        layers.append(DenseLayer(units=units, activation="relu"))

    if args.loss == "cce":
        layers.append(DenseLayer(units=2, activation="softmax"))
    else:
        layers.append(DenseLayer(units=1, activation="sigmoid"))

    model = CustomSequential(layers)
    model.ft_build(input_dim)

    model.ft_compile(
        optimizer=args.optimizer,
        loss=args.loss,
        metrics=args.metrics,
        learning_rate=args.learning_rate,
    )

    return model


def ft_run_split(args: argparse.Namespace):
    if args.dataset is None:
        raise ValueError("--dataset est obligatoire en mode split")

    dataprocess = DataProcessor(args.dataset)
    X, y = dataprocess.ft_load_dataset()
    X_train, y_train, X_valid, y_valid = dataprocess.ft_train_valid_split(
        X, y, valid_ratio=args.valid_ratio, seed=args.seed
    )
    X_train = dataprocess.ft_normalize(X_train)
    X_valid = dataprocess.ft_normalize(X_valid)

    np.savez("./datasets/train.npz", X=X_train, y=y_train)
    np.savez("./datasets/valid.npz", X=X_valid, y=y_valid)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")
    print(f"Train saved to {"./datasets/train.npz"}")
    print(f"Valid saved to {"./datasets/valid.npz"}")

    plot_scatter(
        X, y,
        feature_x="radius_mean",
        feature_y="texture_mean",
        feature_names=FEATURE_NAMES
    )
    plot_histogram(X, y, "area_mean", FEATURE_NAMES)
    plot_boxplot(X, y, "concavity_mean", FEATURE_NAMES)


def ft_run_train(args: argparse.Namespace):
    if os.path.exists(
            "./datasets/train.npz") is None or \
                os.path.exists("./datasets/valid.npz") is None:
        raise ValueError("You should split the data before train.")
    if args.loss is None:
        raise ValueError("You should add a loss method: bce, cce or mse")

    train = np.load("./datasets/train.npz")
    valid = np.load("./datasets/valid.npz")

    X_train, y_train = train["X"], train["y"]
    X_valid, y_valid = valid["X"], valid["y"]

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    model = ft_build_model(input_dim=X_train.shape[1], args=args)

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
    ft_export_history_txt(history, args.curve_prefix)

    print(f"Learning curves saved with prefix '{args.curve_prefix}_*.png'")


def ft_run_predict(args: argparse.Namespace):
    if args.model_path is None:
        raise ValueError("--model_path is mandatory for predict mode")
    if args.predict_data is None:
        raise ValueError("--predict_data is mandatory for predict mode")

    if args.predict_data.endswith(".npz"):
        data = np.load(args.predict_data)
        X = data["X"]
        y = data["y"] if "y" in data.files else None

    elif args.predict_data.endswith(".csv"):
        dp = DataProcessor(args.predict_data)
        X, y = dp.ft_load_dataset()
        X = dp.ft_normalize(X)

    else:
        raise ValueError(
            "Unsupported file format for predict_data (use .csv or .npz)")

    model = CustomSequential()
    model.ft_load(args.model_path)

    model.ft_compile(
        optimizer="Adam",
        loss="BinaryCrossEntropy",
        metrics=["Accuracy"],
        learning_rate=0.001,
    )

    y_pred = model.ft_predict(X)

    print("DEBUG y_pred shape:", y_pred.shape)
    print(
        "DEBUG y_pred min/max:",
        float(
            np.min(y_pred)),
        float(
            np.max(y_pred)))

    if y is not None:
        loss, metrics = model.ft_evaluate(X, y)
        print(f"Binary cross-entropy on dataset: {loss:.6f}")
        for name, value in metrics.items():
            print(f"{name} on dataset: {value:.4f}")

        ft_export_predictions_txt(y_pred, y)
    else:
        print("Predictions (first 10 samples):")
        ft_export_predictions_txt(y_pred, y_true=None)


def main():
    parser = argparse.ArgumentParser(
        description="Multilayer Perceptron - 42 MLP")
    parser.add_argument(
        "--mode",
        choices=["split", "train", "predict"],
        required=True,
        help="Which phase to run: split, train, or predict.",
    )
    # **************** MANDATORY ******************#

    # ***** MODE SPLIT ****#
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to original full \
            CSV dataset (with 'diagnosis' column) [mode=split].")
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="Validation ratio for train/valid split [mode=split].")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting [mode=split].")

    # ***** MODE TRAIN ****#
    parser.add_argument("--learning_rate", type=float, default=0.0314,
                        help="Learning rate [mode=train].")
    parser.add_argument(
        "--loss",
        type=str,
        default="cce",
        choices=[
            "cce",
            "bce",
            "mse"],
        help="Calculation method for cost function [mode=train].")
    parser.add_argument("--model_path", type=str, default="./models/model.pkl",
                        help="Path to save/load model [mode=train/predict].")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Hidden layer sizes, e.g. --layers 24 24 24 [mode=train].")
    parser.add_argument("--epochs", type=int, default=70,
                        help="Number of training epochs [mode=train].")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size [mode=train].")
    parser.add_argument(
        "--curve_prefix",
        type=str,
        default="mlp",
        help="Prefix for saved learning curve images [mode=train].")

    # ***** MODE PREDIDCT ****#
    parser.add_argument(
        "--predict_data",
        type=str,
        help="Path to dataset npz file (X,y) \
            used for prediction [mode=predict].")

    # **************** BONUS ******************#

    # ***** MODE TRAIN ****#
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=[
            "SGD",
            "Adam"],
        help="Optimizer to use [mode=train].")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=["Accuracy"],
        help="List of metrics to evaluate during training [mode=train].")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping (bonus).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping [mode=train].")
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Min improvement for early stopping [mode=train].")

    args = parser.parse_args()
    if args.mode == "split":
        ft_run_split(args)
    elif args.mode == "train":
        ft_run_train(args)
    elif args.mode == "predict":
        ft_run_predict(args)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(f"KeyboardInterrupt or error {e}")
