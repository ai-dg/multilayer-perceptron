import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def ft_load_dataset(self):
        dataframe = pd.read_csv(self.csv_path, header=None)

        y = dataframe.iloc[:, 1].to_numpy()

        y = (y == 'M').astype(float)
        X = dataframe.iloc[:, 2:].to_numpy(dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def ft_train_valid_split(self, X, y, valid_ratio=0.2, seed=42):
        np.random.seed(seed)
        N = X.shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)

        split = int(N * (1 - valid_ratio))

        train_idx = indices[:split]
        valid_idx = indices[split:]

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        return X_train, y_train, X_valid, y_valid

    def ft_normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / std
        self.X = X_norm
        return X_norm


def main():
    data_processor = DataProcessor("./datasets/data.csv")
    X, y = data_processor.ft_load_dataset()
    X_train, y_train, X_valid, y_valid = data_processor.ft_train_valid_split(
        X, y)
    X_train = data_processor.ft_normalize(X_train)
    print("Dataset split and saved successfully.")


if __name__ == "__main__":
    main()
