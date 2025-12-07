import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.X_train = None
        self.X_valid = None

    def ft_load_dataset(self):
        df = pd.read_csv(self.csv_path, header=None)

        y = df.iloc[:, 1].to_numpy()
        y = (y == 'M').astype(float)
        
        X = df.iloc[:, 2:].to_numpy(dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def ft_train_valid_split(self, X, y, valid_ratio=0.2, seed=42):
        np.random.seed(seed)
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)

        split = int(n * (1 - valid_ratio))

        train_idx = indices[:split]
        valid_idx = indices[split:]

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        return X_train, y_train, X_valid, y_valid

    def ft_normalize(self, X_train, X_valid):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        std[std == 0] = 1e-8

        X_train_norm = (X_train - mean) / std
        X_valid_norm = (X_valid - mean) / std

        self.X_train = X_train_norm
        self.X_valid = X_valid_norm

        return X_train_norm, X_valid_norm

    def ft_save_split(self, out_train_path, out_valid_path):
        if self.X_train is None or self.X_valid is None:
            raise RuntimeError("Normalize must be called before save_split().")

        np.save(out_train_path, self.X_train)
        np.save(out_valid_path, self.X_valid)

        print(f"Train data saved to {out_train_path}")
        print(f"Valid data saved to {out_valid_path}")


def main():
    data_processor = DataProcessor("./datasets/data.csv")


    X, y = data_processor.ft_load_dataset()

    X_train, y_train, X_valid, y_valid = data_processor.ft_train_valid_split(X, y)


    X_train, X_valid = data_processor.ft_normalize(X_train, X_valid)

    data_processor.ft_save_split("./datasets/train.npy", "./datasets/valid.npy")

    print("Dataset split and saved successfully.")


if __name__ == "__main__":
    main()
