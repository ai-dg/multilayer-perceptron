import numpy as np
from custom_model import CustomSequential
from data_processor import DataProcessor


class CustomPredictor:
    def __init__(self, model_path: str):
        self.model = CustomSequential()
        self.model.ft_load(model_path)

        self.model.ft_compile(
            optimizer="Adam",
            loss="BinaryCrossEntropy",
            metrics=["Accuracy"],
            learning_rate=0.001,
        )

    def ft_predict_file(self, csv_path: str) -> float:
        data_processor = DataProcessor(csv_path)
        X, y = data_processor.ft_load_dataset()
        X_norm, _ = data_processor.ft_normalize(X, X)
        y_pred = self.model.ft_predict(X_norm)
        loss = float(self.model.loss(y, y_pred))
        return loss


def main():
    predictor = CustomPredictor("./models/model.pkl")
    loss = predictor.ft_predict_file("./datasets/data.csv")
    print(f"Final binary cross-entropy on file: {loss:.6f}")


if __name__ == "__main__":
    main()
