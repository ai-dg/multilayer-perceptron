import numpy as np


class BaseMetric:
    def __call__(self, y_true, y_pred) -> float:
        return self.ft_evaluate(y_true, y_pred)

    def ft_evaluate(self, y_true, y_pred) -> float:
        raise NotImplementedError


def ft_y_to_binary_or_labels(y):
    if y.ndim == 2 and y.shape[1] > 1:
        if y.shape[1] == 2:
            y = y[:, 0].astype(int)
            return y
        y = np.argmax(y, axis=1).astype(int)
        return y
    y = y.reshape(-1)
    if np.all(np.isclose(y, np.round(y))):
        y = y.astype(int)
        return y
    y = (y >= 0.5).astype(int)
    return y


class Accuracy(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = ft_y_to_binary_or_labels(y_true)
        y_pred = ft_y_to_binary_or_labels(y_pred)
        accuracy = float(np.mean(y_true == y_pred))
        return accuracy


class Precision(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = ft_y_to_binary_or_labels(y_true)
        y_pred = ft_y_to_binary_or_labels(y_pred)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        if TP + FP == 0:
            return 0.0
        precision = float(TP / (TP + FP))
        return precision


class Recall(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = ft_y_to_binary_or_labels(y_true)
        y_pred = ft_y_to_binary_or_labels(y_pred)
        TP = np.sum((y_pred == 1) & (y_pred == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        if TP + FN == 0:
            return 0.0
        recall = float(TP / (TP + FN))
        return recall


class F1Score(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        precision_class = Precision()
        recall_class = Recall()
        precision = precision_class.ft_evaluate(y_true, y_pred)
        recall = recall_class.ft_evaluate(y_true, y_pred)
        if (recall + precision) == 0:
            return 0.0
        F1 = float(2 * ((precision * recall) / (precision + recall)))
        return F1


def main():
    accuracy = Accuracy()
    precision = Precision()
    recall = Recall()
    f1_score = F1Score()
    y_true = np.array([[0, 1], [1, 0]])
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1]])

    print("accuracy =", accuracy.ft_evaluate(y_true, y_pred))
    print("precision =", precision.ft_evaluate(y_true, y_pred))
    print("recall =", recall.ft_evaluate(y_true, y_pred))
    print("f1_score =", f1_score.ft_evaluate(y_true, y_pred))


if __name__ == "__main__":
    main()
