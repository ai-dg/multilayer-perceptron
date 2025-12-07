import numpy as np


class BaseMetric:
    def __call__(self, y_true, y_pred) -> float:
        return self.ft_evaluate(y_true, y_pred)

    def ft_evaluate(self, y_true, y_pred) -> float:
        raise NotImplementedError


def _to_labels_true(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y.reshape(-1)


def _to_labels_pred(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    y_flat = y.reshape(-1)
    return (y_flat >= 0.5).astype(int)


class Accuracy(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = _to_labels_true(y_true)
        y_pred = _to_labels_pred(y_pred)
        return float(np.mean(y_true == y_pred))


class Precision(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = _to_labels_true(y_true)
        y_pred = _to_labels_pred(y_pred)

        classes = np.unique(y_true)
        precisions = []

        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))

            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))

        return float(np.mean(precisions)) if len(precisions) > 0 else 0.0


class Recall(BaseMetric):

    def ft_evaluate(self, y_true, y_pred) -> float:
        y_true = _to_labels_true(y_true)
        y_pred = _to_labels_pred(y_pred)

        classes = np.unique(y_true)
        recalls = []

        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fn = np.sum((y_pred != c) & (y_true == c))

            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))

        return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


class F1Score(BaseMetric):
    def ft_evaluate(self, y_true, y_pred) -> float:
        precision_metric = Precision()
        recall_metric = Recall()

        p = precision_metric.ft_evaluate(y_true, y_pred)
        r = recall_metric.ft_evaluate(y_true, y_pred)

        if p + r == 0:
            return 0.0

        return float(2.0 * p * r / (p + r))


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
