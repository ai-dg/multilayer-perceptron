import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import BaseMetric, Accuracy, Precision, Recall, F1Score


class TestBaseMetric:
    """Tests pour la classe BaseMetric."""

    def test_base_metric_call_not_implemented(self):
        """Test que __call__ lève NotImplementedError."""
        metric = BaseMetric()
        with pytest.raises(NotImplementedError):
            metric(np.array([1]), np.array([0.5]))


class TestAccuracy:
    """Tests pour Accuracy."""

    def test_accuracy_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        acc = Accuracy()
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1]])
        result = acc.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_accuracy_worst_prediction(self):
        """Test avec pire prédiction possible."""
        acc = Accuracy()
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0, 1], [1, 0]])
        result = acc.ft_evaluate(y_true, y_pred)
        assert result == 0.0

    def test_accuracy_partial_correct(self):
        """Test avec prédiction partiellement correcte."""
        acc = Accuracy()
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [1, 0], [0, 1]])
        result = acc.ft_evaluate(y_true, y_pred)
        assert np.allclose(result, 1.0 / 3.0)

    def test_accuracy_binary_predictions(self):
        """Test avec prédictions binaires (probabilités)."""
        acc = Accuracy()
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        result = acc.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_accuracy_binary_predictions_wrong(self):
        """Test avec prédictions binaires incorrectes."""
        acc = Accuracy()
        y_true = np.array([[1], [0]])
        y_pred = np.array([[0.3], [0.7]])
        result = acc.ft_evaluate(y_true, y_pred)
        assert result == 0.0

    def test_accuracy_call_method(self):
        """Test que __call__ fonctionne."""
        acc = Accuracy()
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1]])
        result = acc(y_true, y_pred)
        assert result == 1.0


class TestPrecision:
    """Tests pour Precision."""

    def test_precision_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        prec = Precision()
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])
        result = prec.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_precision_all_false_positives(self):
        """Test avec tous les positifs prédits incorrects."""
        prec = Precision()
        y_true = np.array([[0, 1], [0, 1], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0], [1, 0]])
        result = prec.ft_evaluate(y_true, y_pred)
        assert result == 0.0

    def test_precision_mixed(self):
        """Test avec cas mixte."""
        prec = Precision()
        y_true = np.array([[1, 0], [1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0], [1, 0]])
        result = prec.ft_evaluate(y_true, y_pred)
        assert result > 0

    def test_precision_no_predictions(self):
        """Test quand aucune prédiction n'est faite pour une classe."""
        prec = Precision()
        y_true = np.array([[1, 0], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1]])
        result = prec.ft_evaluate(y_true, y_pred)
        assert result >= 0


class TestRecall:
    """Tests pour Recall."""

    def test_recall_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        rec = Recall()
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])
        result = rec.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_recall_all_false_negatives(self):
        """Test avec tous les vrais positifs manqués."""
        rec = Recall()
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1], [0, 1]])
        result = rec.ft_evaluate(y_true, y_pred)
        assert result == 0.0

    def test_recall_mixed(self):
        """Test avec cas mixte."""
        rec = Recall()
        y_true = np.array([[1, 0], [1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0], [0, 1]])
        result = rec.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_recall_no_true_positives(self):
        """Test quand il n'y a pas de vrais positifs."""
        rec = Recall()
        y_true = np.array([[0, 1], [0, 1]])
        y_pred = np.array([[0, 1], [0, 1]])
        result = rec.ft_evaluate(y_true, y_pred)
        assert result >= 0


class TestF1Score:
    """Tests pour F1Score."""

    def test_f1_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        f1 = F1Score()
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])
        result = f1.ft_evaluate(y_true, y_pred)
        assert result == 1.0

    def test_f1_worst_prediction(self):
        """Test avec pire prédiction possible."""
        f1 = F1Score()
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1], [0, 1]])
        result = f1.ft_evaluate(y_true, y_pred)
        assert result == 0.0

    def test_f1_balanced(self):
        """Test avec precision et recall équilibrés."""
        f1 = F1Score()
        y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        result = f1.ft_evaluate(y_true, y_pred)
        assert 0 <= result <= 1

    def test_f1_zero_precision_and_recall(self):
        """Test quand precision et recall sont 0."""
        f1 = F1Score()
        y_true = np.array([[1, 0], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1]])
        result = f1.ft_evaluate(y_true, y_pred)
        assert result == 0.0


class TestMetricsIntegration:
    """Tests d'intégration pour les métriques."""

    def test_all_metrics_range(self):
        """Test que toutes les métriques sont entre 0 et 1."""
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        
        acc = Accuracy()
        prec = Precision()
        rec = Recall()
        f1 = F1Score()
        
        assert 0 <= acc.ft_evaluate(y_true, y_pred) <= 1
        assert 0 <= prec.ft_evaluate(y_true, y_pred) <= 1
        assert 0 <= rec.ft_evaluate(y_true, y_pred) <= 1
        assert 0 <= f1.ft_evaluate(y_true, y_pred) <= 1

    def test_metrics_binary_classification(self):
        """Test avec classification binaire."""
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        
        acc = Accuracy()
        prec = Precision()
        rec = Recall()
        f1 = F1Score()
        
        assert acc.ft_evaluate(y_true, y_pred) == 1.0
        assert prec.ft_evaluate(y_true, y_pred) == 1.0
        assert rec.ft_evaluate(y_true, y_pred) == 1.0
        assert f1.ft_evaluate(y_true, y_pred) == 1.0

    def test_metrics_multiclass(self):
        """Test avec classification multi-classes."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        
        acc = Accuracy()
        prec = Precision()
        rec = Recall()
        f1 = F1Score()
        
        assert acc.ft_evaluate(y_true, y_pred) == 1.0
        assert prec.ft_evaluate(y_true, y_pred) == 1.0
        assert rec.ft_evaluate(y_true, y_pred) == 1.0
        assert f1.ft_evaluate(y_true, y_pred) == 1.0

    def test_metrics_consistency(self):
        """Test de cohérence entre les métriques."""
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])
        
        acc = Accuracy()
        prec = Precision()
        rec = Recall()
        f1 = F1Score()
        
        assert acc.ft_evaluate(y_true, y_pred) == 1.0
        assert prec.ft_evaluate(y_true, y_pred) == 1.0
        assert rec.ft_evaluate(y_true, y_pred) == 1.0
        assert f1.ft_evaluate(y_true, y_pred) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

