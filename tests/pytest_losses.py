import sys
import os
import numpy as np
import pytest

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from losses import BaseLoss, BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError


class TestBaseLoss:
    """Tests pour la classe BaseLoss."""

    def test_base_loss_initialization(self):
        """Test de l'initialisation de BaseLoss."""
        loss = BaseLoss()
        assert loss.eps == 1e-15

    def test_base_loss_initialization_custom_eps(self):
        """Test de l'initialisation avec eps personnalisé."""
        loss = BaseLoss(eps=1e-10)
        assert loss.eps == 1e-10

    def test_base_loss_call_not_implemented(self):
        """Test que __call__ lève NotImplementedError."""
        loss = BaseLoss()
        with pytest.raises(NotImplementedError):
            loss(np.array([1]), np.array([0.5]))

    def test_base_loss_gradient_not_implemented(self):
        """Test que ft_gradient lève NotImplementedError."""
        loss = BaseLoss()
        with pytest.raises(NotImplementedError):
            loss.ft_gradient(np.array([1]), np.array([0.5]))


class TestBinaryCrossEntropy:
    """Tests pour BinaryCrossEntropy."""

    def test_bce_initialization(self):
        """Test de l'initialisation."""
        bce = BinaryCrossEntropy()
        assert bce.eps == 1e-15

    def test_bce_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[1.0], [0.0]])
        loss = bce(y_true, y_pred)
        assert loss < 1e-10

    def test_bce_worst_prediction(self):
        """Test avec pire prédiction possible."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.0], [1.0]])
        loss = bce(y_true, y_pred)
        assert loss > 10

    def test_bce_clipping(self):
        """Test que les valeurs sont bien clippées."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.0], [1.0]])
        loss = bce(y_true, y_pred)
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_bce_gradient_shape(self):
        """Test que le gradient a la bonne shape."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0], [1.0]])
        y_pred = np.array([[0.8], [0.2], [0.9]])
        grad = bce.ft_gradient(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_bce_gradient_perfect_prediction(self):
        """Test du gradient avec prédiction parfaite."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[1.0 - 1e-8], [1e-8]])
        grad = bce.ft_gradient(y_true, y_pred)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))
        assert grad.shape == y_true.shape

    def test_bce_gradient_values(self):
        """Test des valeurs du gradient."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.5], [0.5]])
        grad = bce.ft_gradient(y_true, y_pred)
        assert grad[0, 0] < 0
        assert grad[1, 0] > 0

    def test_bce_multiple_samples(self):
        """Test avec plusieurs échantillons."""
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        loss = bce(y_true, y_pred)
        assert isinstance(loss, (float, np.floating))
        assert loss > 0


class TestCategoricalCrossEntropy:
    """Tests pour CategoricalCrossEntropy."""

    def test_cce_initialization(self):
        """Test de l'initialisation."""
        cce = CategoricalCrossEntropy()
        assert cce.eps == 1e-15

    def test_cce_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss = cce(y_true, y_pred)
        assert loss < 1e-10

    def test_cce_worst_prediction(self):
        """Test avec pire prédiction possible."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        loss = cce(y_true, y_pred)
        assert loss > 10

    def test_cce_clipping(self):
        """Test que les valeurs sont bien clippées."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.0, 1.0], [1.0, 0.0]])
        loss = cce(y_true, y_pred)
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_cce_gradient_shape(self):
        """Test que le gradient a la bonne shape."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        y_pred = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        grad = cce.ft_gradient(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_cce_gradient_values(self):
        """Test des valeurs du gradient."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
        grad = cce.ft_gradient(y_true, y_pred)
        assert grad[0, 0] < 0
        assert grad[1, 1] < 0

    def test_cce_three_classes(self):
        """Test avec 3 classes."""
        cce = CategoricalCrossEntropy()
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        loss = cce(y_true, y_pred)
        assert loss > 0
        assert not np.isnan(loss)


class TestMeanSquaredError:
    """Tests pour MeanSquaredError."""

    def test_mse_initialization(self):
        """Test de l'initialisation."""
        mse = MeanSquaredError()
        assert mse.eps == 1e-15

    def test_mse_perfect_prediction(self):
        """Test avec prédiction parfaite."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        loss = mse(y_true, y_pred)
        assert loss == 0.0

    def test_mse_simple_case(self):
        """Test avec un cas simple."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[2.0], [3.0]])
        loss = mse(y_true, y_pred)
        assert np.allclose(loss, 1.0)

    def test_mse_gradient_shape(self):
        """Test que le gradient a la bonne shape."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        grad = mse.ft_gradient(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_mse_gradient_values(self):
        """Test des valeurs du gradient."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[2.0], [1.0]])
        grad = mse.ft_gradient(y_true, y_pred)
        assert np.allclose(grad[0, 0], 1.0)
        assert np.allclose(grad[1, 0], -1.0)

    def test_mse_multiple_samples(self):
        """Test avec plusieurs échantillons."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.1], [2.1], [2.9], [3.9]])
        loss = mse(y_true, y_pred)
        assert loss > 0
        assert not np.isnan(loss)

    def test_mse_multidimensional(self):
        """Test avec prédictions multidimensionnelles."""
        mse = MeanSquaredError()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])
        loss = mse(y_true, y_pred)
        assert np.allclose(loss, 0.01)


class TestLossesIntegration:
    """Tests d'intégration pour les fonctions de perte."""

    def test_all_losses_positive(self):
        """Test que toutes les pertes sont positives (sauf cas parfait)."""
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.7], [0.3]])
        
        bce = BinaryCrossEntropy()
        cce = CategoricalCrossEntropy()
        mse = MeanSquaredError()
        
        loss_bce = bce(y_true, y_pred)
        loss_mse = mse(y_true, y_pred)
        
        assert loss_bce > 0
        assert loss_mse > 0
        
        y_true_cat = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred_cat = np.array([[0.7, 0.3], [0.3, 0.7]])
        loss_cce = cce(y_true_cat, y_pred_cat)
        assert loss_cce > 0

    def test_losses_gradient_consistency(self):
        """Test de cohérence des gradients."""
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.5], [0.5]])
        
        bce = BinaryCrossEntropy()
        mse = MeanSquaredError()
        
        grad_bce = bce.ft_gradient(y_true, y_pred)
        grad_mse = mse.ft_gradient(y_true, y_pred)
        
        assert grad_bce.shape == y_true.shape
        assert grad_mse.shape == y_true.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

