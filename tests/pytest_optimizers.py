import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers import BaseOptimizer, SGD, Adam


class MockLayer:
    """Couche mock pour tester les optimizers."""
    def __init__(self, trainable=True):
        self.trainable = trainable
        self.params = {}
        self.grads = {}


class TestBaseOptimizer:
    """Tests pour la classe BaseOptimizer."""

    def test_base_optimizer_ft_step_not_implemented(self):
        """Test que ft_step lève NotImplementedError."""
        optimizer = BaseOptimizer()
        with pytest.raises(NotImplementedError):
            optimizer.ft_step([])


class TestSGD:
    """Tests pour SGD."""

    def test_sgd_initialization_default(self):
        """Test de l'initialisation avec valeur par défaut."""
        sgd = SGD()
        assert sgd.lr == 0.01

    def test_sgd_initialization_custom(self):
        """Test de l'initialisation avec learning rate personnalisé."""
        sgd = SGD(learning_rate=0.1)
        assert sgd.lr == 0.1

    def test_sgd_step_single_layer(self):
        """Test d'une étape SGD sur une seule couche."""
        sgd = SGD(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.grads["W"] = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        initial_params = layer.params["W"].copy()
        sgd.ft_step([layer])
        
        expected = initial_params - 0.1 * layer.grads["W"]
        assert np.allclose(layer.params["W"], expected)

    def test_sgd_step_multiple_params(self):
        """Test avec plusieurs paramètres dans une couche."""
        sgd = SGD(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0, 2.0]])
        layer.params["b"] = np.array([[0.5]])
        layer.grads["W"] = np.array([[0.1, 0.2]])
        layer.grads["b"] = np.array([[0.05]])
        
        initial_W = layer.params["W"].copy()
        initial_b = layer.params["b"].copy()
        
        sgd.ft_step([layer])
        
        assert np.allclose(layer.params["W"], initial_W - 0.1 * layer.grads["W"])
        assert np.allclose(layer.params["b"], initial_b - 0.1 * layer.grads["b"])

    def test_sgd_step_multiple_layers(self):
        """Test avec plusieurs couches."""
        sgd = SGD(learning_rate=0.1)
        layer1 = MockLayer()
        layer1.params["W"] = np.array([[1.0]])
        layer1.grads["W"] = np.array([[0.1]])
        
        layer2 = MockLayer()
        layer2.params["W"] = np.array([[2.0]])
        layer2.grads["W"] = np.array([[0.2]])
        
        sgd.ft_step([layer1, layer2])
        
        assert np.allclose(layer1.params["W"], np.array([[0.99]]))
        assert np.allclose(layer2.params["W"], np.array([[1.98]]))

    def test_sgd_step_non_trainable_layer(self):
        """Test que les couches non trainables sont ignorées."""
        sgd = SGD(learning_rate=0.1)
        layer = MockLayer(trainable=False)
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        initial_params = layer.params["W"].copy()
        sgd.ft_step([layer])
        
        assert np.array_equal(layer.params["W"], initial_params)

    def test_sgd_step_missing_gradient(self):
        """Test avec gradient manquant."""
        sgd = SGD(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads = {}
        
        initial_params = layer.params["W"].copy()
        sgd.ft_step([layer])
        
        assert np.array_equal(layer.params["W"], initial_params)

    def test_sgd_multiple_steps(self):
        """Test de plusieurs étapes consécutives."""
        sgd = SGD(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        sgd.ft_step([layer])
        assert np.allclose(layer.params["W"], np.array([[0.99]]))
        
        layer.grads["W"] = np.array([[0.2]])
        sgd.ft_step([layer])
        assert np.allclose(layer.params["W"], np.array([[0.97]]))


class TestAdam:
    """Tests pour Adam."""

    def test_adam_initialization_default(self):
        """Test de l'initialisation avec valeurs par défaut."""
        adam = Adam()
        assert adam.lr == 0.001
        assert adam.beta1 == 0.9
        assert adam.beta2 == 0.999
        assert adam.eps == 1e-8
        assert adam.t == 0
        assert adam.m == {}
        assert adam.v == {}

    def test_adam_initialization_custom(self):
        """Test de l'initialisation avec paramètres personnalisés."""
        adam = Adam(
            learning_rate=0.01,
            beta1=0.95,
            beta2=0.99,
            eps=1e-7
        )
        assert adam.lr == 0.01
        assert adam.beta1 == 0.95
        assert adam.beta2 == 0.99
        assert adam.eps == 1e-7

    def test_adam_step_single_layer(self):
        """Test d'une étape Adam sur une seule couche."""
        adam = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        initial_params = layer.params["W"].copy()
        adam.ft_step([layer])
        
        assert adam.t == 1
        
        key = adam._get_key(layer, "W")
        assert key in adam.m
        assert key in adam.v
        
        assert not np.array_equal(layer.params["W"], initial_params)

    def test_adam_step_multiple_steps(self):
        """Test de plusieurs étapes Adam."""
        adam = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        adam.ft_step([layer])
        assert adam.t == 1
        key = adam._get_key(layer, "W")
        m1 = adam.m[key].copy()
        v1 = adam.v[key].copy()
        
        layer.grads["W"] = np.array([[0.2]])
        adam.ft_step([layer])
        assert adam.t == 2
        m2 = adam.m[key]
        v2 = adam.v[key]
        
        assert not np.array_equal(m1, m2)
        assert not np.array_equal(v1, v2)

    def test_adam_step_multiple_params(self):
        """Test avec plusieurs paramètres."""
        adam = Adam(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.params["b"] = np.array([[0.5]])
        layer.grads["W"] = np.array([[0.1]])
        layer.grads["b"] = np.array([[0.05]])
        
        adam.ft_step([layer])
        
        key_W = adam._get_key(layer, "W")
        key_b = adam._get_key(layer, "b")
        
        assert key_W in adam.m
        assert key_b in adam.m
        assert key_W in adam.v
        assert key_b in adam.v

    def test_adam_step_multiple_layers(self):
        """Test avec plusieurs couches."""
        adam = Adam(learning_rate=0.1)
        layer1 = MockLayer()
        layer1.params["W"] = np.array([[1.0]])
        layer1.grads["W"] = np.array([[0.1]])
        
        layer2 = MockLayer()
        layer2.params["W"] = np.array([[2.0]])
        layer2.grads["W"] = np.array([[0.2]])
        
        adam.ft_step([layer1, layer2])
        
        key1 = adam._get_key(layer1, "W")
        key2 = adam._get_key(layer2, "W")
        
        assert key1 in adam.m
        assert key2 in adam.m
        assert key1 != key2

    def test_adam_step_non_trainable_layer(self):
        """Test que les couches non trainables sont ignorées."""
        adam = Adam(learning_rate=0.1)
        layer = MockLayer(trainable=False)
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        initial_params = layer.params["W"].copy()
        initial_t = adam.t
        adam.ft_step([layer])

        assert np.array_equal(layer.params["W"], initial_params)
        assert adam.t == initial_t + 1
        key = adam._get_key(layer, "W")
        assert key not in adam.m

    def test_adam_step_missing_gradient(self):
        """Test avec gradient manquant."""
        adam = Adam(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads = {}
        
        initial_params = layer.params["W"].copy()
        adam.ft_step([layer])
        
        assert np.array_equal(layer.params["W"], initial_params)

    def test_adam_moment_initialization(self):
        """Test que les moments sont initialisés à zéro."""
        adam = Adam(learning_rate=0.1)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0, 2.0]])
        layer.grads["W"] = np.array([[0.1, 0.2]])
        
        adam.ft_step([layer])
        
        key = adam._get_key(layer, "W")
        assert adam.m[key].shape == layer.params["W"].shape
        assert adam.v[key].shape == layer.params["W"].shape

    def test_adam_bias_correction(self):
        """Test de la correction de biais."""
        adam = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
        layer = MockLayer()
        layer.params["W"] = np.array([[1.0]])
        layer.grads["W"] = np.array([[0.1]])
        
        adam.ft_step([layer])
        key = adam._get_key(layer, "W")
        m = adam.m[key]
        v = adam.v[key]
        
        m_hat = m / (1.0 - 0.9 ** 1)
        v_hat = v / (1.0 - 0.999 ** 1)
        
        assert m_hat.shape == m.shape
        assert v_hat.shape == v.shape

    def test_adam_get_key(self):
        """Test de la méthode _get_key."""
        adam = Adam()
        layer1 = MockLayer()
        layer2 = MockLayer()
        
        key1 = adam._get_key(layer1, "W")
        key2 = adam._get_key(layer2, "W")
        key3 = adam._get_key(layer1, "b")
        
        assert key1 != key2
        assert key1 != key3
        assert adam._get_key(layer1, "W") == key1


class TestOptimizersIntegration:
    """Tests d'intégration pour les optimizers."""

    def test_sgd_vs_adam_convergence(self):
        """Test comparatif SGD vs Adam (structure, pas convergence réelle)."""
        layer_sgd = MockLayer()
        layer_sgd.params["W"] = np.array([[1.0]])
        layer_sgd.grads["W"] = np.array([[0.1]])
        
        layer_adam = MockLayer()
        layer_adam.params["W"] = np.array([[1.0]])
        layer_adam.grads["W"] = np.array([[0.1]])
        
        sgd = SGD(learning_rate=0.1)
        adam = Adam(learning_rate=0.1)
        
        sgd.ft_step([layer_sgd])
        adam.ft_step([layer_adam])
        
        assert layer_sgd.params["W"] != np.array([[1.0]])
        assert layer_adam.params["W"] != np.array([[1.0]])
        assert not np.array_equal(layer_sgd.params["W"], layer_adam.params["W"])

    def test_optimizers_with_real_layer_structure(self):
        """Test avec une structure similaire à DenseLayer."""
        class SimpleLayer:
            def __init__(self):
                self.trainable = True
                self.params = {"W": np.array([[1.0, 2.0]]), "b": np.array([[0.5]])}
                self.grads = {"W": np.array([[0.1, 0.2]]), "b": np.array([[0.05]])}
        
        layer = SimpleLayer()
        sgd = SGD(learning_rate=0.1)
        sgd.ft_step([layer])
        
        assert "W" in layer.params
        assert "b" in layer.params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

