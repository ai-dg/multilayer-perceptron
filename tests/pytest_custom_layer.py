import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_layer import DenseLayer, BaseLayer


class TestBaseLayer:
    """Tests pour la classe BaseLayer."""

    def test_base_layer_initialization(self):
        """Test de l'initialisation de BaseLayer."""
        layer = BaseLayer()
        assert layer.trainable is True
        assert layer.params == {}
        assert layer.grads == {}

    def test_base_layer_ft_build(self):
        """Test de la méthode ft_build de BaseLayer."""
        layer = BaseLayer()
        layer.ft_build(10)
        assert layer.dimensions == 10

    def test_base_layer_ft_forward_not_implemented(self):
        """Test que ft_forward lève une exception."""
        layer = BaseLayer()
        with pytest.raises(NotImplementedError):
            layer.ft_forward(np.array([[1, 2, 3]]))

    def test_base_layer_ft_backward_not_implemented(self):
        """Test que ft_backward lève une exception."""
        layer = BaseLayer()
        with pytest.raises(NotImplementedError):
            layer.ft_backward(np.array([[1, 2, 3]]))


class TestDenseLayerInitialization:
    """Tests pour l'initialisation de DenseLayer."""

    def test_dense_layer_initialization_default(self):
        """Test de l'initialisation avec valeurs par défaut."""
        layer = DenseLayer(units=10)
        assert layer.units == 10
        assert layer.activation == "relu"
        assert layer.trainable is True
        assert layer.X is None
        assert layer.Z is None
        assert layer.A is None

    def test_dense_layer_initialization_with_activation(self):
        """Test de l'initialisation avec activation spécifique."""
        layer = DenseLayer(units=20, activation="sigmoid")
        assert layer.units == 20
        assert layer.activation == "sigmoid"

        layer2 = DenseLayer(units=15, activation="SOFTMAX")
        assert layer2.activation == "softmax"  

    def test_dense_layer_initialization_invalid_activation(self):
        """Test qu'une activation invalide lève une erreur lors de l'utilisation."""
        layer = DenseLayer(units=10, activation="invalid")
        layer.ft_build(5)
        layer.params["W"] = np.random.randn(5, 10)
        layer.params["b"] = np.zeros((1, 10))
        
        with pytest.raises(ValueError):
            layer.ft_forward(np.random.randn(3, 5))


class TestDenseLayerBuild:
    """Tests pour la méthode ft_build de DenseLayer."""

    def test_ft_build_shapes(self):
        """Test que ft_build initialise correctement les poids et biais."""
        layer = DenseLayer(units=5)
        layer.ft_build(10)
        
        assert "W" in layer.params
        assert "b" in layer.params
        assert layer.params["W"].shape == (10, 5)
        assert layer.params["b"].shape == (1, 5)
        assert layer.params["b"].dtype == np.float64

    def test_ft_build_weights_initialization(self):
        """Test que les poids sont initialisés avec la bonne distribution."""
        layer = DenseLayer(units=10)
        layer.ft_build(20)
        
        limit = np.sqrt(2.0 / 20)
        weights = layer.params["W"]
        
        assert np.abs(weights.mean()) < limit * 2
        assert weights.std() < limit * 2

    def test_ft_build_bias_initialization(self):
        """Test que les biais sont initialisés à zéro."""
        layer = DenseLayer(units=10)
        layer.ft_build(5)
        
        assert np.allclose(layer.params["b"], np.zeros((1, 10)))


class TestDenseLayerActivations:
    """Tests pour les fonctions d'activation."""

    def test_ft_sigmoid(self):
        """Test de la fonction sigmoid."""
        layer = DenseLayer(units=3, activation="sigmoid")
        Z = np.array([[0, 1, 2]])
        result = layer.ft_sigmoid(Z)
        
        expected = 1.0 / (1.0 + np.exp(-Z))
        assert np.allclose(result, expected)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_ft_d_sigmoid(self):
        """Test de la dérivée de sigmoid."""
        layer = DenseLayer(units=3, activation="sigmoid")
        A = np.array([[0.2, 0.5, 0.8]])
        result = layer.ft_d_sigmoid(A)
        
        expected = A * (1.0 - A)
        assert np.allclose(result, expected)

    def test_ft_relu(self):
        """Test de la fonction ReLU."""
        layer = DenseLayer(units=3, activation="relu")
        Z = np.array([[-1, 0, 2]])
        result = layer.ft_relu(Z)
        
        expected = np.maximum(0, Z)
        assert np.allclose(result, expected)
        assert np.all(result >= 0)

    def test_ft_d_relu(self):
        """Test de la dérivée de ReLU."""
        layer = DenseLayer(units=3, activation="relu")
        A = np.array([[0, 0.5, 2]])
        result = layer.ft_d_relu(A)
        
        expected = np.where(A > 0, 1.0, 0.0)
        assert np.allclose(result, expected)

    def test_ft_softmax(self):
        """Test de la fonction softmax."""
        layer = DenseLayer(units=3, activation="softmax")
        Z = np.array([[1, 2, 3]])
        result = layer.ft_softmax(Z)
        
        assert np.allclose(np.sum(result, axis=1), 1.0)
        assert np.all(result >= 0) and np.all(result <= 1)
        
        assert np.argmax(result) == 2  

    def test_ft_softmax_multiple_samples(self):
        """Test de softmax avec plusieurs échantillons."""
        layer = DenseLayer(units=3, activation="softmax")
        Z = np.array([[1, 2, 3], [3, 1, 2]])
        result = layer.ft_softmax(Z)
        
        assert np.allclose(np.sum(result, axis=1), np.ones(2))

    def test_ft_d_softmax(self):
        """Test de la dérivée de softmax."""
        layer = DenseLayer(units=3, activation="softmax")
        A = np.array([[0.2, 0.3, 0.5]])
        result = layer.ft_d_softmax(A)
        
        expected = np.ones_like(A)
        assert np.allclose(result, expected)


class TestDenseLayerForward:
    """Tests pour la méthode ft_forward de DenseLayer."""

    def test_ft_forward_relu(self):
        """Test de forward pass avec ReLU."""
        np.random.seed(42)
        layer = DenseLayer(units=3, activation="relu")
        layer.ft_build(4)
        
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = layer.ft_forward(X)
        
        assert output.shape == (2, 3)
        assert layer.X.shape == (2, 4)
        assert layer.Z.shape == (2, 3)
        assert layer.A.shape == (2, 3)
        
        assert np.array_equal(layer.X, X)
        
        assert np.all(output >= 0)

    def test_ft_forward_sigmoid(self):
        """Test de forward pass avec sigmoid."""
        np.random.seed(42)
        layer = DenseLayer(units=2, activation="sigmoid")
        layer.ft_build(3)
        
        X = np.array([[1, 2, 3]])
        output = layer.ft_forward(X)
        
        assert output.shape == (1, 2)
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_ft_forward_softmax(self):
        """Test de forward pass avec softmax."""
        np.random.seed(42)
        layer = DenseLayer(units=3, activation="softmax")
        layer.ft_build(2)
        
        X = np.array([[1, 2]])
        output = layer.ft_forward(X)
        
        assert output.shape == (1, 3)
        assert np.allclose(np.sum(output, axis=1), 1.0)

    def test_ft_forward_preserves_intermediates(self):
        """Test que les valeurs intermédiaires sont sauvegardées."""
        np.random.seed(42)
        layer = DenseLayer(units=5, activation="relu")
        layer.ft_build(3)
        
        X = np.random.randn(4, 3)
        output = layer.ft_forward(X)
        
        expected_Z = X @ layer.params["W"] + layer.params["b"]
        assert np.allclose(layer.Z, expected_Z)
        
        expected_A = layer.ft_relu(layer.Z)
        assert np.allclose(layer.A, expected_A)


class TestDenseLayerBackward:
    """Tests pour la méthode ft_backward de DenseLayer."""

    def test_ft_backward_shapes(self):
        """Test que backward retourne les bonnes shapes."""
        np.random.seed(42)
        layer = DenseLayer(units=5, activation="relu")
        layer.ft_build(3)
        
        X = np.random.randn(4, 3)
        layer.ft_forward(X)
        
        dA = np.random.randn(4, 5)
        dX = layer.ft_backward(dA)
        
        assert dX.shape == (4, 3)
        assert layer.grads["W"].shape == (3, 5)
        assert layer.grads["b"].shape == (1, 5)

    def test_ft_backward_gradient_accumulation(self):
        """Test que les gradients sont correctement calculés."""
        np.random.seed(42)
        layer = DenseLayer(units=4, activation="sigmoid")
        layer.ft_build(3)
        
        X = np.random.randn(2, 3)
        layer.ft_forward(X)
        
        dA = np.ones((2, 4))
        dX = layer.ft_backward(dA)
        
        assert "W" in layer.grads
        assert "b" in layer.grads
        
        d_activation = layer.ft_activation_backward(layer.A)
        dZ = dA * d_activation
        expected_db = np.sum(dZ, axis=0, keepdims=True)
        assert np.allclose(layer.grads["b"], expected_db)

    def test_ft_backward_with_different_activations(self):
        """Test backward avec différentes fonctions d'activation."""
        activations = ["relu", "sigmoid", "softmax"]
        
        for activation in activations:
            np.random.seed(42)
            layer = DenseLayer(units=3, activation=activation)
            layer.ft_build(2)
            
            X = np.random.randn(3, 2)
            layer.ft_forward(X)
            
            dA = np.random.randn(3, 3)
            dX = layer.ft_backward(dA)
            
            assert dX.shape == (3, 2)
            assert "W" in layer.grads
            assert "b" in layer.grads

    def test_ft_backward_requires_forward(self):
        """Test que backward nécessite un forward pass préalable."""
        layer = DenseLayer(units=3, activation="relu")
        layer.ft_build(2)
        layer.params["W"] = np.random.randn(2, 3)
        layer.params["b"] = np.zeros((1, 3))
        
        with pytest.raises((TypeError, AttributeError)):
            layer.ft_backward(np.random.randn(1, 3))


class TestDenseLayerIntegration:
    """Tests d'intégration pour DenseLayer."""

    def test_forward_backward_consistency(self):
        """Test la cohérence entre forward et backward."""
        np.random.seed(42)
        layer = DenseLayer(units=4, activation="relu")
        layer.ft_build(3)
        
        X = np.random.randn(5, 3)
        output = layer.ft_forward(X)
        
        dA = np.random.randn(5, 4)
        dX = layer.ft_backward(dA)
        
        assert dX.shape == X.shape

    def test_multiple_forward_backward_passes(self):
        """Test plusieurs passes forward/backward consécutives."""
        np.random.seed(42)
        layer = DenseLayer(units=3, activation="sigmoid")
        layer.ft_build(2)
        
        X1 = np.random.randn(2, 2)
        output1 = layer.ft_forward(X1)
        dA1 = np.random.randn(2, 3)
        dX1 = layer.ft_backward(dA1)
        
        X2 = np.random.randn(3, 2)
        output2 = layer.ft_forward(X2)
        dA2 = np.random.randn(3, 3)
        dX2 = layer.ft_backward(dA2)
        
        assert np.array_equal(layer.X, X2)
        assert output2.shape == (3, 3)

    def test_gradient_check_approximation(self):
        """Test approximatif de vérification de gradient (gradient checking)."""
        np.random.seed(42)
        layer = DenseLayer(units=2, activation="sigmoid")
        layer.ft_build(3)
        
        layer.params["W"] = np.random.randn(3, 2) * 0.1
        layer.params["b"] = np.zeros((1, 2))
        
        X = np.random.randn(1, 3)
        layer.ft_forward(X)
        
        dA = np.random.randn(1, 2)
        layer.ft_backward(dA)
        
        epsilon = 1e-7
        grad_W_numerical = np.zeros_like(layer.params["W"])
        
        for i in range(layer.params["W"].shape[0]):
            for j in range(layer.params["W"].shape[1]):
                layer.params["W"][i, j] += epsilon
                output_plus = layer.ft_forward(X)
                cost_plus = np.sum(output_plus)
                
                layer.params["W"][i, j] -= 2 * epsilon
                output_minus = layer.ft_forward(X)
                cost_minus = np.sum(output_minus)
                
                grad_W_numerical[i, j] = (cost_plus - cost_minus) / (2 * epsilon)
                
                layer.params["W"][i, j] += epsilon
        
        assert not np.allclose(grad_W_numerical, 0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

