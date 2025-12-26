import sys
import os
import numpy as np
import pytest
import tempfile
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import CustomSequential
from custom_layer import DenseLayer
from optimizers import SGD, Adam
from losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy
from metrics import Accuracy, Precision, Recall, F1Score
from callbacks import History, EarlyStopping


class TestCustomSequentialInitialization:
    """Tests pour l'initialisation de CustomSequential."""

    def test_initialization_empty(self):
        """Test d'initialisation sans couches."""
        model = CustomSequential()
        assert model.layers == []
        assert model.isbuilt is False
        assert model.optimizer is None
        assert model.loss is None
        assert model.metrics == []

    def test_initialization_with_layers(self):
        """Test d'initialisation avec des couches."""
        layers = [
            DenseLayer(units=5, activation="relu"),
            DenseLayer(units=3, activation="sigmoid")
        ]
        model = CustomSequential(layers=layers)
        assert len(model.layers) == 2
        assert model.layers == layers
        assert model.isbuilt is False


class TestCustomSequentialAdd:
    """Tests pour la méthode ft_add."""

    def test_add_single_layer(self):
        """Test d'ajout d'une seule couche."""
        model = CustomSequential()
        layer = DenseLayer(units=5, activation="relu")
        model.ft_add(layer)
        assert len(model.layers) == 1
        assert model.layers[0] == layer

    def test_add_multiple_layers(self):
        """Test d'ajout de plusieurs couches."""
        model = CustomSequential()
        layer1 = DenseLayer(units=5, activation="relu")
        layer2 = DenseLayer(units=3, activation="sigmoid")
        model.ft_add(layer1)
        model.ft_add(layer2)
        assert len(model.layers) == 2
        assert model.layers[0] == layer1
        assert model.layers[1] == layer2


class TestCustomSequentialBuild:
    """Tests pour la méthode ft_build."""

    def test_build_single_layer(self):
        """Test de build avec une seule couche."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_build(num_features=10)
        assert model.isbuilt is True
        assert "W" in model.layers[0].params
        assert model.layers[0].params["W"].shape == (10, 5)

    def test_build_multiple_layers(self):
        """Test de build avec plusieurs couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=10)
        assert model.isbuilt is True
        assert model.layers[0].params["W"].shape == (10, 5)
        assert model.layers[1].params["W"].shape == (5, 3)

    def test_build_three_layers(self):
        """Test de build avec trois couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=8, activation="relu"))
        model.ft_add(DenseLayer(units=4, activation="relu"))
        model.ft_add(DenseLayer(units=2, activation="sigmoid"))
        model.ft_build(num_features=16)
        assert model.isbuilt is True
        assert model.layers[0].params["W"].shape == (16, 8)
        assert model.layers[1].params["W"].shape == (8, 4)
        assert model.layers[2].params["W"].shape == (4, 2)


class TestCustomSequentialCompile:
    """Tests pour la méthode ft_compile."""

    def test_compile_sgd_bce(self):
        """Test de compilation avec SGD et BinaryCrossEntropy."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        model.ft_compile(optimizer="sgd", loss="bce", learning_rate=0.01)
        assert isinstance(model.optimizer, SGD)
        assert model.optimizer.lr == 0.01
        assert isinstance(model.loss, BinaryCrossEntropy)

    def test_compile_adam_mse(self):
        """Test de compilation avec Adam et MeanSquaredError."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_compile(optimizer="adam", loss="mse", learning_rate=0.001)
        assert isinstance(model.optimizer, Adam)
        assert model.optimizer.lr == 0.001
        assert isinstance(model.loss, MeanSquaredError)

    def test_compile_categorical_crossentropy(self):
        """Test de compilation avec CategoricalCrossEntropy."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="softmax"))
        model.ft_compile(optimizer="adam", loss="cce")
        assert isinstance(model.loss, CategoricalCrossEntropy)

    def test_compile_with_metrics(self):
        """Test de compilation avec des métriques."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        model.ft_compile(
            optimizer="sgd",
            loss="bce",
            metrics=["accuracy", "precision", "recall", "f1"]
        )
        assert len(model.metrics) == 4
        assert isinstance(model.metrics[0], Accuracy)
        assert isinstance(model.metrics[1], Precision)
        assert isinstance(model.metrics[2], Recall)
        assert isinstance(model.metrics[3], F1Score)

    def test_compile_invalid_optimizer(self):
        """Test avec un optimizer invalide."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        with pytest.raises(ValueError):
            model.ft_compile(optimizer="invalid", loss="bce")

    def test_compile_invalid_loss(self):
        """Test avec une loss invalide."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        with pytest.raises(ValueError):
            model.ft_compile(optimizer="sgd", loss="invalid")

    def test_compile_invalid_metric(self):
        """Test avec une métrique invalide."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        with pytest.raises(ValueError):
            model.ft_compile(optimizer="sgd", loss="bce", metrics=["invalid"])

    def test_compile_loss_variations(self):
        """Test des variations de noms pour les losses."""
        model1 = CustomSequential()
        model1.ft_add(DenseLayer(units=5, activation="sigmoid"))
        model1.ft_compile(optimizer="sgd", loss="binary_crossentropy")
        assert isinstance(model1.loss, BinaryCrossEntropy)

        model2 = CustomSequential()
        model2.ft_add(DenseLayer(units=5, activation="relu"))
        model2.ft_compile(optimizer="sgd", loss="mean_squared_error")
        assert isinstance(model2.loss, MeanSquaredError)

        model3 = CustomSequential()
        model3.ft_add(DenseLayer(units=5, activation="softmax"))
        model3.ft_compile(optimizer="sgd", loss="categorical_crossentropy")
        assert isinstance(model3.loss, CategoricalCrossEntropy)


class TestCustomSequentialForward:
    """Tests pour la méthode ft_forward."""

    def test_forward_single_layer(self):
        """Test de forward avec une seule couche."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=3, activation="relu"))
        model.ft_build(num_features=5)
        X = np.random.randn(10, 5)
        output = model.ft_forward(X)
        assert output.shape == (10, 3)

    def test_forward_multiple_layers(self):
        """Test de forward avec plusieurs couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=8, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=10)
        X = np.random.randn(5, 10)
        output = model.ft_forward(X)
        assert output.shape == (5, 3)

    def test_forward_three_layers(self):
        """Test de forward avec trois couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=6, activation="relu"))
        model.ft_add(DenseLayer(units=4, activation="relu"))
        model.ft_add(DenseLayer(units=2, activation="softmax"))
        model.ft_build(num_features=8)
        X = np.random.randn(3, 8)
        output = model.ft_forward(X)
        assert output.shape == (3, 2)
        assert np.allclose(np.sum(output, axis=1), 1.0)


class TestCustomSequentialBackward:
    """Tests pour la méthode ft_backward."""

    def test_backward_single_layer(self):
        """Test de backward avec une seule couche."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=3, activation="relu"))
        model.ft_build(num_features=5)
        X = np.random.randn(10, 5)
        model.ft_forward(X)
        d_out = np.random.randn(10, 3)
        model.ft_backward(d_out)
        assert "W" in model.layers[0].grads
        assert "b" in model.layers[0].grads

    def test_backward_multiple_layers(self):
        """Test de backward avec plusieurs couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=8, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=10)
        X = np.random.randn(5, 10)
        model.ft_forward(X)
        d_out = np.random.randn(5, 3)
        model.ft_backward(d_out)    
        assert "W" in model.layers[0].grads
        assert "W" in model.layers[1].grads


class TestCustomSequentialFit:
    """Tests pour la méthode ft_fit."""

    def test_fit_simple(self):
        """Test d'entraînement simple."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_compile(optimizer="sgd", loss="bce", learning_rate=0.1)
        
        X_train = np.random.randn(20, 10)
        y_train = np.random.randint(0, 2, (20, 1)).astype(float)
        
        history = model.ft_fit(X_train, y_train, epochs=2, batch_size=10)
        assert isinstance(history, History)
        assert len(history.history["loss"]) == 2

    def test_fit_with_validation(self):
        """Test d'entraînement avec validation."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_compile(optimizer="sgd", loss="bce", learning_rate=0.1)
        
        X_train = np.random.randn(20, 10)
        y_train = np.random.randint(0, 2, (20, 1)).astype(float)
        X_valid = np.random.randn(10, 10)
        y_valid = np.random.randint(0, 2, (10, 1)).astype(float)
        
        history = model.ft_fit(
            X_train, y_train,
            X_valid=X_valid, y_valid=y_valid,
            epochs=2, batch_size=10
        )
        assert "val_loss" in history.history
        assert len(history.history["val_loss"]) == 2

    def test_fit_with_metrics(self):
        """Test d'entraînement avec métriques."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_compile(
            optimizer="sgd",
            loss="bce",
            metrics=["accuracy"],
            learning_rate=0.1
        )
        
        X_train = np.random.randn(20, 10)
        y_train = np.random.randint(0, 2, (20, 1)).astype(float)
        
        history = model.ft_fit(X_train, y_train, epochs=2, batch_size=10)
        assert "accuracy" in history.history

    def test_fit_with_callbacks(self):
        """Test d'entraînement avec callbacks personnalisés."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_compile(optimizer="sgd", loss="bce", learning_rate=0.1)
        
        X_train = np.random.randn(20, 10)
        y_train = np.random.randint(0, 2, (20, 1)).astype(float)
        
        custom_history = History()
        early_stopping = EarlyStopping(patience=3)
        
        history = model.ft_fit(
            X_train, y_train,
            epochs=5,
            batch_size=10,
            callbacks=[custom_history, early_stopping]
        )
        assert history == custom_history

    def test_fit_not_compiled(self):
        """Test que fit lève une erreur si le modèle n'est pas compilé."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        X_train = np.random.randn(10, 5)
        y_train = np.random.randint(0, 2, (10, 1)).astype(float)
        
        with pytest.raises(RuntimeError):
            model.ft_fit(X_train, y_train, epochs=1)

    def test_fit_auto_build(self):
        """Test que fit construit automatiquement le modèle si nécessaire."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_compile(optimizer="sgd", loss="bce", learning_rate=0.1)
        
        X_train = np.random.randn(20, 10)
        y_train = np.random.randint(0, 2, (20, 1)).astype(float)
        
        assert model.isbuilt is False
        model.ft_fit(X_train, y_train, epochs=1, batch_size=10)
        assert model.isbuilt is True


class TestCustomSequentialEvaluate:
    """Tests pour la méthode ft_evaluate."""

    def test_evaluate_simple(self):
        """Test d'évaluation simple."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_build(num_features=10)
        model.ft_compile(optimizer="sgd", loss="bce")
        
        X = np.random.randn(10, 10)
        y = np.random.randint(0, 2, (10, 1)).astype(float)
        
        loss, metrics = model.ft_evaluate(X, y)
        assert isinstance(loss, (float, np.floating))
        assert isinstance(metrics, list)
        assert len(metrics) == 0  

    def test_evaluate_with_metrics(self):
        """Test d'évaluation avec métriques."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        model.ft_build(num_features=10)
        model.ft_compile(optimizer="sgd", loss="bce", metrics=["accuracy"])
        
        X = np.random.randn(10, 10)
        y = np.random.randint(0, 2, (10, 1)).astype(float)
        
        loss, metrics = model.ft_evaluate(X, y)
        assert isinstance(loss, (float, np.floating))
        assert len(metrics) == 1
        assert 0 <= metrics[0] <= 1  

    def test_evaluate_not_compiled(self):
        """Test que evaluate lève une erreur si le modèle n'est pas compilé."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="sigmoid"))
        model.ft_build(num_features=10)
        X = np.random.randn(10, 10)
        y = np.random.randint(0, 2, (10, 1)).astype(float)
        
        with pytest.raises(RuntimeError):
            model.ft_evaluate(X, y)


class TestCustomSequentialPredict:
    """Tests pour la méthode ft_predict."""

    def test_predict_simple(self):
        """Test de prédiction simple."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=5)
        
        X = np.random.randn(10, 5)
        predictions = model.ft_predict(X)
        assert predictions.shape == (10, 3)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_predict_multiple_layers(self):
        """Test de prédiction avec plusieurs couches."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=8, activation="relu"))
        model.ft_add(DenseLayer(units=2, activation="softmax"))
        model.ft_build(num_features=10)
        
        X = np.random.randn(5, 10)
        predictions = model.ft_predict(X)
        assert predictions.shape == (5, 2)
        assert np.allclose(np.sum(predictions, axis=1), 1.0)


class TestCustomSequentialWeights:
    """Tests pour les méthodes de gestion des poids."""

    def test_get_weights(self):
        """Test de récupération des poids."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=10)
        
        weights = model.ft_get_weights()
        assert len(weights) == 2
        assert "W" in weights[0]
        assert "b" in weights[0]
        assert "W" in weights[1]
        assert "b" in weights[1]

    def test_set_weights(self):
        """Test de définition des poids."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="sigmoid"))
        model.ft_build(num_features=10)
        
        original_weights = model.ft_get_weights()
        
        new_weights = [
            {"W": np.random.randn(10, 5), "b": np.random.randn(1, 5)},
            {"W": np.random.randn(5, 3), "b": np.random.randn(1, 3)}
        ]
        
        model.ft_set_weights(new_weights)
        
        updated_weights = model.ft_get_weights()
        assert not np.array_equal(original_weights[0]["W"], updated_weights[0]["W"])
        assert np.array_equal(new_weights[0]["W"], updated_weights[0]["W"])

    def test_set_weights_clears_grads(self):
        """Test que set_weights efface les gradients."""
        model = CustomSequential()
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_build(num_features=10)
        
        X = np.random.randn(5, 10)
        model.ft_forward(X)
        d_out = np.random.randn(5, 5)
        model.ft_backward(d_out)
        
        assert "W" in model.layers[0].grads
        
        weights = model.ft_get_weights()
        model.ft_set_weights(weights)
        
        assert model.layers[0].grads == {}


class TestCustomSequentialSaveLoad:
    """Tests pour les méthodes de sauvegarde et chargement."""

    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        np.random.seed(42)
        model1 = CustomSequential()
        model1.ft_add(DenseLayer(units=5, activation="relu"))
        model1.ft_add(DenseLayer(units=2, activation="sigmoid"))
        model1.ft_build(num_features=10)
        
        weights1 = model1.ft_get_weights()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model1.ft_save(temp_path)
            
            model2 = CustomSequential()
            model2.ft_load(temp_path)
            
            assert len(model2.layers) == 2
            assert model2.isbuilt is True
            assert model2.layers[0].units == 5
            assert model2.layers[1].units == 2
            
            weights2 = model2.ft_get_weights()
            assert len(weights2) == 2
            assert np.array_equal(weights1[0]["W"], weights2[0]["W"])
            assert np.array_equal(weights1[1]["W"], weights2[1]["W"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_load_preserves_architecture(self):
        """Test que save/load préserve l'architecture."""
        model1 = CustomSequential()
        model1.ft_add(DenseLayer(units=8, activation="relu"))
        model1.ft_add(DenseLayer(units=4, activation="relu"))
        model1.ft_add(DenseLayer(units=2, activation="softmax"))
        model1.ft_build(num_features=16)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model1.ft_save(temp_path)
            
            model2 = CustomSequential()
            model2.ft_load(temp_path)
            
            assert len(model2.layers) == 3
            assert model2.layers[0].units == 8
            assert model2.layers[0].activation == "relu"
            assert model2.layers[1].units == 4
            assert model2.layers[2].units == 2
            assert model2.layers[2].activation == "softmax"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestCustomSequentialIntegration:
    """Tests d'intégration pour CustomSequential."""

    def test_full_training_pipeline(self):
        """Test d'un pipeline d'entraînement complet."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=10, activation="relu"))
        model.ft_add(DenseLayer(units=5, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="sigmoid"))
        
        model.ft_compile(
            optimizer="adam",
            loss="bce",
            metrics=["accuracy"],
            learning_rate=0.01
        )
        
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, (50, 1)).astype(float)
        X_valid = np.random.randn(20, 20)
        y_valid = np.random.randint(0, 2, (20, 1)).astype(float)
        
        history = model.ft_fit(
            X_train, y_train,
            X_valid=X_valid, y_valid=y_valid,
            epochs=3,
            batch_size=10
        )
        
        loss, metrics = model.ft_evaluate(X_valid, y_valid)
        assert isinstance(loss, (float, np.floating))
        assert len(metrics) == 1
        
        predictions = model.ft_predict(X_valid)
        assert predictions.shape == (20, 1)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_multiclass_classification(self):
        """Test avec classification multi-classes."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=10, activation="relu"))
        model.ft_add(DenseLayer(units=3, activation="softmax"))
        
        model.ft_compile(
            optimizer="adam",
            loss="cce",
            metrics=["accuracy"],
            learning_rate=0.01
        )
        
        X_train = np.random.randn(30, 5)
        y_train = np.eye(3)[np.random.randint(0, 3, 30)]  
        
        history = model.ft_fit(X_train, y_train, epochs=2, batch_size=10)
        assert len(history.history["loss"]) == 2

    def test_regression(self):
        """Test avec régression."""
        np.random.seed(42)
        model = CustomSequential()
        model.ft_add(DenseLayer(units=10, activation="relu"))
        model.ft_add(DenseLayer(units=1, activation="relu"))
        
        model.ft_compile(
            optimizer="sgd",
            loss="mse",
            learning_rate=0.01
        )
        
        X_train = np.random.randn(30, 5)
        y_train = np.random.randn(30, 1)
        
        history = model.ft_fit(X_train, y_train, epochs=2, batch_size=10)
        assert len(history.history["loss"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

