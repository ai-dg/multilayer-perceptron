import sys
import os
import numpy as np
import pytest
import tempfile
import argparse

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp import build_model, run_split, run_train, run_predict, main
from custom_model import CustomSequential
from custom_layer import DenseLayer
from data_processor import DataProcessor
import pandas as pd


def create_test_csv(path: str, n_samples: int = 100, n_features: int = 10):
    """Crée un fichier CSV de test avec des données simulées.
    Format: ID, diagnosis (M/B), feature1, feature2, ...
    """
    np.random.seed(42)
    ids = np.arange(1000, 1000 + n_samples)
    diagnosis = np.random.choice(["B", "M"], size=n_samples)
    features = np.random.randn(n_samples, n_features)
    data = np.column_stack([ids, diagnosis, features])
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, header=False)
    return df


class TestBuildModel:
    """Tests pour la fonction build_model."""

    def test_build_model_default_layers(self):
        """Test avec les layers par défaut (au moins 2 hidden layers)."""
        args = argparse.Namespace()
        args.layers = None
        args.optimizer = "Adam"
        args.metrics = ["Accuracy"]
        args.learning_rate = 0.001
        
        model = build_model(input_dim=10, args=args)
        
        assert len(model.layers) >= 3
        assert model.layers[0].activation == "relu"
        assert model.layers[1].activation == "relu"
        assert model.layers[-1].activation == "sigmoid"
        assert model.layers[-1].units == 1

    def test_build_model_custom_layers(self):
        """Test avec des layers personnalisés."""
        args = argparse.Namespace()
        args.layers = [24, 24, 24]
        args.optimizer = "SGD"
        args.metrics = ["Accuracy"]
        args.learning_rate = 0.01
        
        model = build_model(input_dim=10, args=args)
        
        assert len(model.layers) == 4
        assert model.layers[0].units == 24
        assert model.layers[1].units == 24
        assert model.layers[2].units == 24
        assert model.layers[3].units == 1

    def test_build_model_single_layer_adds_more(self):
        """Test qu'un seul layer est complété pour avoir au moins 2."""
        args = argparse.Namespace()
        args.layers = [16]
        args.optimizer = "Adam"
        args.metrics = ["Accuracy"]
        args.learning_rate = 0.001
        
        model = build_model(input_dim=10, args=args)
        
        assert len(model.layers) >= 3

    def test_build_model_compiles_correctly(self):
        """Test que le modèle est correctement compilé."""
        args = argparse.Namespace()
        args.layers = [24, 24]
        args.optimizer = "Adam"
        args.metrics = ["Accuracy", "Precision"]
        args.learning_rate = 0.0314
        
        model = build_model(input_dim=10, args=args)
        
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) == 2
        assert model.isbuilt is True


class TestRunSplit:
    """Tests pour la fonction run_split."""

    def test_run_split_basic(self):
        """Test de split basique."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        
        try:
            create_test_csv(csv_path, n_samples=100, n_features=10)
            
            args = argparse.Namespace()
            args.dataset = csv_path
            args.train_out = train_path
            args.valid_out = valid_path
            args.valid_ratio = 0.2
            args.seed = 42
            
            run_split(args)
            
            assert os.path.exists(train_path)
            assert os.path.exists(valid_path)
            
            train_data = np.load(train_path)
            valid_data = np.load(valid_path)
            
            assert "X" in train_data
            assert "y" in train_data
            assert "X" in valid_data
            assert "y" in valid_data
            
            assert len(train_data["X"]) == 80
            assert len(valid_data["X"]) == 20
        finally:
            for path in [csv_path, train_path, valid_path]:
                if os.path.exists(path):
                    os.remove(path)

    def test_run_split_missing_dataset(self):
        """Test que run_split lève une erreur si dataset est manquant."""
        args = argparse.Namespace()
        args.dataset = None
        
        with pytest.raises(ValueError):
            run_split(args)

    def test_run_split_custom_ratio(self):
        """Test avec un ratio personnalisé."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        
        try:
            create_test_csv(csv_path, n_samples=100, n_features=10)
            
            args = argparse.Namespace()
            args.dataset = csv_path
            args.train_out = train_path
            args.valid_out = valid_path
            args.valid_ratio = 0.3
            args.seed = 42
            
            run_split(args)
            
            train_data = np.load(train_path)
            valid_data = np.load(valid_path)
            
            assert len(train_data["X"]) == 70
            assert len(valid_data["X"]) == 30
        finally:
            for path in [csv_path, train_path, valid_path]:
                if os.path.exists(path):
                    os.remove(path)


class TestRunTrain:
    """Tests pour la fonction run_train."""

    def test_run_train_basic(self):
        """Test d'entraînement basique."""
        np.random.seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            curve_prefix = os.path.splitext(f.name)[0]
        
        try:
            X_train = np.random.randn(50, 10)
            y_train = np.random.randint(0, 2, (50, 1)).astype(float)
            X_valid = np.random.randn(20, 10)
            y_valid = np.random.randint(0, 2, (20, 1)).astype(float)
            
            np.savez(train_path, X=X_train, y=y_train)
            np.savez(valid_path, X=X_valid, y=y_valid)
            
            args = argparse.Namespace()
            args.train_data = train_path
            args.valid_data = valid_path
            args.model_path = model_path
            args.layers = [24, 24]
            args.epochs = 2
            args.batch_size = 8
            args.optimizer = "Adam"
            args.learning_rate = 0.01
            args.metrics = ["Accuracy"]
            args.early_stopping = False
            args.patience = 5
            args.min_delta = 0.0
            args.curve_prefix = curve_prefix
            
            run_train(args)
            
            assert os.path.exists(model_path)
            
            assert os.path.exists(f"{curve_prefix}_loss.png")
            assert os.path.exists(f"{curve_prefix}_accuracy.png")
        finally:
            for path in [train_path, valid_path, model_path]:
                if os.path.exists(path):
                    os.remove(path)
            for suffix in ["_loss.png", "_accuracy.png"]:
                path = f"{curve_prefix}{suffix}"
                if os.path.exists(path):
                    os.remove(path)

    def test_run_train_with_early_stopping(self):
        """Test d'entraînement avec early stopping."""
        np.random.seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            curve_prefix = os.path.splitext(f.name)[0]
        
        try:
            X_train = np.random.randn(30, 10)
            y_train = np.random.randint(0, 2, (30, 1)).astype(float)
            X_valid = np.random.randn(10, 10)
            y_valid = np.random.randint(0, 2, (10, 1)).astype(float)
            
            np.savez(train_path, X=X_train, y=y_train)
            np.savez(valid_path, X=X_valid, y=y_valid)
            
            args = argparse.Namespace()
            args.train_data = train_path
            args.valid_data = valid_path
            args.model_path = model_path
            args.layers = [24, 24]
            args.epochs = 10
            args.batch_size = 8
            args.optimizer = "SGD"
            args.learning_rate = 0.01
            args.metrics = ["Accuracy"]
            args.early_stopping = True
            args.patience = 2
            args.min_delta = 0.0
            args.curve_prefix = curve_prefix
            
            run_train(args)
            
            assert os.path.exists(model_path)
        finally:
            for path in [train_path, valid_path, model_path]:
                if os.path.exists(path):
                    os.remove(path)
            for suffix in ["_loss.png", "_accuracy.png"]:
                path = f"{curve_prefix}{suffix}"
                if os.path.exists(path):
                    os.remove(path)

    def test_run_train_missing_data(self):
        """Test que run_train lève une erreur si les données sont manquantes."""
        args = argparse.Namespace()
        args.train_data = None
        args.valid_data = None
        
        with pytest.raises(ValueError):
            run_train(args)

    def test_run_train_multiple_metrics(self):
        """Test avec plusieurs métriques."""
        np.random.seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            curve_prefix = os.path.splitext(f.name)[0]
        
        try:
            X_train = np.random.randn(40, 10)
            y_train = np.random.randint(0, 2, (40, 1)).astype(float)
            X_valid = np.random.randn(10, 10)
            y_valid = np.random.randint(0, 2, (10, 1)).astype(float)
            
            np.savez(train_path, X=X_train, y=y_train)
            np.savez(valid_path, X=X_valid, y=y_valid)
            
            args = argparse.Namespace()
            args.train_data = train_path
            args.valid_data = valid_path
            args.model_path = model_path
            args.layers = [24, 24]
            args.epochs = 2
            args.batch_size = 8
            args.optimizer = "Adam"
            args.learning_rate = 0.01
            args.metrics = ["Accuracy", "Precision", "Recall"]
            args.early_stopping = False
            args.patience = 5
            args.min_delta = 0.0
            args.curve_prefix = curve_prefix
            
            run_train(args)
            
            assert os.path.exists(model_path)
        finally:
            for path in [train_path, valid_path, model_path]:
                if os.path.exists(path):
                    os.remove(path)
            for suffix in ["_loss.png", "_accuracy.png"]:
                path = f"{curve_prefix}{suffix}"
                if os.path.exists(path):
                    os.remove(path)


class TestRunPredict:
    """Tests pour la fonction run_predict."""

    def test_run_predict_basic(self):
        """Test de prédiction basique."""
        np.random.seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            predict_data_path = f.name
        
        try:
            model = CustomSequential()
            model.ft_add(DenseLayer(units=24, activation="relu"))
            model.ft_add(DenseLayer(units=24, activation="relu"))
            model.ft_add(DenseLayer(units=1, activation="sigmoid"))
            model.ft_build(input_dim=10)
            model.ft_save(model_path)
            
            X = np.random.randn(20, 10)
            y = np.random.randint(0, 2, (20, 1)).astype(float)
            np.savez(predict_data_path, X=X, y=y)
            
            args = argparse.Namespace()
            args.model_path = model_path
            args.predict_data = predict_data_path
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                run_predict(args)
            
            output = f.getvalue()
            assert "Binary cross-entropy" in output
        finally:
            for path in [model_path, predict_data_path]:
                if os.path.exists(path):
                    os.remove(path)

    def test_run_predict_missing_model(self):
        """Test que run_predict lève une erreur si le modèle est manquant."""
        args = argparse.Namespace()
        args.model_path = None
        args.predict_data = "dummy.npz"
        
        with pytest.raises(ValueError):
            run_predict(args)

    def test_run_predict_missing_data(self):
        """Test que run_predict lève une erreur si les données sont manquantes."""
        args = argparse.Namespace()
        args.model_path = "dummy.pkl"
        args.predict_data = None
        
        with pytest.raises(ValueError):
            run_predict(args)

    def test_run_predict_calculates_loss(self):
        """Test que la loss est correctement calculée."""
        np.random.seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            predict_data_path = f.name
        
        try:
            model = CustomSequential()
            model.ft_add(DenseLayer(units=16, activation="relu"))
            model.ft_add(DenseLayer(units=1, activation="sigmoid"))
            model.ft_build(input_dim=10)
            model.ft_save(model_path)
            
            X = np.random.randn(15, 10)
            y = np.random.randint(0, 2, (15, 1)).astype(float)
            np.savez(predict_data_path, X=X, y=y)
            
            args = argparse.Namespace()
            args.model_path = model_path
            args.predict_data = predict_data_path
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                run_predict(args)
            
            output = f.getvalue()
            assert "Binary cross-entropy" in output
            assert ":" in output
        finally:
            for path in [model_path, predict_data_path]:
                if os.path.exists(path):
                    os.remove(path)


class TestMlpIntegration:
    """Tests d'intégration pour mlp.py."""

    def test_full_pipeline_split_train_predict(self):
        """Test d'un pipeline complet : split → train → predict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            train_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            valid_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            curve_prefix = os.path.splitext(f.name)[0]
        
        try:
            create_test_csv(csv_path, n_samples=100, n_features=10)
            
            args_split = argparse.Namespace()
            args_split.dataset = csv_path
            args_split.train_out = train_path
            args_split.valid_out = valid_path
            args_split.valid_ratio = 0.2
            args_split.seed = 42
            run_split(args_split)
            
            args_train = argparse.Namespace()
            args_train.train_data = train_path
            args_train.valid_data = valid_path
            args_train.model_path = model_path
            args_train.layers = [24, 24]
            args_train.epochs = 3
            args_train.batch_size = 8
            args_train.optimizer = "Adam"
            args_train.learning_rate = 0.01
            args_train.metrics = ["Accuracy"]
            args_train.early_stopping = False
            args_train.patience = 5
            args_train.min_delta = 0.0
            args_train.curve_prefix = curve_prefix
            run_train(args_train)
            
            args_predict = argparse.Namespace()
            args_predict.model_path = model_path
            args_predict.predict_data = valid_path
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                run_predict(args_predict)
            
            output = f.getvalue()
            assert "Binary cross-entropy" in output
            
            assert os.path.exists(model_path)
            assert os.path.exists(f"{curve_prefix}_loss.png")
            assert os.path.exists(f"{curve_prefix}_accuracy.png")
        finally:
            for path in [csv_path, train_path, valid_path, model_path]:
                if os.path.exists(path):
                    os.remove(path)
            for suffix in ["_loss.png", "_accuracy.png"]:
                path = f"{curve_prefix}{suffix}"
                if os.path.exists(path):
                    os.remove(path)

    def test_model_has_at_least_two_hidden_layers(self):
        """Test que le modèle respecte l'exigence d'au moins 2 hidden layers."""
        args = argparse.Namespace()
        args.layers = None
        args.optimizer = "Adam"
        args.metrics = ["Accuracy"]
        args.learning_rate = 0.001
        
        model = build_model(input_dim=10, args=args)
        
        hidden_count = sum(1 for layer in model.layers if layer.activation == "relu")
        assert hidden_count >= 2, "Le sujet exige au moins 2 hidden layers"

    def test_output_layer_is_sigmoid_for_binary_classification(self):
        """Test que la couche de sortie utilise sigmoid pour classification binaire."""
        args = argparse.Namespace()
        args.layers = [24, 24]
        args.optimizer = "Adam"
        args.metrics = ["Accuracy"]
        args.learning_rate = 0.001
        
        model = build_model(input_dim=10, args=args)
        
        assert model.layers[-1].activation == "sigmoid"
        assert model.layers[-1].units == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

