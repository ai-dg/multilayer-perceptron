import sys
import os
import numpy as np
import pytest
import tempfile
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_predictor import CustomPredictor
from custom_model import CustomSequential
from custom_layer import DenseLayer
from data_processor import DataProcessor


def create_test_model(path: str, input_dim: int = 10):
    """Crée et sauvegarde un modèle de test."""
    model = CustomSequential()
    model.ft_add(DenseLayer(units=5, activation="relu"))
    model.ft_add(DenseLayer(units=1, activation="sigmoid"))
    model.ft_build(input_dim=input_dim)
    model.ft_save(path)
    return model


def create_test_csv(path: str, n_samples: int = 50, n_features: int = 10):
    """Crée un fichier CSV de test avec des données simulées.
    Format: ID, diagnosis (M/B), feature1, feature2, ...
    """
    np.random.seed(42)
    ids = np.arange(1000, 1000 + n_samples)
    
    diagnosis = np.random.choice(["B", "M"], size=n_samples)
    
    features = np.random.randn(n_samples, n_features)
    
    data = np.column_stack([ids, diagnosis, features])
    import pandas as pd
    df = pd.DataFrame(data)
    
    df.to_csv(path, index=False, header=False)
    return df


class TestCustomPredictorInitialization:
    """Tests pour l'initialisation de CustomPredictor."""

    def test_initialization(self):
        """Test de l'initialisation."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            
            predictor = CustomPredictor(model_path)
            
            assert predictor.model is not None
            assert len(predictor.model.layers) == 2
            assert predictor.model.isbuilt is True
            assert predictor.model.optimizer is not None
            assert predictor.model.loss is not None
            assert len(predictor.model.metrics) == 1
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_initialization_compiles_model(self):
        """Test que le modèle est compilé avec les bons paramètres."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            
            predictor = CustomPredictor(model_path)
            
            from optimizers import Adam
            assert isinstance(predictor.model.optimizer, Adam)
            assert predictor.model.optimizer.lr == 0.001
            
            from losses import BinaryCrossEntropy
            assert isinstance(predictor.model.loss, BinaryCrossEntropy)
            
            from metrics import Accuracy
            assert isinstance(predictor.model.metrics[0], Accuracy)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_initialization_missing_model_file(self):
        """Test avec un fichier modèle inexistant."""
        with pytest.raises(FileNotFoundError):
            CustomPredictor("nonexistent_model.pkl")


class TestCustomPredictorPredictFile:
    """Tests pour la méthode predict_file."""

    def test_predict_file_basic(self):
        """Test de prédiction basique."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            
            create_test_csv(csv_path, n_samples=20, n_features=10)
            
            predictor = CustomPredictor(model_path)
            loss = predictor.ft_predict_file(csv_path)
            
            assert isinstance(loss, float)
            assert loss >= 0
            assert not np.isnan(loss)
            assert not np.isinf(loss)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_different_sizes(self):
        """Test avec différentes tailles de dataset."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=5)
            create_test_csv(csv_path, n_samples=50, n_features=5)
            
            predictor = CustomPredictor(model_path)
            loss = predictor.ft_predict_file(csv_path)
            
            assert isinstance(loss, float)
            assert loss >= 0
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_normalization(self):
        """Test que la normalisation est appliquée correctement."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=5)
            create_test_csv(csv_path, n_samples=30, n_features=5)
            
            predictor = CustomPredictor(model_path)
            
            data_processor = DataProcessor(csv_path)
            X, y = data_processor.ft_load_dataset()
            
            loss = predictor.ft_predict_file(csv_path)
            
            assert isinstance(loss, float)
            assert loss >= 0
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_uses_correct_loss(self):
        """Test que la loss BinaryCrossEntropy est utilisée."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            create_test_csv(csv_path, n_samples=20, n_features=10)
            
            predictor = CustomPredictor(model_path)
            
            from losses import BinaryCrossEntropy
            assert isinstance(predictor.model.loss, BinaryCrossEntropy)
            
            loss = predictor.ft_predict_file(csv_path)
            assert isinstance(loss, float)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_missing_csv(self):
        """Test avec un fichier CSV inexistant."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            
            predictor = CustomPredictor(model_path)
            
            with pytest.raises(FileNotFoundError):
                predictor.ft_predict_file("nonexistent.csv")
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_predict_file_shape_mismatch(self):
        """Test avec un mismatch de dimensions entre modèle et données."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            
            create_test_csv(csv_path, n_samples=20, n_features=5)
            
            predictor = CustomPredictor(model_path)
            
            with pytest.raises((ValueError, IndexError)):
                predictor.ft_predict_file(csv_path)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_returns_binary_crossentropy(self):
        """Test que la valeur retournée est bien la BinaryCrossEntropy."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            create_test_csv(csv_path, n_samples=30, n_features=10)
            
            predictor = CustomPredictor(model_path)
            loss = predictor.ft_predict_file(csv_path)
            
            data_processor = DataProcessor(csv_path)
            X, y = data_processor.ft_load_dataset()
            X_norm, _ = data_processor.ft_normalize(X, X)
            y_pred = predictor.model.ft_predict(X_norm)
            expected_loss = float(predictor.model.loss(y, y_pred))
            
            assert np.isclose(loss, expected_loss, rtol=1e-5)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


class TestCustomPredictorIntegration:
    """Tests d'intégration pour CustomPredictor."""

    def test_full_pipeline(self):
        """Test d'un pipeline complet : créer modèle, sauvegarder, charger, prédire."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            model = CustomSequential()
            model.ft_add(DenseLayer(units=8, activation="relu"))
            model.ft_add(DenseLayer(units=1, activation="sigmoid"))
            model.ft_build(input_dim=10)
            model.ft_save(model_path)

            create_test_csv(csv_path, n_samples=40, n_features=10)
            
            predictor = CustomPredictor(model_path)
            
            loss = predictor.ft_predict_file(csv_path)
            
            assert isinstance(loss, float)
            assert loss >= 0
            assert not np.isnan(loss)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_with_different_models(self):
        """Test avec différents modèles."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path1 = f.name
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path2 = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path1, input_dim=5)
            
            model2 = CustomSequential()
            model2.ft_add(DenseLayer(units=10, activation="relu"))
            model2.ft_add(DenseLayer(units=5, activation="relu"))
            model2.ft_add(DenseLayer(units=1, activation="sigmoid"))
            model2.ft_build(input_dim=5)
            model2.ft_save(model_path2)
            
            create_test_csv(csv_path, n_samples=30, n_features=5)
            
            predictor1 = CustomPredictor(model_path1)
            loss1 = predictor1.ft_predict_file(csv_path)
            
            predictor2 = CustomPredictor(model_path2)
            loss2 = predictor2.ft_predict_file(csv_path)
            
            assert isinstance(loss1, float)
            assert isinstance(loss2, float)
            assert loss1 >= 0
            assert loss2 >= 0
        finally:
            if os.path.exists(model_path1):
                os.remove(model_path1)
            if os.path.exists(model_path2):
                os.remove(model_path2)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_reproducibility(self):
        """Test de reproductibilité."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=10)
            create_test_csv(csv_path, n_samples=25, n_features=10)
            
            predictor1 = CustomPredictor(model_path)
            loss1 = predictor1.ft_predict_file(csv_path)
            
            predictor2 = CustomPredictor(model_path)
            loss2 = predictor2.ft_predict_file(csv_path)
            
            assert np.isclose(loss1, loss2, rtol=1e-10)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_predict_file_handles_edge_cases(self):
        """Test avec des cas limites."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_model(model_path, input_dim=5)
            
            create_test_csv(csv_path, n_samples=5, n_features=5)
            
            predictor = CustomPredictor(model_path)
            loss = predictor.ft_predict_file(csv_path)
            
            assert isinstance(loss, float)
            assert loss >= 0
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

