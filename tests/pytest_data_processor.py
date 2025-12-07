import sys
import os
import numpy as np
import pandas as pd
import pytest
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processor import DataProcessor


def create_test_csv(path: str, n_samples: int = 100, n_features: int = 5):
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


class TestDataProcessorInitialization:
    """Tests pour l'initialisation de DataProcessor."""

    def test_initialization(self):
        """Test de l'initialisation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            processor = DataProcessor(temp_path)
            assert processor.csv_path == temp_path
            assert processor.X_train is None
            assert processor.X_valid is None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestDataProcessorLoadDataset:
    """Tests pour la méthode load_dataset."""

    def test_load_dataset_basic(self):
        """Test de chargement basique d'un dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df = create_test_csv(temp_path, n_samples=50, n_features=5)
            
            processor = DataProcessor(temp_path)
            X, y = processor.ft_load_dataset()
            
            assert X.shape == (50, 5)
            assert y.shape == (50, 1)
            assert X.dtype == np.float64
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_dataset_different_sizes(self):
        """Test avec différentes tailles de dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df = create_test_csv(temp_path, n_samples=200, n_features=10)
            
            processor = DataProcessor(temp_path)
            X, y = processor.ft_load_dataset()
            
            assert X.shape == (200, 10)
            assert y.shape == (200, 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_dataset_diagnosis_column(self):
        """Test que la colonne diagnosis est correctement extraite."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df = create_test_csv(temp_path, n_samples=20, n_features=3)
            
            processor = DataProcessor(temp_path)
            X, y = processor.ft_load_dataset()
            
            assert X.shape[1] == 3
            original_diagnosis = df.iloc[:, 1].values  # Colonne 1 = diagnosis
            expected_y = (original_diagnosis == 'M').astype(float)
            assert np.array_equal(y.flatten(), expected_y)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_dataset_missing_file(self):
        """Test avec un fichier inexistant."""
        processor = DataProcessor("nonexistent_file.csv")
        with pytest.raises(FileNotFoundError):
            processor.ft_load_dataset()


class TestDataProcessorTrainValidSplit:
    """Tests pour la méthode train_valid_split."""

    def test_train_valid_split_default_ratio(self):
        """Test de split avec ratio par défaut (0.2)."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(X, y)

        assert len(X_train) == 80
        assert len(X_valid) == 20
        assert len(y_train) == 80
        assert len(y_valid) == 20
        
        assert len(X_train) + len(X_valid) == 100

    def test_train_valid_split_custom_ratio(self):
        """Test de split avec ratio personnalisé."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(X, y, valid_ratio=0.3)
        
        assert len(X_train) == 70
        assert len(X_valid) == 30

    def test_train_valid_split_different_seeds(self):
        """Test que différents seeds donnent des splits différents."""
        processor = DataProcessor("dummy.csv")
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        X_train1, _, _, _ = processor.ft_train_valid_split(X, y, seed=42)
        X_train2, _, _, _ = processor.ft_train_valid_split(X, y, seed=123)
        
        assert not np.array_equal(X_train1, X_train2)

    def test_train_valid_split_same_seed(self):
        """Test que le même seed donne le même split."""
        processor = DataProcessor("dummy.csv")
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        X_train1, y_train1, X_valid1, y_valid1 = processor.ft_train_valid_split(X, y, seed=42)
        X_train2, y_train2, X_valid2, y_valid2 = processor.ft_train_valid_split(X, y, seed=42)
        
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(X_valid1, X_valid2)
        assert np.array_equal(y_valid1, y_valid2)

    def test_train_valid_split_no_overlap(self):
        """Test qu'il n'y a pas de chevauchement entre train et valid."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(X, y, seed=42)
        
        train_set = set(tuple(row) for row in X_train)
        valid_set = set(tuple(row) for row in X_valid)
        assert len(train_set.intersection(valid_set)) == 0

    def test_train_valid_split_small_dataset(self):
        """Test avec un petit dataset."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        
        X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(X, y, valid_ratio=0.2)
        
        assert len(X_train) == 8
        assert len(X_valid) == 2


class TestDataProcessorNormalize:
    """Tests pour la méthode normalize."""

    def test_normalize_basic(self):
        """Test de normalisation basique."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 10 + 5
        X_valid = np.random.randn(20, 5) * 10 + 5
        
        X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
        
        assert np.allclose(np.mean(X_train_norm, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_norm, axis=0), 1, atol=1e-10)
        
        assert processor.X_train is not None
        assert processor.X_valid is not None
        assert np.array_equal(processor.X_train, X_train_norm)
        assert np.array_equal(processor.X_valid, X_valid_norm)

    def test_normalize_uses_train_stats(self):
        """Test que la normalisation utilise les stats du train pour valid."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 2 + 10
        X_valid = np.random.randn(20, 5) * 2 + 10
        
        X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
        
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        
        expected_valid_norm = (X_valid - train_mean) / train_std
        assert np.allclose(X_valid_norm, expected_valid_norm)

    def test_normalize_zero_std(self):
        """Test avec une feature ayant std=0 (colonne constante)."""
        processor = DataProcessor("dummy.csv")
        X_train = np.random.randn(100, 5)
        X_train[:, 2] = 5.0
        X_valid = np.random.randn(20, 5)
        X_valid[:, 2] = 5.0
        
        X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
        
        assert np.allclose(X_train_norm[:, 2], 0, atol=1e-6)

    def test_normalize_different_shapes(self):
        """Test avec des shapes différentes pour train et valid."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_valid = np.random.randn(30, 5)
        
        X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
        
        assert X_train_norm.shape == (100, 5)
        assert X_valid_norm.shape == (30, 5)

    def test_normalize_preserves_data_integrity(self):
        """Test que la normalisation préserve l'intégrité des données."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        X_valid = np.random.randn(10, 3)
        
        X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
        
        assert not np.any(np.isnan(X_train_norm))
        assert not np.any(np.isnan(X_valid_norm))
        assert not np.any(np.isinf(X_train_norm))
        assert not np.any(np.isinf(X_valid_norm))


class TestDataProcessorSaveSplit:
    """Tests pour la méthode save_split."""

    def test_save_split_basic(self):
        """Test de sauvegarde basique."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_valid = np.random.randn(20, 5)
        
        processor.ft_normalize(X_train, X_valid)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_train:
            train_path = f_train.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_valid:
            valid_path = f_valid.name
        
        try:
            processor.ft_save_split(train_path, valid_path)
            
            assert os.path.exists(train_path)
            assert os.path.exists(valid_path)
            
            X_train_loaded = np.load(train_path)
            X_valid_loaded = np.load(valid_path)
            
            assert np.array_equal(X_train_loaded, processor.X_train)
            assert np.array_equal(X_valid_loaded, processor.X_valid)
        finally:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(valid_path):
                os.remove(valid_path)

    def test_save_split_without_normalize(self):
        """Test que save_split lève une erreur si normalize n'a pas été appelé."""
        processor = DataProcessor("dummy.csv")
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_train:
            train_path = f_train.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_valid:
            valid_path = f_valid.name
        
        try:
            with pytest.raises(RuntimeError):
                processor.ft_save_split(train_path, valid_path)
        finally:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(valid_path):
                os.remove(valid_path)

    def test_save_split_preserves_data(self):
        """Test que les données sauvegardées sont identiques."""
        processor = DataProcessor("dummy.csv")
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        X_valid = np.random.randn(10, 3)
        
        processor.ft_normalize(X_train, X_valid)
        original_train = processor.X_train.copy()
        original_valid = processor.X_valid.copy()
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_train:
            train_path = f_train.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_valid:
            valid_path = f_valid.name
        
        try:
            processor.ft_save_split(train_path, valid_path)
            
            assert np.array_equal(processor.X_train, original_train)
            assert np.array_equal(processor.X_valid, original_valid)
        finally:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(valid_path):
                os.remove(valid_path)


class TestDataProcessorIntegration:
    """Tests d'intégration pour DataProcessor."""

    def test_full_pipeline(self):
        """Test d'un pipeline complet."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_csv(csv_path, n_samples=100, n_features=5)
            
            processor = DataProcessor(csv_path)
            
            X, y = processor.ft_load_dataset()
            assert X.shape[0] == 100
            assert y.shape[0] == 100
            
            X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(X, y, valid_ratio=0.2)
            assert len(X_train) == 80
            assert len(X_valid) == 20
            
            X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
            assert X_train_norm.shape == (80, 5)
            assert X_valid_norm.shape == (20, 5)
            
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_train:
                train_path = f_train.name
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_valid:
                valid_path = f_valid.name
            
            try:
                processor.ft_save_split(train_path, valid_path)
                
                assert os.path.exists(train_path)
                assert os.path.exists(valid_path)
                
                X_train_loaded = np.load(train_path)
                X_valid_loaded = np.load(valid_path)
                
                assert np.array_equal(X_train_loaded, X_train_norm)
                assert np.array_equal(X_valid_loaded, X_valid_norm)
            finally:
                if os.path.exists(train_path):
                    os.remove(train_path)
                if os.path.exists(valid_path):
                    os.remove(valid_path)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_pipeline_with_different_ratios(self):
        """Test du pipeline avec différents ratios de split."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_csv(csv_path, n_samples=200, n_features=10)
            
            processor = DataProcessor(csv_path)
            X, y = processor.ft_load_dataset()
            
            X_train, y_train, X_valid, y_valid = processor.ft_train_valid_split(
                X, y, valid_ratio=0.3, seed=42
            )
            assert len(X_train) == 140
            assert len(X_valid) == 60
            
            X_train_norm, X_valid_norm = processor.ft_normalize(X_train, X_valid)
            assert X_train_norm.shape == (140, 10)
            assert X_valid_norm.shape == (60, 10)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_pipeline_reproducibility(self):
        """Test de reproductibilité du pipeline."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            create_test_csv(csv_path, n_samples=100, n_features=5)
            
            processor1 = DataProcessor(csv_path)
            X1, y1 = processor1.ft_load_dataset()
            X_train1, y_train1, X_valid1, y_valid1 = processor1.ft_train_valid_split(
                X1, y1, seed=42
            )
            X_train_norm1, X_valid_norm1 = processor1.ft_normalize(X_train1, X_valid1)
            
            processor2 = DataProcessor(csv_path)
            X2, y2 = processor2.ft_load_dataset()
            X_train2, y_train2, X_valid2, y_valid2 = processor2.ft_train_valid_split(
                X2, y2, seed=42
            )
            X_train_norm2, X_valid_norm2 = processor2.ft_normalize(X_train2, X_valid2)
            
            assert np.array_equal(X_train_norm1, X_train_norm2)
            assert np.array_equal(X_valid_norm1, X_valid_norm2)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

