import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from callbacks import Callback, History, EarlyStopping


class TestCallback:
    """Tests pour la classe Callback de base."""

    def test_callback_initialization(self):
        """Test de l'initialisation de Callback."""
        callback = Callback()
        assert callback is not None

    def test_callback_on_train_begin(self):
        """Test de ft_clear_history."""
        callback = Callback()
        # callback.ft_on_train_begin()
        callback.ft_on_train_begin({"loss": 0.5})

    def test_callback_on_train_end(self):
        """Test de ft_on_train_end."""
        callback = Callback()
        callback.ft_on_train_end()
        callback.ft_on_train_end({"loss": 0.5})

    def test_callback_on_epoch_begin(self):
        """Test de ft_on_epoch_begin."""
        callback = Callback()
        callback.ft_on_epoch_begin(0)
        callback.ft_on_epoch_begin(5, {"loss": 0.5})

    def test_callback_on_epoch_end(self):
        """Test de ft_on_epoch_end."""
        callback = Callback()
        callback.ft_on_epoch_end(0)
        callback.ft_on_epoch_end(5, {"loss": 0.5})


class TestHistory:
    """Tests pour la classe History."""

    def test_history_initialization(self):
        """Test de l'initialisation de History."""
        history = History()
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert "accuracy" in history.history
        assert "val_accuracy" in history.history
        assert all(len(v) == 0 for v in history.history.values())

    def test_history_on_train_begin(self):
        """Test que ft_clear_history réinitialise l'historique."""
        history = History()
        history.history["loss"] = [0.5, 0.4, 0.3]
        history.ft_clear_history()
        assert len(history.history["loss"]) == 0
        assert len(history.history["val_loss"]) == 0

    def test_history_on_epoch_end_single_metric(self):
        """Test d'enregistrement d'une métrique."""
        history = History()
        history.ft_on_epoch_end(0, {"loss": 0.5})
        assert history.history["loss"] == [0.5]
        assert len(history.history["val_loss"]) == 0

    def test_history_on_epoch_end_multiple_metrics(self):
        """Test d'enregistrement de plusieurs métriques."""
        history = History()
        history.ft_on_epoch_end(0, {
            "loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.8,
            "val_accuracy": 0.75
        })
        assert history.history["loss"] == [0.5]
        assert history.history["val_loss"] == [0.6]
        assert history.history["accuracy"] == [0.8]
        assert history.history["val_accuracy"] == [0.75]

    def test_history_on_epoch_end_multiple_epochs(self):
        """Test d'enregistrement sur plusieurs époques."""
        history = History()
        history.ft_on_epoch_end(0, {"loss": 0.5, "val_loss": 0.6})
        history.ft_on_epoch_end(1, {"loss": 0.4, "val_loss": 0.5})
        history.ft_on_epoch_end(2, {"loss": 0.3, "val_loss": 0.4})
        
        assert history.history["loss"] == [0.5, 0.4, 0.3]
        assert history.history["val_loss"] == [0.6, 0.5, 0.4]

    def test_history_on_epoch_end_new_metric(self):
        """Test qu'une nouvelle métrique est automatiquement ajoutée."""
        history = History()
        history.ft_on_epoch_end(0, {"custom_metric": 0.9})
        assert "custom_metric" in history.history
        assert history.history["custom_metric"] == [0.9]

    def test_history_on_epoch_end_none_logs(self):
        """Test que None logs ne cause pas d'erreur."""
        history = History()
        history.ft_on_epoch_end(0, None)
        assert len(history.history["loss"]) == 0


class TestEarlyStopping:
    """Tests pour la classe EarlyStopping."""

    def test_early_stopping_initialization_default(self):
        """Test de l'initialisation avec valeurs par défaut."""
        es = EarlyStopping()
        assert es.monitor == "val_loss"
        assert es.patience == 5
        assert es.min_delta == 0.0
        assert es.mode == "min"
        assert es.best is None
        assert es.wait == 0
        assert es.stop_training is False

    # def test_early_stopping_initialization_custom(self):
    #     """Test de l'initialisation avec paramètres personnalisés."""
    #     es = EarlyStopping(
    #         monitor="loss",
    #         patience=10,
    #         min_delta=0.01,
    #         mode="max"
    #     )
    #     assert es.monitor == "loss"
    #     assert es.patience == 10
    #     assert es.min_delta == 0.01
    #     assert es.mode == "max"

    def test_early_stopping_on_train_begin(self):
        """Test que ft_clear_history réinitialise l'état."""
        es = EarlyStopping()
        es.best = 0.5
        es.wait = 3
        es.stop_training = True
        es.ft_on_train_begin()
        assert es.best is None
        assert es.wait == 0
        assert es.stop_training is False

    def test_early_stopping_min_mode_improvement(self):
        """Test du mode 'min' avec amélioration."""
        es = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"val_loss": 0.5})
        assert es.best == 0.5
        assert es.wait == 0
        assert es.stop_training is False
        
        es.ft_on_epoch_end(1, {"val_loss": 0.4})
        assert es.best == 0.4
        assert es.wait == 0
        assert es.stop_training is False

    def test_early_stopping_min_mode_no_improvement(self):
        """Test du mode 'min' sans amélioration."""
        es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"val_loss": 0.5})
        es.ft_on_epoch_end(1, {"val_loss": 0.6})  
        assert es.wait == 1
        assert es.stop_training is False
        
        es.ft_on_epoch_end(2, {"val_loss": 0.7})  
        assert es.wait == 2
        assert es.stop_training is True

    def test_early_stopping_max_mode_improvement(self):
        """Test du mode 'max' avec amélioration."""
        es = EarlyStopping(monitor="accuracy", mode="max", patience=3)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"accuracy": 0.5})
        assert es.best == 0.5
        
        es.ft_on_epoch_end(1, {"accuracy": 0.6})  
        assert es.best == 0.6
        assert es.wait == 0

    def test_early_stopping_max_mode_no_improvement(self):
        """Test du mode 'max' sans amélioration."""
        es = EarlyStopping(monitor="accuracy", mode="max", patience=2)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"accuracy": 0.8})
        es.ft_on_epoch_end(1, {"accuracy": 0.7})  
        assert es.wait == 1
        
        es.ft_on_epoch_end(2, {"accuracy": 0.6})  
        assert es.wait == 2
        assert es.stop_training is True

    def test_early_stopping_min_delta(self):
        """Test avec min_delta."""
        es = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.1, patience=2)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"val_loss": 0.5})
        es.ft_on_epoch_end(1, {"val_loss": 0.45})
        assert es.wait == 1  
        
        es.ft_on_epoch_end(2, {"val_loss": 0.35})
        assert es.wait == 0  

    def test_early_stopping_missing_monitor(self):
        """Test quand la métrique monitor n'est pas dans logs."""
        es = EarlyStopping(monitor="val_loss", patience=2)
        es.ft_on_train_begin()
        
        es.ft_on_epoch_end(0, {"loss": 0.5})
        assert es.best is None  

    def test_early_stopping_none_logs(self):
        """Test avec logs=None."""
        es = EarlyStopping(monitor="val_loss", patience=2)
        es.ft_on_train_begin()
        es.ft_on_epoch_end(0, None)
        assert es.best is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

