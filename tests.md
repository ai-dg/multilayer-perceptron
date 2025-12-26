```bash
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --train_out ./datasets/train.npz \
  --valid_out ./datasets/valid.npz \
  --valid_ratio 0.2 \
  --seed 42
```

```bash
python mlp.py --mode train \
  --model_path ./models/model.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp
```


```bash
python mlp.py --mode train \
  --model_path ./models/model_metrics.pkl \
  --layers 24 24 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy Precision Recall F1Score \
  --curve_prefix mlp_full
```


```bash
python mlp.py --mode train \
  --model_path ./models/model_earlystop.pkl \
  --layers 24 24 \
  --epochs 100 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --early_stopping \
  --patience 10 \
  --min_delta 0.001 \
  --curve_prefix mlp_earlystop
```
---

```bash
python mlp.py --mode train \
  --model_path ./models/model_3layers.pkl \
  --layers 24 24 24 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_3layers
```

```bash
python mlp.py --mode train \
  --model_path ./models/model_large.pkl \
  --layers 48 48 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_large
```

```bash
python mlp.py --mode train \
  --model_path ./models/model_sgd.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer SGD \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_sgd
```

```bash
python mlp.py --mode predict \
  --model_path ./models/model.pkl \
  --predict_data ./datasets/valid.npz
```
---


```bash
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --valid_ratio 0.2 \
  --seed 42

# 2. Train
python mlp.py --mode train \
  --model_path ./models/model.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp

# 3. Predict
python mlp.py --mode predict \
  --model_path ./models/model.pkl \
  --predict_data ./datasets/valid.npz
```

---

## Test 8 : Vérification de la modularité

Teste la modularité du programme avec différents paramètres.

```bash
# Test avec un seul hidden layer (sera complété automatiquement)
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model_auto.pkl \
  --layers 24 \
  --epochs 30 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_auto
```

**Résultat attendu :**
- Message : `⚠️ Le sujet demande au moins 2 hidden layers. On en ajoute automatiquement.`
- Le modèle aura quand même 2 hidden layers minimum

---

## Test 9 : Vérification des graphiques

Vérifie que les graphiques sont bien générés.

```bash
# Après un entraînement, vérifier les fichiers
ls -lh mlp_*.png

# Les fichiers doivent contenir :
# - mlp_loss.png : Courbe de loss (train et validation)
# - mlp_accuracy.png : Courbe d'accuracy (train et validation)
```

---

## Test 10 : Tests unitaires

Exécute tous les tests unitaires pour vérifier que tout fonctionne.

```bash
# Tous les tests
python -m pytest tests/ -v

# Tests spécifiques
python -m pytest tests/pytest_mlp.py -v
python -m pytest tests/pytest_custom_model.py -v
python -m pytest tests/pytest_data_processor.py -v
```

**Résultat attendu :**
- Tous les tests passent (216 tests au total)

---

## Résultats attendus typiques

### Sur le Wisconsin Breast Cancer Dataset :

- **Accuracy de validation** : 95-98%
- **Loss finale** : 0.04-0.08 (Binary Cross-Entropy)
- **Convergence** : Généralement après 40-60 epochs
- **Temps d'entraînement** : ~10-30 secondes selon la machine

### Comparaison des optimizers :

- **Adam** : Convergence plus rapide, meilleure accuracy finale
- **SGD** : Convergence plus lente, peut nécessiter plus d'epochs

### Comparaison des architectures :

- **24-24** (2 hidden layers) : Bon compromis, rapide
- **24-24-24** (3 hidden layers) : Peut améliorer légèrement l'accuracy
- **48-48** (plus de neurones) : Peut overfitter si pas assez de données

---

## Notes pour la présentation

1. **Démarrer par le Test 1** (split) pour montrer la séparation des données
2. **Ensuite le Test 2** (train) pour montrer l'entraînement avec affichage des métriques
3. **Montrer les graphiques** générés (mlp_loss.png et mlp_accuracy.png)
4. **Tester la prédiction** (Test 6) pour montrer l'évaluation finale
5. **Démontrer la modularité** (Test 8) pour montrer la flexibilité du code
6. **Exécuter les tests unitaires** (Test 10) pour montrer la robustesse

---

## Commandes rapides de référence

```bash
# Split rapide
python mlp.py --mode split --dataset ./datasets/data.csv --train_out ./datasets/train.npz --valid_out ./datasets/valid.npz

# Train rapide (configuration par défaut)
python mlp.py --mode train --train_data ./datasets/train.npz --valid_data ./datasets/valid.npz --layers 24 24 --epochs 70

# Predict rapide
python mlp.py --mode predict --model_path ./models/model.pkl --predict_data ./datasets/valid.npz
```

