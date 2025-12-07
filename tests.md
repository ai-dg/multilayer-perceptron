# Tests manuels pour la présentation

Ce document contient les commandes à exécuter manuellement pour tester le projet lors de la présentation.

## Prérequis

- Le fichier `datasets/data.csv` doit exister
- Les dossiers `datasets/` et `models/` seront créés automatiquement

---

## Test 1 : Split du dataset

Sépare le dataset en train (80%) et validation (20%).

```bash
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --train_out ./datasets/train.npz \
  --valid_out ./datasets/valid.npz \
  --valid_ratio 0.2 \
  --seed 42
```

**Résultat attendu :**
```
x_train shape : (XXX, 30)
x_valid shape : (XXX, 30)
Train saved to ./datasets/train.npz
Valid saved to ./datasets/valid.npz
```

---

## Test 2 : Entraînement du modèle (configuration par défaut)

Entraîne le modèle avec la configuration recommandée du sujet.

```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp
```

**Résultat attendu :**
- Affichage des métriques à chaque epoch :
  ```
  epoch 01/70 - loss: 0.XXXX - val_loss: 0.XXXX - accuracy: 0.XXXX - val_accuracy: 0.XXXX
  epoch 02/70 - loss: 0.XXXX - val_loss: 0.XXXX - accuracy: 0.XXXX - val_accuracy: 0.XXXX
  ...
  epoch 70/70 - loss: 0.XXXX - val_loss: 0.XXXX - accuracy: 0.XXXX - val_accuracy: 0.XXXX
  > saving model './models/model.pkl' to disk...
  Learning curves saved with prefix 'mlp_*.png'
  ```
- Génération de `mlp_loss.png` et `mlp_accuracy.png`
- Accuracy finale typique : **95-98%**

---

## Test 3 : Entraînement avec plusieurs métriques

Entraîne le modèle en affichant plusieurs métriques (bonus).

```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model_metrics.pkl \
  --layers 24 24 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy Precision Recall F1Score \
  --curve_prefix mlp_full
```

**Résultat attendu :**
- Affichage de toutes les métriques à chaque epoch
- Génération des graphiques avec toutes les métriques

---

## Test 4 : Entraînement avec Early Stopping (bonus)

Entraîne le modèle avec early stopping pour éviter le surapprentissage.

```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
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

**Résultat attendu :**
- L'entraînement s'arrête automatiquement si la validation loss ne s'améliore plus
- Message : `Early stopping at epoch XX`

---

## Test 5 : Entraînement avec différentes architectures

Teste différentes architectures de réseau.

### Architecture 1 : 3 hidden layers
```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model_3layers.pkl \
  --layers 24 24 24 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_3layers
```

### Architecture 2 : Plus de neurones
```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model_large.pkl \
  --layers 48 48 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_large
```

### Architecture 3 : SGD au lieu d'Adam
```bash
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
  --model_path ./models/model_sgd.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer SGD \
  --learning_rate 0.0314 \
  --metrics Accuracy \
  --curve_prefix mlp_sgd
```

---

## Test 6 : Prédiction et évaluation

Charge un modèle entraîné et évalue sur le dataset de validation.

```bash
python mlp.py --mode predict \
  --model_path ./models/model.pkl \
  --predict_data ./datasets/valid.npz
```

**Résultat attendu :**
```
Binary cross-entropy on dataset: 0.XXXXXX
```

---

## Test 7 : Pipeline complet

Exécute le pipeline complet : split → train → predict.

```bash
# 1. Split
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --train_out ./datasets/train.npz \
  --valid_out ./datasets/valid.npz \
  --valid_ratio 0.2 \
  --seed 42

# 2. Train
python mlp.py --mode train \
  --train_data ./datasets/train.npz \
  --valid_data ./datasets/valid.npz \
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

