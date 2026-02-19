# 📓 Guide d'utilisation du Notebook Jupyter pour Google Colab

Ce guide explique comment utiliser le notebook `MLP_Presentation.ipynb` sur Google Colab pour présenter le projet aux recruteurs.

## 🚀 Démarrage Rapide

### Méthode 1 : Clonage depuis GitHub (Recommandé)

Le notebook inclut automatiquement le clonage du repository GitHub au début.

1. **Ouvrir Google Colab**
   - Allez sur [colab.research.google.com](https://colab.research.google.com)
   - Uploadez `MLP_Presentation.ipynb` ou créez un nouveau notebook

2. **Modifier l'URL du repository**
   - Dans la première cellule d'installation, remplacez `REPO_URL` par l'URL de votre repository GitHub
   ```python
   REPO_URL = "https://github.com/votre-username/multilayer-perceptron.git"
   ```

3. **Exécuter le notebook**
   - Le notebook clonera automatiquement le repository
   - Tous les fichiers (classes Custom, modules, dataset) seront disponibles
   - Exécutez les cellules dans l'ordre (Runtime → Run All)

### Méthode 2 : Upload manuel

1. **Cloner le repository**
   ```python
   !git clone https://github.com/votre-username/multilayer-perceptron.git
   %cd multilayer-perceptron
   ```

2. **Exécuter le notebook**
   - Ouvrez `MLP_Presentation.ipynb`
   - Exécutez toutes les cellules

### Méthode 3 : Via Google Drive

1. **Uploader sur Drive**
   - Uploadez le dossier complet du projet sur Google Drive
   - Ouvrez le notebook depuis Drive avec Colab

2. **Monter Drive dans Colab**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/path/to/multilayer-perceptron
   ```

## 📋 Structure des fichiers requis

```
multilayer-perceptron/
├── MLP_Presentation.ipynb    # Notebook principal
├── custom_model.py            # Modèle séquentiel
├── custom_layer.py            # Couches denses
├── optimizers.py              # Optimiseurs
├── losses.py                  # Fonctions de perte
├── metrics.py                 # Métriques
├── callbacks.py               # Callbacks
├── data_processor.py          # Traitement des données
├── plotting.py                # Visualisations
└── datasets/
    └── data.csv               # Dataset Wisconsin Breast Cancer
```

## ⚙️ Configuration

Le notebook installe automatiquement les dépendances nécessaires :
- `numpy`
- `pandas`
- `matplotlib`
- `tabulate`

## 🎯 Fonctionnalités du Notebook

Le notebook présente :

1. **Introduction** : Contexte et objectifs du projet
2. **Architecture** : Structure du code et du réseau
3. **Composants** : Explication détaillée de chaque module
4. **Exemple complet** : 
   - Chargement des données
   - Construction du modèle
   - Entraînement
   - Visualisation des résultats
   - Évaluation
5. **Résultats** : Métriques de performance
6. **Conclusion** : Points forts et compétences développées

## 📊 Résultats attendus

Après exécution complète, vous devriez voir :

- ✅ Courbes d'apprentissage (loss et metrics)
- ✅ Métriques de performance (Accuracy ~95-98%)
- ✅ Prédictions sur l'ensemble de validation
- ✅ Graphiques sauvegardés dans `plots/`

## 🔧 Dépannage

### Erreur d'importation
- Vérifiez que tous les fichiers `.py` sont présents
- Vérifiez que vous êtes dans le bon répertoire

### Dataset introuvable
- Vérifiez que `data.csv` est dans `datasets/`
- Vérifiez le chemin du fichier

### Erreurs de visualisation
- Les graphiques sont sauvegardés dans `plots/`
- Utilisez `display(Image("plots/mlp_loss.png"))` pour les afficher

## 💡 Conseils pour la présentation

1. **Exécutez le notebook avant la présentation** pour vérifier que tout fonctionne
2. **Mettez en avant l'API Keras-like** :
   - Expliquez que vous avez créé des classes Custom (`CustomSequential`, `DenseLayer`, etc.)
   - Montrez la similarité avec l'API Keras (`compile()`, `fit()`, `predict()`, `evaluate()`)
   - Démontrez votre compréhension de l'architecture de Keras
3. **Préparez des réponses** aux questions sur :
   - La rétropropagation
   - Les optimiseurs (SGD vs Adam)
   - Le choix des hyperparamètres
   - Les métriques d'évaluation
   - Pourquoi avoir choisi une API Keras-like ?
4. **Montrez le code source** si demandé (les fichiers `.py` avec les classes Custom)
5. **Expliquez les choix techniques** :
   - Pourquoi ReLU pour les couches cachées ?
   - Pourquoi l'initialisation de He ?
   - Pourquoi Adam plutôt que SGD ?
   - Comment vous avez structuré les classes Custom pour imiter Keras ?

## 📝 Notes pour les recruteurs

Ce projet démontre :
- ✅ Compréhension approfondie des réseaux de neurones
- ✅ Compétences en Python et NumPy
- ✅ Maîtrise des mathématiques (algèbre linéaire, calcul différentiel)
- ✅ Bonnes pratiques de développement (code modulaire, documentation)
- ✅ Capacité à implémenter des algorithmes complexes depuis zéro

**Score obtenu** : 125% (mandatory + bonus)

---

Pour toute question, consultez le `README.md` principal du projet.
