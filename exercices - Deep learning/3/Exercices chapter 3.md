# Exercice 3.1 — Why Probability?

## Objectif
Simuler des incertitudes simples avec NumPy pour comprendre pourquoi on a besoin de la probabilité.

---

## Étape 1 : Stochasticité inhérente
Simule 1000 lancers de pièce équilibrée (pile ou face).
Vérifie que la proportion de "pile" tend vers 0.5.

```python
import numpy as np

np.random.seed(42)
coin_flips = np.random.choice([0, 1], size=1000)  # 0 = pile, 1 = face
print("Proportion de pile:", np.mean(coin_flips == 0))
```

## Étape 2 : Observation incomplète
Supposons que tu tires une carte parmi 3 (2 chèvres, 1 voiture).
Sans connaître où est la voiture → incertitude pour le joueur.

```py
doors = np.array([0, 0, 1])  # 0 = chèvre, 1 = voiture
choice = np.random.choice(doors)
print("Gain:", "voiture" if choice == 1 else "chèvre")
```

## Étape 3 : Modèle simplifié
- Un robot mesure une position (vraie valeur = 10).
- Mais son capteur ajoute un bruit gaussien.

```py
true_position = 10
noisy_measurements = true_position + np.random.normal(0, 2, size=1000)
print("Moyenne mesurée:", noisy_measurements.mean())
```

## À retenir
- Les phénomènes aléatoires apparaissent naturellement.
- On ne peut pas les ignorer → on a besoin de la probabilité pour raisonner dessus.