# Exercices Python — Linear Algebra (Goodfellow, Chapitre 2)

> Convention : texte explicatif en français, **termes techniques en anglais britannique** (matrix, vector, Hadamard product, QR decomposition, orthonormal, orthogonal, normalise, eigen, etc.).

---

## 2.1 — Scalars, Vectors, Matrices and Tensors

### Exercice 1

**Énoncé :** 
Créer un *scalar*, un *vector* 1‑D, une *matrix* 2‑D et un *tensor* 3‑D.  

**Instructions :**
- Choisir des `dtype` explicites (ex. `np.float32`).  
- Afficher pour chacun : `value`, `np.ndim`, `np.shape`, `dtype`.  
- (Optionnel) vérifier que `scalar` a `shape == ()`.  

**Fonctions utiles :** 
- `np.array`.
- `np.asarray`.
- `np.random.rand`.
- `np.ndim`.
- `np.shape`.

---

## 2.2 — Matrix and Vector Operations

### Exercice 1 — Addition de vecteurs

**Énoncé :** 
Additionner deux *vectors* de même taille, élément par élément.  

**Instructions :** 
- Vérifie que les deux vecteurs ont la même forme.
- Utilise une méthode explicite (pas `a + b` directement).

**Fonctions utiles :** 
- `np.add`.
- `np.shape`.

---

### Exercice 2 — Hadamard product

**Énoncé :** 
Multiplier deux *matrices* de même forme, élément par élément (≠ produit matriciel).  

**Instructions :** 
- Vérifie la compatibilité des dimensions.
- Applique une multiplication élément-par-élément (≠ produit matriciel).

**Fonctions utiles :** 
- `np.asarray`
- `np.multiply`
- `np.shape`

---

### Exercice 3 — Centrage d’un vecteur

**Énoncé :** 
Centrer un *vector* en retirant sa moyenne.  

**Instructions :** 
- Convertis le vecteur si besoin.
- Calcule la moyenne du vecteur.
- Soustrais la moyenne à chaque élément.

**Fonctions utiles :** 
- `np.mean`
- `np.asarray`

---

### Exercice 4 – Vérification du centrage

**Énoncé :** 
Vérifie que la somme des éléments du vecteur centré est (presque) nulle.

**Instructions :** 
- Calcule la somme du vecteur centré.
- Compare avec 0 avec une tolérance.

**Fonctions utiles :** 
- `np.sum`
- `np.abs`

---
## 2.3 — Multiplying Matrices and Vectors

### Exercice 1 — Produit matrice-vecteur

**Énoncé :**  
Écris une fonction qui effectue le produit $y = A \times x$ entre une matrice $A$ et un vecteur $x$, **sans utiliser** `@` ni `np.dot`.

**Instructions :**
- Implémente-le à la main avec des boucles `for` et des sommes.
- Vérifie que les dimensions sont compatibles (`A.shape[1] == x.shape[0]`).

**Fonctions utiles :**  
`np.shape`, `np.asarray`

---

### Exercice 2 — Vérification de l’associativité

**Énoncé :**  
Vérifie numériquement que :  
$$(A @ B) @ x \approx A @ (B @ x)$$

**Instructions :**
- Crée trois matrices/vecteurs compatibles (`A`, `B`, `x`).
- Calcule les deux côtés.
- Compare les résultats avec une tolérance.

**Fonctions utiles :**  
`np.allclose`, `np.matmul` (ou `@`), `np.linalg.norm`

---

### Exercice 3 — Vérification de la distributivité

**Énoncé :**  
Vérifie que :  
$$A @ (x + y) \approx A @ x + A @ y$$

**Instructions :**
- Crée `A`, `x`, `y` compatibles.
- Calcule les deux membres.
- Compare les résultats.

**Fonctions utiles :**  
`np.add`, `np.matmul`, `np.allclose`

---

## 2.4 — Identity and Inverse Matrices

**Objectif :**  
Explorer les matrices identité et inverses avec NumPy.

**Instructions :**
- Créer une matrice carrée `A` (3×3) avec des `float32` au hasard (ex. `np.random.rand(3, 3)`) ou une matrice fixe.
- Créer une matrice identité `I` de même taille que `A` avec `np.eye(...)`.
- Vérifie que $A \times I = A$ (produit matriciel avec `np.matmul` ou `@`) pour valider le rôle de la matrice identité.
- Essaie d’inverser `A` avec `np.linalg.inv(A)` :
    - Si l’inverse existe, stocke-le dans `A_inv`.
    - Vérifie que $A @ A_{inv} \approx I$ avec `np.allclose(...)`.
    - Affiche la matrice inverse et le résultat de la multiplication $A @ A_{inv}$.
- *(Optionnel)* Gère l’exception si `A` est non inversible (`LinAlgError`) avec un `try/except`.

**Fonctions / objets à utiliser :**  
`np.eye`, `np.linalg.inv`, `np.matmul` ou `@`, `np.allclose`  
*(optionnel)* `try` + `except np.linalg.LinAlgError`

---

## 2.5 — Linear Dependence and Span

### Exercice 1 — Rang d’une matrice

**Énoncé :**  
Calcule le rang d’une matrice `A` et interprète ce que ça signifie en termes de dépendance linéaire.

**Instructions :**
- Crée une matrice `A` avec certaines colonnes dépendantes.
- Utilise la fonction de rang.
- Vérifie : si `rank(A) < n_cols`, les colonnes sont linéairement dépendantes.

**Fonctions utiles :**  
`np.linalg.matrix_rank`

---

### Exercice 2 — Vecteur dans le span

**Énoncé :**  
Teste si un vecteur `v` appartient au span des colonnes d’une matrice `B`.

**Instructions :**
- Résous le système $B @ x \approx v$ (méthode des moindres carrés).
- Vérifie si la norme du résidu est ≈ 0.

**Fonctions utiles :**  
`np.linalg.lstsq`, `np.allclose` ou `np.linalg.norm`

---

### Exercice 3 — Dépendance explicite

**Énoncé :**  
Montre explicitement une combinaison linéaire non triviale entre des vecteurs.

**Instructions :**
- Construis 3 vecteurs en 2D (par ex. $(1,0)$, $(0,1)$, $(1,1)$).
- Montre qu’il existe des coefficients $c_1, c_2, c_3$ (pas tous nuls) tels que  
    $c_1 \cdot v_1 + c_2 \cdot v_2 + c_3 \cdot v_3 = 0$.

**Fonctions utiles :**  
`np.array`, `np.linalg.matrix_rank` (pour confirmer la dépendance)

---

## 2.6 — Norms

### Exercice 1 — Lᵖ norm (implémentation)

**Énoncé :** 

Implémenter `lp_norm(x, p)` pour `p ≥ 1` et comparer à `np.linalg.norm`.  

**Instructions :** 
- Écris ta propre fonction `lp_norm(x, p)`.
- Compare avec `np.linalg.norm(x, ord=p)`.

**Fonctions utiles :** 
- `np.abs`
- `np.sum`
- `np.power`
- `np.max`
- `np.linalg.norm`.

---

### Exercice 2 — Comparaison L¹, L², L∞

**Énoncé :** 
Pour un même vecteur, calcule et compare ses normes :
- L¹ (somme des valeurs absolues)
- L² (distance euclidienne)
- L∞ (valeur absolue maximale)

**Instructions :** 
- Crée un vecteur x.
- Calcule et affiche chaque norme.
- Explique la différence géométrique entre elles.

**Fonctions utiles :** 
- `np.linalg.norm (ord=1, 2, np.inf)`.
- `np.max`.

---

### Exercice 3 — Normalisation de vectors

**Énoncé :** 
Normaliser chaque *row vector* par sa L² norm.  

**Instructions :** 
- Crée une matrice 2D (plusieurs vecteurs ligne).
- Normalise chaque vecteur ligne.
- Gère le cas où la norme est 0 (évite division par 0).

**Fonctions utiles :** 
- `np.linalg.norm(axis=1, keepdims=True)`. 
- `np.where`, diffusion (broadcasting).

---

## 2.7 — Special Kinds of Matrices and Vectors

### Exercice 1 — Symmetric / Skew‑symmetric

**Énoncé :** 
Tester si `A` est **symmetric** (`A ≈ Aᵀ`) ou **skew‑symmetric** (`A ≈ −Aᵀ`).  

**Instructions :**
- Crée une matrice carrée `A`.
- Vérifie séparément `A ≈ A.T` et `A ≈ -A.T` (même tolérance).

**Fonctions utiles :** 
- `np.allclose`.
- `A.T` (transpose).
- (optionnel) construction de `A_sym = (A + A.T)/2`, `A_skew = (A - A.T)/2`.

---

### Exercice 2 — Diagonal matrix et application efficace

**Énoncé :**
À partir d’un vecteur d, construis `D = diag(d)` et compare `D @ x` avec le produit élément-par-élément `d * x`.

**Instructions :**
- Crée `d` (1-D) et un vecteur x compatible.
- Construis la diagonal matrix à partir de `d`.
- Calcule `D @ x` et `d * x` et vérifie l’égalité (≈).

**Fonctions utiles :** 
- `np.diag`.
- `@` ou `np.matmul`.
- `np.allclose`

---

### Exercice 3 — Orthonormal basis via QR

**Énoncé :**
Construis une orthonormal basis `Q` à partir d’une matrice pleine colonne via QR decomposition et vérifie `QᵀQ ≈ I`.

**Instructions :** 
- Crée une matrice `A` (m×n, rang plein, m≥n).
- Fais `Q, R = np.linalg.qr(A)` (mode réduit par défaut).
- Vérifie `Q.T @ Q ≈ Iₙ` et que `R` est **upper-triangular** (tolérance sur les éléments sous-diagonaux).

**Fonctions utiles :** 
- `np.linalg.qr`.
- `np.eye`.
- `np.tril`, `np.allclose`.

---

### Exercice 4 — Orthogonal matrix

**Énoncé :**
Teste si une matrice `Q` est orthogonal (colonnes orthonormées) : `QᵀQ ≈ I` et `Q⁻¹ ≈ Qᵀ`.  

**Instructions :**
- Utilise `Q` issu de l’Ex.3 (ou construis-en un).
- Calcule `Q.T @ Q` et compare à `I`.
- Calcule `np.linalg.inv(Q)` et compare à `Q.T`.

**Fonctions utiles :** 
- `np.allclose`. 
- `np.linalg.inv`.
- `np.eye`.

---

### Exercice 5 — Orthogonality & unit vectors

**Énoncé :**
Vérifie l’orthogonality de deux vecteurs `u, v` (`uᵀv ≈ 0`) et la propriété unit norm (`‖u‖₂ = 1`).

**Instructions :**
- Normalise `u` et `v` si nécessaire.
- Calcule `u.T @ v` et `np.linalg.norm(u, 2)`.
- Conclus sur orthogonalité et norme unitaire.

**Fonctions utiles :** 
- `np.linalg.norm`.
- `np.dot` ou `@`.
- (optionnel) `u / np.linalg.norm(u)`

---

# 2.8 — Eigendecomposition

## Exercice 1 — Calcul des eigenvalues et eigenvectors

**Énoncé :**  
Étant donné une matrice carrée `A`, calcule ses **eigenvalues** et **eigenvectors**.

**Instructions :**
- Construire une matrice carrée `A` (ex. 3×3).  
- Utiliser `np.linalg.eig(A)` pour obtenir les `λ` et les `v`.  
- Pour chaque couple `(λᵢ, vᵢ)`, vérifier que `A @ vᵢ ≈ λᵢ · vᵢ`.  
- Utiliser une tolérance (`np.allclose`) car les calculs sont numériques.

**Fonctions utiles :**
- `np.linalg.eig(A)`  
- `np.allclose`, `@`  

---

## Exercice 2 — Reconstruction via eigendecomposition

**Énoncé :**  
Recompose `A` à partir de ses eigenvectors et eigenvalues.

**Instructions :**
- Appelle `eigvals, eigvecs = np.linalg.eig(A)`  
- Forme `Λ = np.diag(eigvals)`  
- Forme `V = eigvecs` (matrice colonnes)  
- Calcule `V @ Λ @ V⁻¹` et compare à `A`  
- Vérifie `A ≈ V @ Λ @ V⁻¹` avec tolérance

**Fonctions utiles :**
- `np.diag`, `np.linalg.inv`, `np.allclose`  

---

## Exercice 3 — Eigendecomposition de matrices symétriques

**Énoncé :**  
Pour toute matrice réelle et symétrique, les eigenvectors sont **orthonormaux**.

**Instructions :**
- Construire une matrice symétrique `A` (ex. `A = A + A.T`)  
- Utiliser `np.linalg.eigh(A)` (optimisé pour les matrices symétriques)  
- Récupérer `Q = eigvecs`  
- Vérifier que `Q.T @ Q ≈ I`

**Fonctions utiles :**
- `np.linalg.eigh`  
- `np.eye`, `np.allclose`

---

## Exercice 4 — Détecter si une matrice est singulière

**Énoncé :**  
Une matrice est **singular** (non inversible) ssi un eigenvalue est ≈ 0.

**Instructions :**
- Calculer les `eigvals` de `A`  
- Tester `np.any(np.isclose(eigvals, 0.0))`  
- Si oui → `A` est singulière.

**Fonctions utiles :**
- `np.linalg.eig`, `np.isclose`, `np.any`  

---

## Exercice 5 — Quadratic form et eigenvalue maximale

**Énoncé :**  
Considère `f(x) = xᵀ A x` avec `‖x‖₂ = 1`.  
Le **maximum de f(x)** est atteint pour `x = eigenvector_max` et vaut `λ_max`.

**Instructions :**
- Construire une matrice symétrique `A`  
- Calculer ses eigenvalues avec `np.linalg.eigh`  
- Générer plusieurs vecteurs `x` aléatoires normalisés  
- Calculer `f(x)` pour chacun  
- Vérifier que `max(f(x)) ≈ max(eigvals)`

**Fonctions utiles :**
- `np.random.randn`, `np.linalg.norm`, `np.max`  
- `x.T @ A @ x` ou `np.dot(x, A @ x)`

---

# 2.9 — Singular Value Decomposition (SVD)

> On utilise la factorisation matricielle suivante :  
> **A = U · Σ · Vᵀ**, où :
> - `U` est une matrice orthogonale (colonnes = left singular vectors)
> - `Σ` est diagonale avec les **singular values**
> - `Vᵀ` contient les right singular vectors (transposés)

---

## Exercice 1 — Décomposer une matrice A en U, Σ, Vᵀ

**Énoncé :**  
Effectuer la décomposition SVD d’une matrice réelle `A`.

**Instructions :**
- Construire une matrice `A` (pas nécessairement carrée).
- Utiliser `np.linalg.svd(A) → U, S, Vt`
- Afficher les formes (`shape`) de `U`, `S`, `Vt` et vérifier la relation `A ≈ U · Σ · Vᵀ`

**Fonctions utiles :**
- `np.linalg.svd(A)`  
- `np.diag(S)` ou `np.zeros(A.shape)` + `np.fill_diagonal()`  
- `@`, `np.allclose`

---

## Exercice 2 — Interprétation géométrique des singular values

**Énoncé :**  
Les singular values représentent le **facteur d’étirement** maximal de `A` selon chaque direction.

**Instructions :**
- Générer plusieurs vecteurs `x` normés aléatoires (‖x‖₂ = 1)
- Calculer `‖A @ x‖₂` pour chacun
- Montrer que le maximum de ces normes est ≈ `S[0]` (plus grande singular value)

**Fonctions utiles :**
- `np.random.randn`, `np.linalg.norm`, `np.max`  
- `np.linalg.svd`  

---

## Exercice 3 — Reconstruction approchée de A

**Énoncé :**  
Reconstituer une **approximation de rang k** de `A` à partir de ses premiers vecteurs singuliers.

**Instructions :**
- Prendre les `k` premiers vecteurs de `U`, `S`, `Vt`
- Construire `A_k = U_k @ Σ_k @ Vt_k`
- Comparer `A_k` à `A`

**Fonctions utiles :**
- slicing `U[:, :k]`, `S[:k]`, `Vt[:k, :]`  
- `np.diag` ou `np.diagflat` pour `Σ_k`

---

## Exercice 4 — Compression par réduction de rang

**Énoncé :**  
Comparer la taille mémoire de `A` et de sa version approchée `A_k`.

**Instructions :**
- Calculer `size_A = m·n`
- Calculer `size_Ak = m·k + k + k·n` (pour stocker `U_k`, `S_k`, `Vt_k`)
- Afficher le taux de compression

**Fonctions utiles :**
- `np.prod(A.shape)`, opérations simples

---

## Exercice 5 — Réduction de bruit (denoising)

**Énoncé :**  
Appliquer SVD à une matrice bruitée et reconstruire une version "propre".

**Instructions :**
- Ajouter du bruit à une matrice `A` (`A_noisy = A + noise`)
- Faire la SVD de `A_noisy`
- Recomposer `A_clean` avec les premiers `k` composants
- Comparer `A_clean` et `A`

**Fonctions utiles :**
- `np.random.normal(scale=σ)`, `np.linalg.svd`
- reconstruction avec `k` premiers composants
- `np.linalg.norm(A_clean - A)`

---

# 2.10 — The Moore-Penrose Pseudoinverse

> La pseudoinverse est une généralisation de l’inverse matriciel pour les matrices non carrées ou singulières.  
> Notée `A⁺`, elle permet de résoudre des systèmes `Ax = b` même lorsque `A` n’est pas inversible.

---

## Exercice 1 — Calcul de la pseudoinverse

**Énoncé :**  
Calculer la **Moore–Penrose pseudoinverse** d’une matrice rectangulaire `A`.

**Instructions :**
- Crée une matrice `A` (non carrée ou de rang incomplet).
- Calcule `A_pinv = np.linalg.pinv(A)`.
- Vérifie la relation de reconstruction : `A @ A_pinv @ A ≈ A` et `A_pinv @ A @ A_pinv ≈ A_pinv`.

**Fonctions utiles :**
- `np.linalg.pinv`, `np.allclose`, `@`

---

## Exercice 2 — Résolution de systèmes linéaires

**Énoncé :**  
Utiliser la pseudoinverse pour résoudre un système non inversible.

**Instructions :**
- Soit `A` une matrice non carrée (ex: 3×2), et `b` un vecteur (3×1)
- Résous `x = A⁺ @ b`
- Vérifie la solution `Ax ≈ b` avec tolérance

**Fonctions utiles :**
- `np.linalg.pinv`, `np.allclose`, `np.dot`

---

## Exercice 3 — Cas sous-déterminé vs sur-déterminé

**Énoncé :**  
Étudier le comportement de la pseudoinverse sur des systèmes :
- **Sous-déterminé** : plus de variables que d’équations
- **Sur-déterminé** : plus d’équations que de variables

**Instructions :**
- Construire `A₁` (2×3) sous-déterminé, et `A₂` (4×2) sur-déterminé
- Générer des vecteurs `b₁`, `b₂`
- Résoudre avec `A⁺ @ b`
- Comparer les résidus `‖Ax - b‖`

**Fonctions utiles :**
- `np.linalg.pinv`, `np.linalg.norm`

---

## Exercice 4 — Comparaison avec lstsq

**Énoncé :**  
Comparer la solution par pseudoinverse avec celle de `np.linalg.lstsq`.

**Instructions :**
- Pour une matrice `A` non inversible et vecteur `b`, calcule :
  - `x_pinv = A⁺ @ b`
  - `x_lstsq = np.linalg.lstsq(A, b)[0]`
- Compare les deux vecteurs (et leur norme)

**Fonctions utiles :**
- `np.linalg.pinv`, `np.linalg.lstsq`, `np.allclose`

---

## Exercice 5 — Pseudoinverse via SVD

**Énoncé :**  
Implémenter manuellement la pseudoinverse via **SVD**.

**Instructions :**
- Effectuer `U, S, Vt = np.linalg.svd(A)`
- Inverser `S` : `S⁺ = 1/S` (sauf zéros)
- Construire `A⁺ = Vt.T @ diag(S⁺) @ U.T`
- Comparer à `np.linalg.pinv(A)`

**Fonctions utiles :**
- `np.linalg.svd`, `np.diag`, `np.linalg.pinv`, `np.allclose`
- `np.where`, `np.divide`

---

# 2.11 — The Trace Operator

> Le **trace** d’une matrice carrée `A` est la somme de ses éléments diagonaux.  
> Elle possède des propriétés fondamentales utiles en algèbre linéaire et en apprentissage automatique.

---

## Exercice 1 — Calcul de la trace

**Énoncé :**  
Calculer la trace d’une matrice carrée `A`.

**Instructions :**
- Créer une matrice `A` de taille `n×n`
- Utiliser `np.trace(A)` pour obtenir sa trace
- Vérifier le résultat manuellement avec `sum(A[i, i])`

**Fonctions utiles :**
- `np.trace`, slicing `[i, i]`, `np.sum`

---

## Exercice 2 — Propriété : Tr(A + B) = Tr(A) + Tr(B)

**Énoncé :**  
Vérifier que la trace est linéaire.

**Instructions :**
- Créer deux matrices carrées `A` et `B` de même dimension
- Calculer `Tr(A) + Tr(B)` et `Tr(A + B)`
- Vérifier que les deux valeurs sont égales

**Fonctions utiles :**
- `np.trace`, `np.allclose`

---

## Exercice 3 — Propriété : Tr(AB) = Tr(BA)

**Énoncé :**  
Vérifier que la trace est **invariante par permutation cyclique** (si les dimensions permettent).

**Instructions :**
- Créer deux matrices `A (n×m)` et `B (m×n)`
- Calculer `Tr(AB)` et `Tr(BA)`
- Comparer les deux résultats

**Fonctions utiles :**
- `@`, `np.trace`, `np.allclose`

---

## Exercice 4 — Trace et produit scalaire

**Énoncé :**  
Montrer que pour deux matrices `A` et `B` de même taille :

```math
Tr(AᵀB) = ∑ A_ij · B_ij = ⟨A, B⟩
```
C’est-à-dire que Tr(AᵀB) donne le produit scalaire matriciel.

**Instructions :**
- Créer deux matrices `A`, `B` de même dimension
- Calculer `Tr(AᵀB)` et `np.sum(A * B)`
- Vérifier que les deux sont égaux

**Fonctions utiles :**
- `np.trace`, `np.sum`, `*, @, .T`

---

## Exercice 5 - Trace et invariance orthogonale

**Énoncé :**

Vérifier que la trace est invariante par transformation orthogonale :

```math
T_r(Q^TAQ) = T_r(A)
```
Si `Q` est orthogonale `(QᵀQ = I)`

**Instructions :**

- Créer une matrice carrée `A`
- Générer une matrice orthogonale Q (ex: via `np.linalg.qr`)
- Calculer `Tr(Qᵀ A Q)` et `Tr(A)`
- Vérifier leur égalité

**Fonctions utiles :**
- `np.linalg.qr`, `@`, `np.trace`, `np.allclose`

---

# 2.12 — The Determinant

> Le **déterminant** est une fonction scalaire appliquée aux matrices carrées, qui reflète :
> - Le **volume** transformé par la matrice
> - La **singularité** (inversibilité)
> - Le **changement d’orientation**

---

## Exercice 1 — Calcul du déterminant

**Énoncé :**  
Calculer le déterminant d'une matrice carrée `A`.

**Instructions :**
- Créer une matrice `A` (ex: 2×2 ou 3×3)
- Utiliser `np.linalg.det(A)`
- Vérifier manuellement sur de petits cas simples (ex: matrice diagonale ou triangulaire)

**Fonctions utiles :**
- `np.linalg.det`, `np.diag`, `np.tril`, `np.triu`

---

## Exercice 2 — Déterminant et inversibilité

**Énoncé :**  
Vérifier si une matrice est inversible à partir de son déterminant.

**Instructions :**
- Créer une matrice `A`
- Calculer `det(A)`
- Si `|det(A)| > 0`, elle est inversible
- Si `det(A) = 0`, elle est singulière

**Fonctions utiles :**
- `np.linalg.det`, `np.isclose`, `np.linalg.inv`

---

## Exercice 3 — Effet d’un swap de lignes

**Énoncé :**  
Vérifier que l’échange de deux lignes change le **signe** du déterminant.

**Instructions :**
- Créer une matrice `A`
- Créer une copie `A_swapped` avec deux lignes échangées
- Comparer `det(A)` et `det(A_swapped)`

**Fonctions utiles :**
- `np.copy`, slicing `[i], [j] = [j], [i]`, `np.linalg.det`

---

## Exercice 4 — Multiplication par une constante

**Énoncé :**  
Vérifier que multiplier une ligne de `A` par `λ` multiplie le déterminant par `λ`.

**Instructions :**
- Créer une matrice `A`
- Créer une copie `A_scaled` où une ligne est multipliée par une constante `λ`
- Comparer `det(A_scaled)` et `λ * det(A)` (selon le rang de la matrice)

**Fonctions utiles :**
- `np.copy`, `np.linalg.det`, slicing

---

## Exercice 5 — Déterminant d’un produit matriciel

**Énoncé :**  
Vérifier la propriété :  
```math
det(AB) = det(A) · det(B)
```
**Instructions :**
- Créer deux matrices carrées `A` et `B`
- Calculer `det(A) * det(B)` et `det(A @ B)`
- Comparer les deux résultats

**Fonctions utiles :**
- `np.linalg.det`, `@`, `np.allclose`

---

# PCA — Application complète (chapitre 2.12 final)

> Objectif : Appliquer une Analyse en Composantes Principales (PCA) sur des données 2D générées artificiellement. Réduire la dimension d’un jeu de données tout en conservant un maximum d’information (variance, à quel point les données s'écartent de la moyenne).

---

## 🧪 Données

**Génération des données simulées (100 points 2D)** :

```py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Génération de 100 points (introduisant une corrélation entre x et y)
X = np.random.randn(100, 2)
X[:, 1] = 2 * X[:, 0] + 0.5 * np.random.randn(100)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Nuage de points initial")
plt.axis("equal")
plt.show()
```

---

## 📌 Étape 1 — Centrer les données

**Énoncé :**
Centrer les données autour de zéro pour chaque axe (soustraire la moyenne).

**Instructions :**
- Calculer la moyenne des colonnes
- Soustraire cette moyenne de chaque point

**Fonctions utiles :**
- `np.mean`, `axis=0`, broadcasting

---

## 📌 Étape 2 — Calculer la matrice de covariance

**Énoncé :**

Calculer la matrice de covariance `Σ = (1/n) * Xᵀ @ X`

**Instructions :**
- Utiliser la version centrée de `X`
- Attention à bien transposer avant multiplication

**Fonctions utiles :**
- `np.dot`, `.T`, ou `@`

---

## 📌 Étape 3 — Décomposer en vecteurs/vecteurs propres

**Énoncé :**
Appliquer l’eigendecomposition de la matrice de covariance

**Instructions :**
- Utiliser `np.linalg.eigh` (symétrique)
- Trier les valeurs propres par ordre décroissant

**Fonctions utiles :**
- `np.linalg.eigh`, `np.argsort`, `[::-1]`

---

## 📌 Étape 4 — Réduire la dimension (1D)

**Énoncé :**
Projeter les données sur le 1er vecteur propre (plus grande valeur propre)

**Instructions :**
- Garder uniquement le premier vecteur propre `u₁`
- Calculer la projection : `X_proj = X @ u₁`

**Fonctions utiles :**
- `np.dot`, `@`, slicing

---

## 📌 Étape 5 — Visualisation

**Énoncé :**
Afficher les résultats de la projection

**Instructions :**
- Reconstituer les points projetés dans l’espace 2D pour visualiser la composante principale
- Tracer les vecteurs propres superposés au nuage initial (optionnel)

**Fonctions utiles :**
- `matplotlib.pyplot.scatter`, `plt.quiver`, `plt.arrow`

---

## Objectif final

Vérifier visuellement que :
- Le 1er vecteur propre suit la direction de plus grande variance
- La projection réduit correctement la dimension tout en capturant la structure