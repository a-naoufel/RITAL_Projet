# Movie Review Sentiment Classification

Projet de **classification de sentiment** sur des critiques de films en anglais.
L’objectif est de prédire si une critique est **positive** ou **négative**, puis de générer un fichier de sortie au format attendu par la plateforme d’évaluation (`P` / `N`, une ligne par exemple).

---

## Overview

Ce projet compare plusieurs approches de classification de texte, des méthodes classiques aux modèles pré-entraînés :

- **TF–IDF + LinearSVC**
- **TF–IDF word + char ensemble**
- **fastText supervised** *(si ajouté dans le repo)*
- **SetFit** *(si ajouté dans le repo)*
- **DistilBERT**
- **RoBERTa**
- **DeBERTa-v3-base** *(si ajouté dans le repo)*
- **ModernBERT-base** *(optionnel, si ajouté dans le repo)*

Le meilleur résultat documenté dans les expériences actuelles est obtenu avec :

> **TF–IDF word + char ensemble + LinearSVC**

avec une accuracy de validation de **0.8925**.

---

## Dataset

Structure attendue :

```text
dataset/
├── movies1000/
│   ├── pos/   # 1000 fichiers texte
│   └── neg/   # 1000 fichiers texte
└── testSentiment.txt   # 25000 lignes, une critique par ligne
```

### Labels

- `pos/` → critique positive
- `neg/` → critique négative

### Format attendu pour les prédictions

Le fichier final doit contenir **une prédiction par ligne** :

```text
P
N
P
P
...
```

---

## Experimental Setup

- **Task**: binary sentiment classification
- **Split**: 80% train / 20% validation
- **Stratified split**
- **Random seed**: `42`
- **Main metric**: accuracy

Pré-traitement minimal :

- lecture des fichiers en UTF-8 avec tolérance sur les erreurs,
- pas de nettoyage agressif,
- conservation du signal lexical pour les modèles TF–IDF.

---

## Reported Results

| Model | Representation | Key settings | Validation Accuracy |
|---|---|---|---:|
| LinearSVC (baseline) | TF–IDF word | `(1,2)`, `C=1` | 0.8725 |
| LinearSVC (tuned) | TF–IDF word | `(1,2)`, `C=4` | 0.8850 |
| LinearSVC (char) | TF–IDF char
a_l | `(3,5)`, `C=0.5` | 0.8675 |
| **Word + char ensemble** | TF–IDF word + char | `alpha=0.9` | **0.8925** |
| DistilBERT base | Transformer | fine-tuning | 0.8225 |
| DistilBERT SST-2 | Transformer | fine-tuning from sentiment checkpoint | 0.8575 |
| RoBERTa-base | Transformer | fine-tuning | 0.8575 |
| BiLSTM + GloVe | Sequence model | GloVe 100d | ~0.78 |
| Line-based split attempt | Wrong setup | file split → line expansion | ~0.6397 |

> Remarque : les sections `fastText`, `SetFit`, `DeBERTa-v3-base` et `ModernBERT-base` peuvent être ajoutées ou complétées si leurs scripts et résultats sont présents dans le repository.

---

## Repository Structure

Organisation recommandée / attendue à partir des scripts mentionnés dans le rapport :

```text
.
├── dataset/
│   ├── movies1000/
│   │   ├── pos/
│   │   └── neg/
│   └── testSentiment.txt
├── models/
│   ├── baseline/
│   │   ├── train_tfidf_svc_ensemble.py
│   │   ├── predict_ensemble.py
│   │   └── outputs/
│   │       └── preds.txt
│   └── transformers/
│       ├── train_distilbert.py
│       ├── train_roberta.py
│       └── predict_roberta.py
├── report/
│   └── report.tex
└── README.md
```

Si ton arborescence réelle est un peu différente, adapte simplement les chemins dans ce README.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install scikit-learn numpy pandas torch transformers datasets sentencepiece
```

Selon tes scripts, tu peux aussi avoir besoin de :

```bash
pip install tqdm matplotlib
```

### 4. NumPy / PyTorch compatibility fix

Si tu rencontres une incompatibilité entre **PyTorch 2.1** et **NumPy 2.x**, utilise :

```bash
pip uninstall -y numpy
pip install "numpy<2"
```

---

## How to Run

### Baseline final model: TF–IDF word + char ensemble

Entraîner le modèle :

```bash
python3 models/baseline/train_tfidf_svc_ensemble.py
```

Générer les prédictions sur `testSentiment.txt` :

```bash
python3 models/baseline/predict_ensemble.py
```

Fichier de sortie attendu :

```text
models/baseline/outputs/preds.txt
```

---

## Transformer Models

### DistilBERT

```bash
python3 models/transformers/train_distilbert.py
```

### RoBERTa

```bash
python3 models/transformers/train_roberta.py
python3 models/transformers/predict_roberta.py
```

### Other models

Si ton repo contient des scripts pour `fastText`, `SetFit`, `DeBERTa-v3-base` ou `ModernBERT-base`, ajoute-les dans cette section en suivant le même format.

---

## Prediction File Integrity

Un point important dans ce projet est de produire **exactement 25 000 prédictions** pour les **25 000 lignes** de `testSentiment.txt`.

La stratégie correcte est :

- lire le fichier complet,
- faire un `split("\n")`,
- retirer uniquement la dernière chaîne vide si le fichier se termine par un saut de ligne.

Cela évite les erreurs classiques liées à :

- la suppression involontaire de lignes vides,
- des différences entre lecture Python et `wc -l`,
- un nombre de prédictions incorrect.

---

## Why the line-based approach was rejected

Une tentative a consisté à transformer chaque ligne interne d’un fichier en exemple indépendant.
Cette approche a fortement dégradé les performances (~0.6397), car le label porte sur la **critique complète**, pas sur chaque ligne isolée.

En pratique, les retours à la ligne dans ce corpus servent surtout à la mise en forme du texte.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Installe PyTorch :

```bash
pip install torch
```

### NumPy / PyTorch error

```bash
pip uninstall -y numpy
pip install "numpy<2"
```

### Transformer model path error

Si un script de prédiction échoue parce qu’un modèle sauvegardé n’est pas trouvé :

- vérifie le dossier exact du checkpoint,
- charge le bon chemin local,
- passe `str(path)` à `from_pretrained(...)` si nécessaire.

---

## Reproducibility

Pour reproduire les résultats :

- utiliser le même split stratifié,
- fixer `seed = 42`,
- conserver les mêmes hyperparamètres,
- vérifier que les chemins de données et de checkpoints sont corrects.

---

## Best Model Summary

Le meilleur compromis actuel sur ce corpus est :

- **Model**: TF–IDF word + char ensemble + LinearSVC
- **Validation accuracy**: **0.8925**
- **Strengths**:
  - rapide,
  - simple,
  - robuste avec peu de données,
  - plus performant que les modèles profonds testés dans cette configuration.

---

## Future Work

Pistes d’amélioration :

- ajouter les scripts et résultats pour **fastText**, **SetFit**, **DeBERTa-v3-base** et **ModernBERT-base**,
- entraîner sur un corpus plus large,
- affiner le fine-tuning des transformers,
- tester du semi-supervisé ou du pseudo-labeling,
- ajouter des visualisations (courbes train/val, matrice de confusion, comparaison des modèles).


## License

Choisis une licence adaptée pour GitHub, par exemple :

- MIT
- Apache-2.0
- ou `All rights reserved` si tu ne veux pas d’usage libre.
