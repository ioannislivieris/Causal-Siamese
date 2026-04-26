# Causal-Siamese: Similarity-Based Treatment Effect Estimation with Limited Data

This repository contains the PyTorch implementation of **Causal-Siamese**, a Siamese neural network architecture for treatment effect estimation that learns a causally-aware similarity metric to estimate individual potential outcomes from limited observational data.

> **Published in:** *Evolving Systems*, Springer, 2026

> **Demo video:** https://youtu.be/s8xurU1_z0o
---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Backbone Network](#backbone-network)
  - [Three-Head Design](#three-head-design)
  - [Loss Function](#loss-function)
- [Training Procedure](#training-procedure)
- [Inference — Algorithm 2](#inference--algorithm-2)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Data Format](#data-format)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Accurate estimation of individual treatment effects from observational data is a central challenge in medicine, economics, and social sciences. Standard approaches rely on large datasets with sufficient overlap between treatment and control groups — an assumption that often fails in practice.

**Causal-Siamese** addresses treatment effect estimation under limited data by:

- Learning a **causally-informed similarity metric** via contrastive training, ensuring that "similar" instances are comparable with respect to confounding variables
- Estimating potential outcomes **Ŷ(0)** and **Ŷ(1)** by averaging the outcomes of the most similar instances in the respective control and treatment groups
- Jointly estimating the **propensity score** as an auxiliary objective that encourages the backbone to learn balanced, confounding-aware representations

This approach differs fundamentally from standard kNN methods, which rely on fixed distance metrics. Causal-Siamese learns a similarity metric end-to-end, ensuring that identified similar instances are truly comparable with respect to confounding.

---

## Model Architecture

### Backbone Network

Both components of each input pair — the anchor instance **x₀** and the comparison instance **x₁** — are encoded by a **shared backbone network** into fixed-size embeddings Z(**x₀**) and Z(**x₁**). The backbone is composed of dense layers with ELU activations.

### Three-Head Design

The model outputs three quantities simultaneously:

| Head | Output | Description |
|------|--------|-------------|
| **Similarity head** | ŝ(p, 0), ŝ(p, 1) | Conditional similarities for control and treatment groups, shape `(batch, 2)` |
| **Propensity head** | ĝ(**x₀**) | Propensity score P(T=1\|X) for the anchor instance, scalar in [0, 1] |
| **Epsilon layer** | ε | Trainable scalar parameter broadcast over the batch |

The similarity head takes the Lₚ distance between Z(**x₀**) and Z(**x₁**), passes it through batch normalisation, and applies a 2-neuron dense layer with sigmoid activations to produce conditional similarities for both treatment groups.

### Loss Function

The model is trained by minimising:

$$\mathcal{L}(\theta; P) = \frac{1}{n} \sum_i \left[ \alpha \cdot \mathcal{L}_1(\theta; p_i) + (1-\alpha) \cdot \mathcal{L}_2(\theta; p_i) \right]$$

where:

- **L₁** — Binary cross-entropy on the propensity head, used to maximise the likelihood of observed treatment assignments and encourage balanced backbone representations
- **L₂** — Contrastive loss on the similarity head, encouraging instances with similar outcomes to be close in embedding space and dissimilar instances to be separated beyond margin *m*

$$\mathcal{L}_2(\theta; p_i) = \frac{1}{2} \left( (1 - s_i) \cdot S^{nn}(p_i, t_{a_i}; \theta)^2 + s_i \cdot \max\{0,\ m - S^{nn}(p_i, t_{a_i}; \theta)\}^2 \right)$$

---

## Training Procedure

Training follows a two-phase strategy:

1. **Adam warm-up** — a small number of epochs with the Adam optimiser to initialise the network in a stable region of the loss landscape
2. **SGD fine-tuning** — the main training phase using SGD with Nesterov momentum, `ReduceLROnPlateau` scheduling, and early stopping on validation loss

Training pairs are constructed following **Algorithm 1** from the paper: for each instance in the control and treatment groups, one positive pair (same outcome, similarity label 0) and one negative pair (different outcome, similarity label 1) are created. This process can be repeated across multiple iterations to increase pair diversity.

---

## Inference — Algorithm 2

After training, potential outcomes are estimated for each test instance **x** as follows:

1. Compute the similarity score ŝ(**x**, **x_C**) between **x** and every instance in the **control group**
2. Select the *M* instances with the lowest similarity scores (most similar) and average their outcomes → **Ŷ(x, 0)**
3. Repeat for the **treatment group** → **Ŷ(x, 1)**

The parameter *M* controls the neighbourhood size and is specified at inference time.

---

## Project Structure

```
.
├── models/
│   └── causal_siamese.py      # CausalSiamese implementation (PyTorch)
├── utils/
│   └── metrics.py             # Evaluation metrics (PEHE, ATE)
├── run_causal_siamese.py      # Demo script — single run
├── run_simulations.py         # Simulation script — sweep over M and n_iterations
├── requirements.txt           # Python dependencies
├── LICENSE
├── README.md
└── Data/                      # Data directory (not included)
    ├── train.npz
    └── test.npz
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Dependencies

```
torch==2.11.0
numpy==2.4.4
scikit-learn==1.8.0
tqdm==4.67.3
matplotlib==3.10.8
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

**Single run:**
```bash
python run_causal_siamese.py
```

**Simulation sweep** (over M and n_iterations, saves results and plots to `Results/`):
```bash
python run_simulations.py
```

---

## Data Format

The model expects NumPy `.npz` files with the following keys:

```python
{
    'X':           np.ndarray,  # Covariates,            shape (n, d)
    'T':           np.ndarray,  # Treatment indicator,   shape (n,), binary 0/1
    'Y':           np.ndarray,  # Observed outcome,      shape (n,), binary 0/1
    'potential_Y': np.ndarray   # Ground-truth outcomes, shape (n, 2)
                                # col 0: Y(0),  col 1: Y(1)
}
```

> `potential_Y` is required only for evaluation and is typically available in semi-synthetic benchmark settings (e.g., Synthetic, TWINS).

---

## Usage

### Basic usage

```python
from models.causal_siamese import CausalSiamese
import numpy as np

# Load data
data  = np.load('Data/train.npz')
X, T, Y = data['X'], data['T'], data['Y']

# Instantiate and train
model = CausalSiamese(
    hidden_dim=50,
    distance="L1",
    margin=1,
    alpha=0.5,
    epochs_adam=3,
    epochs_sgd=50,
    patience=15,
)
model.fit(X=X, t=T, y=Y)

# Predict — M is specified at inference time
test = np.load('Data/test.npz')
predictions = model.predict(X=test['X'], M=5)

y0_hat = predictions['y0_hat']   # Predicted Y(0), shape (n_test,)
y1_hat = predictions['y1_hat']   # Predicted Y(1), shape (n_test,)
```

### With an explicit validation set

```python
model.fit(X=X_train, t=T_train, y=Y_train,
          valX=X_val, valT=T_val, valY=Y_val)
```

### Sweeping over M at inference

```python
for M in range(1, 11):
    predictions = model.predict(X=X_test, M=M)
    # evaluate ...
```

---

## Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_dim` | Backbone embedding dimension | `50` |
| `distance` | Distance metric for similarity: `'L1'` or `'L2'` | `'L1'` |
| `reg_l2` | L2 weight decay applied to the similarity head | `1e-2` |
| `margin` | Contrastive loss margin *m* | `1.0` |
| `alpha` | Loss weight: `alpha · contrastive + (1-alpha) · BCE` | `0.5` |
| `n_iterations` | Number of pair-creation iterations (Algorithm 1) | `1` |
| `lr_adam` | Adam learning rate (warm-up phase) | `5e-5` |
| `lr_sgd` | SGD learning rate (fine-tuning phase) | `5e-5` |
| `momentum` | SGD momentum | `0.9` |
| `nesterov` | Nesterov momentum for SGD | `True` |
| `batch_size` | Training batch size | `32` |
| `epochs_adam` | Number of Adam warm-up epochs | `3` |
| `epochs_sgd` | Maximum number of SGD fine-tuning epochs | `50` |
| `patience` | Early stopping patience (SGD phase) | `15` |
| `device` | `'cpu'`, `'cuda'`, or `None` (auto-detect) | `None` |

**Inference parameter:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Number of nearest neighbours used to average outcomes (Algorithm 2) | `5` |

> **Tuning `M`**: Larger M increases stability but may introduce bias if similar instances become less representative. The `run_simulations.py` script sweeps M from 1 to 10 and plots ATE error and PEHE for each value, making it easy to select the best M for a given dataset.

> **Tuning `n_iterations`**: Increasing the number of pair-creation iterations augments the training set with more diverse pairs. For small datasets, values of 2–5 typically improve performance. The simulation script also sweeps over `n_iterations` ∈ {1, 2, 3, 4, 5}.

---

## Evaluation Metrics

### PEHE (Precision in Estimation of Heterogeneous Effect)

$$\text{PEHE} = \frac{1}{n} \sum_i \left[ \left( \tau(\mathbf{x}_i) - \hat{\tau}(\mathbf{x}_i) \right)^2 \right], \quad \tau(\mathbf{x}) = Y_1(\mathbf{x}) - Y_0(\mathbf{x})$$

```python
from utils.metrics import PEHE
pehe = PEHE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

### ATE Error

$$|\epsilon_{\text{ATE}}| = \left| \frac{1}{n} \sum_i (Y_1(\mathbf{x}_i) - Y_0(\mathbf{x}_i)) - \frac{1}{n} \sum_i (\hat{Y}_1(\mathbf{x}_i) - \hat{Y}_0(\mathbf{x}_i)) \right|$$

```python
from utils.metrics import ATE
ate_error = ATE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{causal_siamese,
  title={Causal-Siamese: Similarity-Based Treatment Effect Estimation with Limited Data},
  author={Livieris, Ioannis E and Kiriakidou, Niki and Diou, Christos},
  journal={Evolving Systems},
  year={2026},
  publisher={Springer}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ioannis E. Livieris** — livieris@uop.gr

For questions or issues, please open an issue on the repository or contact the authors directly.
