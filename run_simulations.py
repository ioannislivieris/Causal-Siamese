"""
Demo script for CausalSiamese

- Load training / testing data from Data/train.npz and Data/test.npz
- Train a CausalSiamese model
- Evaluate using ATE error and PEHE
"""
import os
import random
import warnings
import numpy as np
from utils.metrics import PEHE, ATE
from models.causal_siamese import CausalSiamese

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = "Data"
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.makedirs("Results", exist_ok=True)
M_values = range(1, 11, 1)  # Example M values to evaluate
n_iterations_values = [1, 2, 3, 4, 5]  # Example iteration values to evaluate

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_dataset(path: str):
    """Load dataset from a NumPy .npz file."""
    data = np.load(path)
    return data["X"], data["T"], data["Y"], data["potential_Y"]


trainX, trainT, trainY, train_potential_Y = load_dataset(
    os.path.join(DATA_DIR, "train.npz")
)
testX, testT, testY, test_potential_Y = load_dataset(os.path.join(DATA_DIR, "test.npz"))

print("[INFO] Dataset loaded successfully")
print(f"[INFO] Training samples : {trainX.shape[0]}")
print(f"[INFO] Testing samples  : {testX.shape[0]}")

# -----------------------------------------------------------------------------
# Model Configuration and Training
# -----------------------------------------------------------------------------
results = []
for n_iterations in n_iterations_values:
    model = CausalSiamese(
        hidden_dim=50,  # backbone units
        distance="L2",  # 'L1' or 'L2'
        reg_l2=1e-2,
        margin=5,
        alpha=0.5,
        n_iterations=n_iterations,
        lr_adam=5e-5,
        lr_sgd=5e-5,
        momentum=0.9,
        nesterov=True,
        batch_size=16,
        epochs_adam=5,
        epochs_sgd=100,
        patience=15,
    )

    model.fit(X=trainX, t=trainT, y=trainY)

    # -----------------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------------
    L_ATE, L_PEHE = [], []
    for M in M_values:
        predictions = model.predict(X=testX, M=M)

        predicted_potential_outcomes = np.column_stack(
            [predictions["y0_hat"], predictions["y1_hat"]]
        )

        # -----------------------------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------------------------
        true_ate = (test_potential_Y[:, 1] - test_potential_Y[:, 0]).mean()
        ate_error = ATE(test_potential_Y, predicted_potential_outcomes)
        pehe_score = PEHE(test_potential_Y, predicted_potential_outcomes)

        print(f"\n[RESULTS - M = {M}]")
        print(f"True ATE     : {true_ate:.3f}")
        print(f"ATE Error    : {ate_error:.3f}")
        print(f"PEHE         : {pehe_score:.3f}")

        L_ATE.append(ate_error)
        L_PEHE.append(pehe_score)

        results.append(
            {
                "n_iterations": n_iterations,
                "similarity_distance": model.distance,
                "results": {"M": M, "ATE_error": ate_error, "PEHE": pehe_score},
            }
        )

# %% Store results
import json

with open("Results/causal_siamese_results.json", "w") as f:
    json.dump(results, f, indent=4)


import json

with open("Results/causal_siamese_results.json", "r") as f:
    results = json.load(f)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
# ---- ATE ERROR ----
plt.subplot(1, 2, 1)
for n_iterations in n_iterations_values:
    filtered = [r for r in results if r["n_iterations"] == n_iterations]

    M_vals = [r["results"]["M"] for r in filtered]
    ATE_vals = [r["results"]["ATE_error"] for r in filtered]

    plt.plot(M_vals, ATE_vals, marker="o", label=f"n_iterations={n_iterations}")
plt.title("ATE Error vs M (Causal Siamese)")
plt.xlabel("M")
plt.ylabel("ATE Error")
plt.legend()

# ---- PEHE ----
plt.subplot(1, 2, 2)
for n_iterations in n_iterations_values:
    filtered = [r for r in results if r["n_iterations"] == n_iterations]

    M_vals = [r["results"]["M"] for r in filtered]
    PEHE_vals = [r["results"]["PEHE"] for r in filtered]

    plt.plot(M_vals, PEHE_vals, marker="o", label=f"n_iterations={n_iterations}")
plt.title("PEHE vs M (Causal Siamese)")
plt.xlabel("M")
plt.ylabel("PEHE")
plt.legend()

# Save figure
plt.tight_layout()
plt.savefig("Results/causal_siamese_results.png", dpi=300)
plt.show()
