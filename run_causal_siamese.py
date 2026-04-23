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
model = CausalSiamese(
    hidden_dim=50,  # backbone units
    distance="L1",  # 'L1' or 'L2'
    reg_l2=1e-2,
    margin=5,
    alpha=0.5,
    n_iterations=1,
    lr_adam=5e-5,
    lr_sgd=5e-5,
    momentum=0.9,
    nesterov=True,
    batch_size=32,
    epochs_adam=5,
    epochs_sgd=20,
    patience=15,
)

model.fit(X=trainX, t=trainT, y=trainY)

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
M = 5  # Number of nearest neighbors to use for prediction (can be tuned)
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
