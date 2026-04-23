import time
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =============================================================================
# Distance functions
# =============================================================================

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L2 (Euclidean) distance between two embedding vectors."""
    return torch.sqrt(torch.clamp(torch.sum((x - y) ** 2, dim=1, keepdim=True), min=1e-8))


def l1_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 (Manhattan) distance between two embedding vectors."""
    return torch.abs(x - y)  # shape: (batch, embedding_dim)


# =============================================================================
# Network modules
# =============================================================================

class BackboneNetwork(nn.Module):
    """
    Shared backbone that encodes covariates into a fixed-size embedding.
    Architecture mirrors the Keras implementation: one Dense(50, elu) layer.
    Extend with more layers for larger datasets.
    """

    def __init__(self, n_covariates: int, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_covariates, hidden_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EpsilonLayer(nn.Module):
    """
    Trainable scalar epsilon — a single learnable parameter broadcast
    to match the batch size, identical to the Keras EpsilonLayer.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.randn(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast epsilon to (batch_size, 1)
        return self.epsilon.expand(x.size(0), 1)


class CausalSiameseNet(nn.Module):
    """
    Three-head architecture:
      Head 1 — propensity score  g(x_anchor)          → scalar in [0,1]
      Head 2 — conditional similarities  ŝ(p, 0/1)   → 2-dim in [0,1]
      Head 3 — epsilon (trainable scalar)
    """

    def __init__(
        self,
        n_covariates: int,
        hidden_dim: int = 50,
        reg_l2: float = 1e-2,
        distance: str = "L1",
    ):
        super().__init__()
        if distance not in ("L1", "L2"):
            raise ValueError(f"Unknown distance '{distance}'. Choose 'L1' or 'L2'.")
        self.distance   = distance
        self.hidden_dim = hidden_dim

        # Shared backbone (weights are shared for both inputs)
        self.backbone = BackboneNetwork(n_covariates, hidden_dim)

        # Head 1 — propensity score
        self.propensity_head = nn.Linear(hidden_dim, 1)

        # Epsilon layer
        self.epsilon_layer = EpsilonLayer()

        # Similarity head input size depends on distance type
        sim_input_dim = hidden_dim if distance == "L1" else 1
        self.sim_bn   = nn.BatchNorm1d(sim_input_dim)
        self.sim_head = nn.Linear(sim_input_dim, 2)

        # L2 regularisation is applied via weight_decay in the optimiser,
        # but we store it for reference.
        self.reg_l2 = reg_l2

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x0: anchor instances,     shape (batch, n_covariates)
            x1: comparison instances, shape (batch, n_covariates)

        Returns:
            sim_pred : shape (batch, 2)  — ŝ(p,0) and ŝ(p,1)
            t_pred   : shape (batch, 1)  — propensity score ĝ(x0)
            epsilons : shape (batch, 1)
        """
        z0 = self.backbone(x0)
        z1 = self.backbone(x1)

        # --- Head 1: propensity score ---
        t_pred   = torch.sigmoid(self.propensity_head(z0))

        # --- Epsilon ---
        epsilons = self.epsilon_layer(t_pred)

        # --- Head 2: similarity ---
        if self.distance == "L2":
            dist = euclidean_distance(z0, z1)       # (batch, 1)
        else:  # L1
            dist = l1_distance(z0, z1)              # (batch, hidden_dim)

        dist_bn  = self.sim_bn(dist)
        sim_pred = torch.sigmoid(self.sim_head(dist_bn))  # (batch, 2)

        return sim_pred, t_pred, epsilons


# =============================================================================
# Loss function
# =============================================================================

class CausalSiameseLoss(nn.Module):
    """
    Composite loss = α · L_contrastive  +  (1-α) · L_BCE_propensity

    L_contrastive : contrastive loss on conditional similarity head
    L_BCE         : binary cross-entropy on propensity head
    """

    def __init__(self, margin: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.margin = margin
        self.alpha  = alpha
        self.bce    = nn.BCELoss()

    def forward(
        self,
        sim_pred: torch.Tensor,   # (batch, 2)
        t_pred:   torch.Tensor,   # (batch, 1)
        y_true:   torch.Tensor,   # (batch,)  similarity label  0=similar, 1=dissimilar
        t_true:   torch.Tensor,   # (batch,)  treatment of anchor
    ) -> torch.Tensor:

        # Clip propensity scores away from 0/1 (numerical stability)
        t_pred_clipped = (t_pred.squeeze(1) + 0.01) / 1.02

        # Select the similarity output corresponding to the anchor's treatment group
        # t_true == 0 → use sim_pred[:,0]; t_true == 1 → use sim_pred[:,1]
        y_pred = (1.0 - t_true) * sim_pred[:, 0] + t_true * sim_pred[:, 1]

        # Contrastive loss
        square_pred   = y_pred ** 2
        margin_square = torch.clamp(self.margin - y_pred, min=0.0) ** 2
        contrastive   = torch.mean(
            (1.0 - y_true) * square_pred + y_true * margin_square
        )

        # Propensity BCE loss
        bce_loss = self.bce(t_pred_clipped, t_true)

        return self.alpha * contrastive + (1.0 - self.alpha) * bce_loss


# =============================================================================
# Data utilities
# =============================================================================

def create_siamese_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (anchor, comparison) training pairs following Algorithm 1 in the paper.

    For each instance in control/treatment group:
      - Positive pair: same outcome       → similarity label 0
      - Negative pair: different outcome  → similarity label 1

    Returns:
        x0, x1 : covariate arrays, shape (N_pairs, n_features)
        y       : similarity labels 0/1,  shape (N_pairs,)
        t       : treatment of anchor,    shape (N_pairs,)
    """
    x0_list, x1_list, y_list, t_list = [], [], [], []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_iterations):
        for group_id in [0, 1]:
            idx   = np.where(T == group_id)[0]
            X_grp = X[idx]
            Y_grp = Y[idx].ravel()

            idx_0 = np.where(Y_grp == 0)[0]
            idx_1 = np.where(Y_grp == 1)[0]

            if idx_0.size == 0 or idx_1.size == 0:
                continue  # skip degenerate splits

            # --- Anchor has Y=0 ---
            # Positive: pair with another Y=0 instance
            perm = idx_0.copy(); rng.shuffle(perm)
            x0_list.append(X_grp[idx_0]);  x1_list.append(X_grp[perm])
            y_list.append(np.zeros(idx_0.size)); t_list.append(np.full(idx_0.size, float(group_id)))

            # Negative: pair with a Y=1 instance
            comp = idx_1[rng.integers(0, idx_1.size, idx_0.size)]
            x0_list.append(X_grp[idx_0]);  x1_list.append(X_grp[comp])
            y_list.append(np.ones(idx_0.size));  t_list.append(np.full(idx_0.size, float(group_id)))

            # --- Anchor has Y=1 ---
            # Positive: pair with another Y=1 instance
            perm = idx_1.copy(); rng.shuffle(perm)
            x0_list.append(X_grp[idx_1]);  x1_list.append(X_grp[perm])
            y_list.append(np.zeros(idx_1.size)); t_list.append(np.full(idx_1.size, float(group_id)))

            # Negative: pair with a Y=0 instance
            comp = idx_0[rng.integers(0, idx_0.size, idx_1.size)]
            x0_list.append(X_grp[idx_1]);  x1_list.append(X_grp[comp])
            y_list.append(np.ones(idx_1.size));  t_list.append(np.full(idx_1.size, float(group_id)))

    x0 = np.concatenate(x0_list).astype(np.float32)
    x1 = np.concatenate(x1_list).astype(np.float32)
    y  = np.concatenate(y_list).astype(np.float32)
    t  = np.concatenate(t_list).astype(np.float32)

    return x0, x1, y, t


# =============================================================================
# Main CausalSiamese class
# =============================================================================

class CausalSiamese:
    """
    Causal Siamese network for treatment effect estimation.

    Estimates potential outcomes Ŷ(0) and Ŷ(1) for every test instance by:
      1. Learning a causally-aware similarity metric via contrastive training.
      2. At inference, finding the M most similar instances from the
         control / treatment group and averaging their outcomes
         (Algorithm 2 from the paper).

    Parameters
    ----------
    hidden_dim       : backbone embedding dimension
    distance         : 'L1' or 'L2'
    reg_l2           : L2 weight decay for the similarity head
    margin           : contrastive loss margin
    alpha            : loss weight  (alpha * contrastive + (1-alpha) * BCE)
    n_iterations     : number of pair-creation iterations (data augmentation)
    lr_adam          : Adam learning rate (warm-up phase)
    lr_sgd           : SGD learning rate (fine-tuning phase)
    momentum         : SGD momentum
    nesterov         : Nesterov momentum flag for SGD
    batch_size       : training batch size
    epochs_adam      : warm-up epochs with Adam
    epochs_sgd       : fine-tuning epochs with SGD
    patience         : early-stopping patience (SGD phase)
    device           : 'cpu', 'cuda', or None (auto-detect)
    """

    def __init__(
        self,
        hidden_dim: int = 50,
        distance: str = "L1",
        reg_l2: float = 1e-2,
        margin: float = 1.0,
        alpha: float = 0.5,
        n_iterations: int = 1,
        lr_adam: float = 5e-5,
        lr_sgd: float = 5e-5,
        momentum: float = 0.9,
        nesterov: bool = True,
        batch_size: int = 32,
        epochs_adam: int = 3,
        epochs_sgd: int = 50,
        patience: int = 15,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dim   = hidden_dim
        self.distance     = distance
        self.reg_l2       = reg_l2
        self.margin       = margin
        self.alpha        = alpha
        self.n_iterations = n_iterations
        self.lr_adam      = lr_adam
        self.lr_sgd       = lr_sgd
        self.momentum     = momentum
        self.nesterov     = nesterov
        self.batch_size   = batch_size
        self.epochs_adam  = epochs_adam
        self.epochs_sgd   = epochs_sgd
        self.patience     = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net_: Optional[CausalSiameseNet] = None

        # Store training data for inference (Algorithm 2)
        self._trainX: Optional[np.ndarray] = None
        self._trainT: Optional[np.ndarray] = None
        self._trainY: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_net(self, n_covariates: int) -> CausalSiameseNet:
        return CausalSiameseNet(
            n_covariates=n_covariates,
            hidden_dim=self.hidden_dim,
            reg_l2=self.reg_l2,
            distance=self.distance,
        ).to(self.device)

    def _make_loader(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x0),
            torch.from_numpy(x1),
            torch.from_numpy(y),
            torch.from_numpy(t),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: CausalSiameseLoss,
        epoch: int,
        n_epochs: int,
        phase_label: str,
    ) -> float:
        """Run one training epoch with a per-batch tqdm progress bar."""
        self.net_.train()
        total_loss = 0.0
        epoch_w    = len(str(n_epochs))

        with tqdm(
            loader,
            desc=f"  {phase_label} {epoch:>{epoch_w}}/{n_epochs}",
            unit="batch",
            bar_format="{l_bar}{bar:35}{r_bar}",
            leave=False,
        ) as pbar:
            for x0_b, x1_b, y_b, t_b in pbar:
                x0_b = x0_b.to(self.device)
                x1_b = x1_b.to(self.device)
                y_b  = y_b.to(self.device)
                t_b  = t_b.to(self.device)

                optimizer.zero_grad()
                sim_pred, t_pred, epsilons = self.net_(x0_b, x1_b)
                loss = criterion(sim_pred, t_pred, y_b, t_b)
                loss.backward()
                optimizer.step()

                batch_loss  = loss.item()
                total_loss += batch_loss * x0_b.size(0)
                pbar.set_postfix(loss=f"{batch_loss:.5f}")

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_loss(
        self,
        loader: DataLoader,
        criterion: CausalSiameseLoss,
    ) -> float:
        self.net_.eval()
        total_loss = 0.0
        for x0_b, x1_b, y_b, t_b in loader:
            x0_b = x0_b.to(self.device)
            x1_b = x1_b.to(self.device)
            y_b  = y_b.to(self.device)
            t_b  = t_b.to(self.device)

            sim_pred, t_pred, epsilons = self.net_(x0_b, x1_b)
            loss = criterion(sim_pred, t_pred, y_b, t_b)
            total_loss += loss.item() * x0_b.size(0)

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        valX: Optional[np.ndarray] = None,
        valT: Optional[np.ndarray] = None,
        valY: Optional[np.ndarray] = None,
    ) -> "CausalSiamese":
        """
        Fit the Causal Siamese model.

        Args:
            X  : Covariates,          shape (n_samples, n_features)
            t  : Treatment indicator, shape (n_samples,),  binary 0/1
            y  : Outcome,             shape (n_samples,),  binary 0/1
            valX/valT/valY : optional validation split for early stopping.
                             If not provided, training data is reused.

        Returns:
            self
        """
        start = time.time()

        # ------------------------------------------------------------------
        # Dataset summary
        # ------------------------------------------------------------------
        n_samples   = X.shape[0]
        n_features  = X.shape[1]
        t_flat      = t.ravel()
        y_flat      = y.ravel()
        n_control   = int((t_flat == 0).sum())
        n_treated   = int((t_flat == 1).sum())
        n_outcome_0 = int((y_flat == 0).sum())
        n_outcome_1 = int((y_flat == 1).sum())

        print("=" * 60)
        print("[INFO] CausalSiamese — fit()")
        print("=" * 60)
        print(f"[INFO] Device              : {self.device}")
        print(f"[INFO] Training samples    : {n_samples}")
        print(f"[INFO] Features            : {n_features}")
        print(f"[INFO] Control  (T=0)      : {n_control}  ({100*n_control/n_samples:.1f}%)")
        print(f"[INFO] Treated  (T=1)      : {n_treated}  ({100*n_treated/n_samples:.1f}%)")
        print(f"[INFO] Outcome  Y=0        : {n_outcome_0}  ({100*n_outcome_0/n_samples:.1f}%)")
        print(f"[INFO] Outcome  Y=1        : {n_outcome_1}  ({100*n_outcome_1/n_samples:.1f}%)")
        print("-" * 60)
        print(f"[INFO] Architecture        : hidden_dim={self.hidden_dim}, distance={self.distance}")
        print(f"[INFO] Loss                : alpha={self.alpha} (contrastive) / {1-self.alpha:.2f} (BCE), margin={self.margin}")
        print(f"[INFO] Batch size          : {self.batch_size}")
        print(f"[INFO] Adam warm-up        : {self.epochs_adam} epoch(s)  lr={self.lr_adam}")
        print(f"[INFO] SGD fine-tuning     : {self.epochs_sgd} epoch(s)  lr={self.lr_sgd}  momentum={self.momentum}  nesterov={self.nesterov}")
        print(f"[INFO] Early stopping      : patience={self.patience}")
        print(f"[INFO] Pair iterations     : {self.n_iterations}")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Store training data for inference (Algorithm 2)
        # ------------------------------------------------------------------
        self._trainX = X.astype(np.float32)
        self._trainT = t_flat.astype(np.float32)
        self._trainY = y_flat.astype(np.float32)

        # ------------------------------------------------------------------
        # Build siamese training pairs (Algorithm 1)
        # ------------------------------------------------------------------
        print("[INFO] Building siamese training pairs (Algorithm 1)...")
        pair_start = time.time()
        x0_tr, x1_tr, y_tr, t_tr = create_siamese_pairs(X, y, t, self.n_iterations)
        print(f"[INFO] Training pairs      : {x0_tr.shape[0]}  (built in {time.time()-pair_start:.2f}s)")

        if valX is not None:
            print("[INFO] Building validation pairs...")
            x0_val, x1_val, y_val, t_val = create_siamese_pairs(valX, valY, valT, 1)
            print(f"[INFO] Validation pairs    : {x0_val.shape[0]}")
        else:
            print("[INFO] No validation set provided — reusing training pairs for val loss")
            x0_val, x1_val, y_val, t_val = x0_tr, x1_tr, y_tr, t_tr

        train_loader = self._make_loader(x0_tr, x1_tr, y_tr, t_tr, shuffle=True)
        val_loader   = self._make_loader(x0_val, x1_val, y_val, t_val, shuffle=False)
        print(f"[INFO] Batches per epoch   : {len(train_loader)}  (batch_size={self.batch_size})")

        # ------------------------------------------------------------------
        # Build network
        # ------------------------------------------------------------------
        n_covariates = X.shape[1]
        self.net_    = self._build_net(n_covariates)
        n_params     = sum(p.numel() for p in self.net_.parameters() if p.requires_grad)
        print(f"[INFO] Network parameters  : {n_params:,}")
        criterion    = CausalSiameseLoss(margin=self.margin, alpha=self.alpha)

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        # ------------------------------------------------------------------
        # Phase 1: Adam warm-up
        # ------------------------------------------------------------------
        print("-" * 60)
        print(f"[PHASE 1] Adam warm-up — {self.epochs_adam} epoch(s), lr={self.lr_adam}")
        print("-" * 60)
        optimizer_adam = optim.Adam(self.net_.parameters(), lr=self.lr_adam)
        phase1_start   = time.time()

        for epoch in range(1, self.epochs_adam + 1):
            tr_loss  = self._train_one_epoch(
                train_loader, optimizer_adam, criterion,
                epoch=epoch, n_epochs=self.epochs_adam, phase_label="Adam",
            )
            val_loss = self._eval_loss(val_loader, criterion)
            print(
                f"  [Adam]  Epoch {epoch:>{len(str(self.epochs_adam))}}/{self.epochs_adam}"
                f" | train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}"
            )

        print(f"[INFO] Phase 1 done in {time.time()-phase1_start:.2f}s")

        # ------------------------------------------------------------------
        # Phase 2: SGD fine-tuning with early stopping
        # ------------------------------------------------------------------
        print("-" * 60)
        print(f"[PHASE 2] SGD fine-tuning — max {self.epochs_sgd} epoch(s), lr={self.lr_sgd}")
        print("-" * 60)
        optimizer_sgd = optim.SGD(
            self.net_.parameters(),
            lr=self.lr_sgd,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sgd, mode="min", factor=0.5, patience=5, 
        )
        phase2_start = time.time()
        epoch_w      = len(str(self.epochs_sgd))

        for epoch in range(1, self.epochs_sgd + 1):
            tr_loss  = self._train_one_epoch(
                train_loader, optimizer_sgd, criterion,
                epoch=epoch, n_epochs=self.epochs_sgd, phase_label="SGD ",
            )
            val_loss   = self._eval_loss(val_loader, criterion)
            current_lr = optimizer_sgd.param_groups[0]["lr"]
            scheduler.step(val_loss)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.net_.state_dict().items()}
                no_improve    = 0
                tag = " ✓ best"
            else:
                no_improve += 1
                tag = f" (no improve {no_improve}/{self.patience})"

            print(
                f"  [SGD]   Epoch {epoch:>{epoch_w}}/{self.epochs_sgd}"
                f" | train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}"
                f"  lr={current_lr:.2e}{tag}"
            )

            if no_improve >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} — no improvement for {self.patience} consecutive epochs.")
                break

        print(f"[INFO] Phase 2 done in {time.time()-phase2_start:.2f}s")

        # ------------------------------------------------------------------
        # Restore best weights
        # ------------------------------------------------------------------
        if best_state is not None:
            self.net_.load_state_dict(best_state)
            print(f"[INFO] Best weights restored  (val_loss={best_val_loss:.5f})")

        print("=" * 60)
        print(f"[INFO] Training completed in {time.time()-start:.2f}s")
        print("=" * 60)

        return self

    @torch.no_grad()
    def _get_similarities(
        self,
        anchor: np.ndarray,       # shape (n_features,)
        candidates: np.ndarray,   # shape (n_cand, n_features)
        group: int,               # 0 or 1
    ) -> np.ndarray:
        """
        Compute similarity scores between one anchor and all candidates
        for a given treatment group (head 0 or head 1).
        Returns array of shape (n_cand,).
        """
        self.net_.eval()
        n    = candidates.shape[0]
        x0   = np.tile(anchor, (n, 1)).astype(np.float32)
        x0_t = torch.from_numpy(x0).to(self.device)
        x1_t = torch.from_numpy(candidates.astype(np.float32)).to(self.device)

        sim_pred, _, _ = self.net_(x0_t, x1_t)
        return sim_pred[:, group].cpu().numpy()

    def _predict_single(self, instance: np.ndarray, M: int) -> Tuple[float, float]:
        """
        Algorithm 2: estimate Ŷ(0) and Ŷ(1) for a single test instance.

        Args:
            instance : covariate vector, shape (n_features,)
            M        : number of nearest neighbours used to average outcomes
        """
        # Control group
        ctrl_mask = self._trainT == 0
        X_C = self._trainX[ctrl_mask]
        Y_C = self._trainY[ctrl_mask]

        # Treatment group
        trt_mask = self._trainT == 1
        X_T = self._trainX[trt_mask]
        Y_T = self._trainY[trt_mask]

        # Similarity scores for control group (head 0)
        sim_c  = self._get_similarities(instance, X_C, group=0)
        top_c  = np.argsort(sim_c)[:M]
        y0_hat = Y_C[top_c].mean()

        # Similarity scores for treatment group (head 1)
        sim_t  = self._get_similarities(instance, X_T, group=1)
        top_t  = np.argsort(sim_t)[:M]
        y1_hat = Y_T[top_t].mean()

        return float(y0_hat), float(y1_hat)

    def predict(self, X: np.ndarray, M: int = 5) -> Dict[str, np.ndarray]:
        """
        Generate potential outcome predictions for all test instances.

        Args:
            X: Covariates, shape (n_samples, n_features)
            M: Number of nearest neighbours used to average outcomes (Algorithm 2)

        Returns:
            Dictionary with:
                - 'y0_hat': Predicted outcome under control  Ŷ(0), shape (n_samples,)
                - 'y1_hat': Predicted outcome under treatment Ŷ(1), shape (n_samples,)
        """
        if self.net_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        n_test    = X.shape[0]
        ctrl_size = int((self._trainT == 0).sum())
        trt_size  = int((self._trainT == 1).sum())

        print("=" * 60)
        print("[INFO] CausalSiamese — predict()")
        print("=" * 60)
        print(f"[INFO] Test samples        : {n_test}")
        print(f"[INFO] Nearest neighbours  : M={M}")
        print(f"[INFO] Control pool size   : {ctrl_size}")
        print(f"[INFO] Treatment pool size : {trt_size}")
        print(f"[INFO] Similarity metric   : {self.distance} distance")
        print("-" * 60)

        y0_list, y1_list = [], []
        pred_start = time.time()

        for instance in tqdm(
            X,
            total=n_test,
            desc="  Predicting",
            unit="sample",
            bar_format="{l_bar}{bar:40}{r_bar}",
        ):
            y0, y1 = self._predict_single(instance, M=M)
            y0_list.append(y0)
            y1_list.append(y1)

        elapsed = time.time() - pred_start
        print(f"[INFO] Inference completed in {elapsed:.2f}s  ({elapsed/n_test*1000:.1f} ms/sample)")
        print("=" * 60)

        return {
            "y0_hat": np.array(y0_list, dtype=np.float32),
            "y1_hat": np.array(y1_list, dtype=np.float32),
        }