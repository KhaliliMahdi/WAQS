from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class LinearProbe:

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs",
            class_weight="balanced", random_state=random_state,
        )
        self.fitted = False

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def fit(self, X, y) -> "LinearProbe":
        X = self._to_np(X)
        y = self._to_np(y).astype(int)
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        assert self.fitted, "Call fit() first"
        X = self.scaler.transform(self._to_np(X))
        return self.clf.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        assert self.fitted, "Call fit() first"
        X = self.scaler.transform(self._to_np(X))
        return self.clf.predict_proba(X)[:, 1]

    def get_weight_vector(self) -> torch.Tensor:
        return torch.tensor(self.clf.coef_[0], dtype=torch.float32)


class QuadraticProbe(nn.Module):

    def __init__(self, hidden_dim: int, rank: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.w = nn.Parameter(torch.zeros(hidden_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.U = nn.Parameter(torch.empty(rank, hidden_dim))
        self.V = nn.Parameter(torch.empty(rank, hidden_dim))

        nn.init.normal_(self.w, std=0.01)
        nn.init.normal_(self.U, std=0.01)
        nn.init.normal_(self.V, std=0.01)

        self.register_buffer("mean_", torch.zeros(hidden_dim))
        self.register_buffer("std_", torch.ones(hidden_dim))

    def _normalize(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean_) / (self.std_ + 1e-8)

    def fit_normalizer(self, X: torch.Tensor) -> None:
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self._normalize(h)
        linear = (h * self.w).sum(-1)
        Uh = h @ self.U.T
        Vh = h @ self.V.T
        quadratic = (Uh * Vh).sum(-1)
        return linear + quadratic + self.b

    @torch.no_grad()
    def compute_steering_vector(self, h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        Uh = self.U @ h
        Vh = self.V @ h
        quad_grad = self.U.T @ Vh + self.V.T @ Uh
        return alpha * (self.w + quad_grad)

    @torch.no_grad()
    def raw_space_params(self):
        """Probe parameters in raw (un-standardized) activation coordinates.

        The probe scores z = (x - mean_) / std_, so the steering map in activation space,
        T(x) = x + alpha * grad_x f(x), uses W_raw = D^-1 W D^-1 and w_raw = D^-1 w - 2 W_raw mean_,
        with D = diag(std_) and W the symmetric part of U^T V. Returns (U_raw, V_raw, w_raw)
        for inject_quadratic_probe(..., U=U_raw, V=V_raw, w_p=w_raw).
        """
        inv_std = 1.0 / (self.std_ + 1e-8)
        U_raw = self.U * inv_std
        V_raw = self.V * inv_std
        S_raw = U_raw.T @ V_raw + V_raw.T @ U_raw
        w_raw = self.w * inv_std - S_raw @ self.mean_
        return U_raw.clone(), V_raw.clone(), w_raw

    def absorbed_forward(self, h: torch.Tensor, steering_vec: torch.Tensor) -> torch.Tensor:
        h_s = self._normalize(h + steering_vec.unsqueeze(0))
        return (h_s * self.w).sum(-1) + self.b


class GDAQuadraticProbe(QuadraticProbe):

    @classmethod
    def fit_closed_form(
        cls,
        X: torch.Tensor,
        y: torch.Tensor,
        rank: int = 32,
        shrinkage: float = 0.1,
        shrink_target: str = "pooled",
        device: str = "cpu",
    ) -> "GDAQuadraticProbe":
        X = _as_tensor(X)
        y = _as_tensor(y).long()
        hidden_dim = X.shape[1]

        if shrink_target not in ("pooled", "identity"):
            raise ValueError(f"Unknown shrink_target: {shrink_target!r} (expected 'pooled' or 'identity')")

        probe = cls(hidden_dim, rank)

        mean_ = X.mean(0)
        std_ = X.std(0)
        Xn = (X - mean_) / (std_ + 1e-8)

        X0, X1 = Xn[y == 0], Xn[y == 1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        n0, n1 = len(X0), len(X1)
        pi0, pi1 = n0 / len(Xn), n1 / len(Xn)

        Sigma0_hat = torch.cov(X0.T)
        Sigma1_hat = torch.cov(X1.T)

        if shrink_target == "pooled":
            target0 = target1 = (n0 * Sigma0_hat + n1 * Sigma1_hat) / (n0 + n1)
        else:
            eye = torch.eye(hidden_dim, dtype=Sigma0_hat.dtype)
            target0 = (torch.trace(Sigma0_hat) / hidden_dim) * eye
            target1 = (torch.trace(Sigma1_hat) / hidden_dim) * eye

        Sigma0 = (1 - shrinkage) * Sigma0_hat + shrinkage * target0
        Sigma1 = (1 - shrinkage) * Sigma1_hat + shrinkage * target1
        Sigma0_inv = torch.linalg.inv(Sigma0)
        Sigma1_inv = torch.linalg.inv(Sigma1)

        W_full = 0.5 * (Sigma0_inv - Sigma1_inv)
        w_full = Sigma1_inv @ mu1 - Sigma0_inv @ mu0
        _, logdet0 = torch.linalg.slogdet(Sigma0)
        _, logdet1 = torch.linalg.slogdet(Sigma1)
        b_full = (
            0.5 * (mu0 @ Sigma0_inv @ mu0 - mu1 @ Sigma1_inv @ mu1)
            - 0.5 * (logdet1 - logdet0)
            + float(np.log(pi1 / pi0))
        )

        W_full = 0.5 * (W_full + W_full.T)
        eigvals, eigvecs = torch.linalg.eigh(W_full)
        order = torch.argsort(eigvals.abs(), descending=True)[:rank]
        top_vals = eigvals[order]
        top_vecs = eigvecs[:, order]

        r = top_vals.shape[0]
        sign = torch.sign(top_vals)
        sqrt_abs = torch.sqrt(top_vals.abs())
        U = (top_vecs * sqrt_abs).T
        V = (top_vecs * (sqrt_abs * sign)).T

        with torch.no_grad():
            probe.w.copy_(w_full)
            probe.b.copy_(torch.tensor([b_full]))
            probe.U[:r].copy_(U)
            probe.V[:r].copy_(V)
            if r < probe.rank:
                probe.U[r:].zero_()
                probe.V[r:].zero_()
            probe.mean_.copy_(mean_)
            probe.std_.copy_(std_)

        return probe.to(device)

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        X = _as_tensor(X).to(next(self.parameters()).device)
        return torch.sigmoid(self.forward(X)).cpu().numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class QuadraticProbeTrainer:

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cpu",
        w_reg: float = 0.0,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.w_reg = w_reg
        self.probe = QuadraticProbe(hidden_dim, rank).to(device)
        self.optimizer = torch.optim.AdamW(self.probe.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = True) -> dict:
        X_train = _as_tensor(X_train).to(self.device)
        y_train = _as_tensor(y_train).to(self.device)

        self.probe.fit_normalizer(X_train)

        n_pos = y_train.sum().item()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor(n_neg / (n_pos + 1e-8)).to(self.device)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True,
        )

        history = {"train_loss": [], "val_auc": []}
        best_val_auc, best_state, best_epoch = -1.0, None, -1

        for epoch in range(self.epochs):
            self.probe.train()
            total_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.probe(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                if self.w_reg > 0:
                    UUt = self.probe.U @ self.probe.U.T
                    VVt = self.probe.V @ self.probe.V.T
                    loss = loss + self.w_reg * (UUt * VVt).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            history["train_loss"].append(total_loss / len(loader))

            if X_val is not None:
                probs = self.predict_proba(X_val)
                auc = roc_auc_score(y_val, probs)
                acc = ((probs >= 0.5).astype(int) == y_val).mean()
                history["val_auc"].append(auc)
                history.setdefault("val_acc", []).append(acc)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs}  loss={history['train_loss'][-1]:.4f}"
                          f"  val_auc={auc:.4f}  val_acc={acc:.4f}")
                if auc > best_val_auc:
                    best_val_auc, best_epoch = auc, epoch + 1
                    best_state = copy.deepcopy(self.probe.state_dict())

        if best_state is not None:
            self.probe.load_state_dict(best_state)
            history["best_val_auc"] = best_val_auc
            history["best_epoch"] = best_epoch
            history["best_val_acc"] = history["val_acc"][best_epoch - 1]

        return history

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.probe.eval()
        X = _as_tensor(X).to(self.device)
        return torch.sigmoid(self.probe(X)).cpu().numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class DiagonalQuadraticProbe(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Parameter(torch.zeros(hidden_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.v = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.w, std=0.01)
        nn.init.normal_(self.v, std=0.01)
        self.register_buffer("mean_", torch.zeros(hidden_dim))
        self.register_buffer("std_", torch.ones(hidden_dim))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / (self.std_ + 1e-8)

    def fit_normalizer(self, X: torch.Tensor) -> None:
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self._normalize(x)
        linear = (xn * self.w).sum(-1)
        quadratic = (self.v * xn * xn).sum(-1)
        return linear + quadratic + self.b

    @torch.no_grad()
    def compute_steering_vector(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return alpha * (self.w + 2.0 * self.v * x)


class DiagonalQuadraticProbeTrainer:

    def __init__(
        self, hidden_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4,
        epochs: int = 50, batch_size: int = 64, device: str = "cpu", v_reg: float = 0.0,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.v_reg = v_reg
        self.probe = DiagonalQuadraticProbe(hidden_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.probe.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = True) -> dict:
        X_train = _as_tensor(X_train).to(self.device)
        y_train = _as_tensor(y_train).to(self.device)
        self.probe.fit_normalizer(X_train)

        n_pos = y_train.sum().item()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor(n_neg / (n_pos + 1e-8)).to(self.device)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True,
        )

        history = {"train_loss": [], "val_auc": []}
        best_val_auc, best_state, best_epoch = -1.0, None, -1

        for epoch in range(self.epochs):
            self.probe.train()
            total_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.probe(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                if self.v_reg > 0:
                    loss = loss + self.v_reg * (self.probe.v ** 2).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            history["train_loss"].append(total_loss / len(loader))

            if X_val is not None:
                probs = self.predict_proba(X_val)
                auc = roc_auc_score(y_val, probs)
                acc = ((probs >= 0.5).astype(int) == y_val).mean()
                history["val_auc"].append(auc)
                history.setdefault("val_acc", []).append(acc)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs}  loss={history['train_loss'][-1]:.4f}"
                          f"  val_auc={auc:.4f}  val_acc={acc:.4f}")
                if auc > best_val_auc:
                    best_val_auc, best_epoch = auc, epoch + 1
                    best_state = copy.deepcopy(self.probe.state_dict())

        if best_state is not None:
            self.probe.load_state_dict(best_state)
            history["best_val_auc"] = best_val_auc
            history["best_epoch"] = best_epoch
            history["best_val_acc"] = history["val_acc"][best_epoch - 1]

        return history

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.probe.eval()
        X = _as_tensor(X).to(self.device)
        return torch.sigmoid(self.probe(X)).cpu().numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def discriminative_subspace(X_train: np.ndarray, y_train: np.ndarray, dim: int, seed: int = 42):
    from sklearn.decomposition import PCA

    lin = LinearProbe(C=1.0, max_iter=2000)
    lin.fit(X_train, y_train)
    w = lin.get_weight_vector().numpy()
    w_raw = w / lin.scaler.scale_
    w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-8)

    proj = X_train @ w_raw
    residual = X_train - np.outer(proj, w_raw)
    pca = PCA(n_components=dim - 1, random_state=seed)
    pca.fit(residual)

    basis = np.concatenate([w_raw[None, :], pca.components_], axis=0)
    return basis, lin


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


_PROBE_CTOR_ARGS = {
    QuadraticProbe: ("hidden_dim", "rank"),
    GDAQuadraticProbe: ("hidden_dim", "rank"),
    DiagonalQuadraticProbe: ("hidden_dim",),
}


def save_probe(probe: nn.Module, path) -> None:
    cls = type(probe)
    if cls not in _PROBE_CTOR_ARGS:
        raise TypeError(
            f"save_probe doesn't know how to reconstruct {cls.__name__}; "
            f"supported types: {[c.__name__ for c in _PROBE_CTOR_ARGS]}"
        )
    ctor_kwargs = {name: getattr(probe, name) for name in _PROBE_CTOR_ARGS[cls]}
    torch.save(
        {
            "class_name": cls.__name__,
            "ctor_kwargs": ctor_kwargs,
            "state_dict": probe.state_dict(),
        },
        path,
    )


def load_probe(path, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes_by_name = {c.__name__: c for c in _PROBE_CTOR_ARGS}
    cls = classes_by_name[ckpt["class_name"]]
    probe = cls(**ckpt["ctor_kwargs"])
    probe.load_state_dict(ckpt["state_dict"])
    return probe.to(device).eval()
