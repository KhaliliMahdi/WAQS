from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh, sqrtm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    categories: Optional[List[str]] = None,
    pairs: Optional[List[Tuple[int, int]]] = None,
    probe_name: str = "probe",
) -> Dict:
    results = {"probe": probe_name}
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    results["auroc"] = float(roc_auc_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    results["tnr"] = float(tn / (tn + fp + 1e-8))
    results["tpr"] = float(tp / (tp + fn + 1e-8))
    results["fpr"] = float(fp / (fp + tn + 1e-8))
    results["fnr"] = float(fn / (fn + tp + 1e-8))

    if pairs:
        n_correct = sum(
            1 for safe_i, unsafe_i in pairs
            if y_pred[safe_i] == 0 and y_pred[unsafe_i] == 1
        )
        results["paired_accuracy"] = float(n_correct / len(pairs))
        results["n_pairs"] = len(pairs)

    if categories is not None:
        per_type = {}
        for cat in sorted(set(categories)):
            mask = np.array([c == cat for c in categories])
            if mask.sum() < 2:
                continue
            yt, yp, ypr = y_true[mask], y_pred[mask], y_proba[mask]
            cat_metrics = {"n": int(mask.sum()), "accuracy": float(accuracy_score(yt, yp))}
            if len(np.unique(yt)) > 1:
                cat_metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
                cat_metrics["auroc"] = float(roc_auc_score(yt, ypr))
            else:
                cat_metrics["balanced_accuracy"] = cat_metrics["accuracy"]
                cat_metrics["auroc"] = float("nan")
            per_type[cat] = cat_metrics
        results["per_type"] = per_type

    return results


def bures_wasserstein_distance(Sigma_pos: np.ndarray, Sigma_neg: np.ndarray) -> float:
    Sigma_pos = np.asarray(Sigma_pos, dtype=np.float64)
    Sigma_neg = np.asarray(Sigma_neg, dtype=np.float64)
    sqrt_pos = sqrtm(Sigma_pos).real
    inner = sqrt_pos @ Sigma_neg @ sqrt_pos
    inner = 0.5 * (inner + inner.T)
    sqrt_inner = sqrtm(inner).real
    val = np.trace(Sigma_pos) + np.trace(Sigma_neg) - 2 * np.trace(sqrt_inner)
    return float(np.sqrt(max(val, 0.0)))


def generalized_covariance_spectrum(Sigma_pos: np.ndarray, Sigma_neg: np.ndarray,
                                     shrinkage: float = 0.0) -> np.ndarray:
    Sigma_pos = np.asarray(Sigma_pos, dtype=np.float64)
    Sigma_neg = np.asarray(Sigma_neg, dtype=np.float64)
    if shrinkage > 0:
        d = Sigma_neg.shape[0]
        scale = np.trace(Sigma_neg) / d
        Sigma_neg = (1 - shrinkage) * Sigma_neg + shrinkage * scale * np.eye(d)
    eigvals = eigh(Sigma_pos, Sigma_neg, eigvals_only=True)
    return np.sort(eigvals)[::-1]


def compute_cluster_divergence(pos: np.ndarray, neg: np.ndarray, shrinkage: float = 0.1) -> Dict:
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    Sigma_pos = np.cov(pos.T)
    Sigma_neg = np.cov(neg.T)

    d_mean = float(np.linalg.norm(mu_pos - mu_neg))
    d_cov = bures_wasserstein_distance(Sigma_pos, Sigma_neg)
    spectrum = generalized_covariance_spectrum(Sigma_pos, Sigma_neg, shrinkage=shrinkage)

    return {
        "d_mean": d_mean,
        "d_cov_bures_wasserstein": d_cov,
        "generalized_cov_spectrum": spectrum,
        "spectrum_log_max_abs": float(np.max(np.abs(np.log(np.clip(spectrum, 1e-8, None))))),
    }
