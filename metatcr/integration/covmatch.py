"""
Covariance Matching integration for MetaTCR meta-vectors (NumPy arrays).

Adapts the covariance-alignment principle of CORAL (Sun, Feng & Saenko, 2016) to the
cross-study domain-shift setting. Two changes over the original: (1) a transferability
feature selection is applied first (features that are discriminative in the source and
distributionally stable between source and target), and (2) the alignment is run on the
selected features only. The underlying matrix transform (`_coral_transform`) is the
original CORAL whitening / re-colouring, kept unchanged.
"""

import logging
from typing import Tuple
import numpy as np

__all__ = ['covmatch_mtx', 'covmatch_transfer', 'select_transfer_features']

logger = logging.getLogger(__name__)


def covmatch_mtx(source_mtx: np.ndarray, target_mtx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the source covariance to the target on the given features (full or pre-selected).
    The source is transformed; the target is returned unchanged.
    """
    if source_mtx.shape[1] != target_mtx.shape[1]:
        raise ValueError("Source and target matrices must have the same number of features (columns).")
    return _coral_transform(source_mtx, target_mtx), target_mtx


def covmatch_multi_mtx(blocks):
    """Multi-batch Covariance Matching: align every block to an equal-cohort pooled target.

    The multi-dataset counterpart of ``covmatch_mtx``. Each cohort's mean and Ledoit-Wolf-shrunk
    covariance are estimated; the target is the *equal-weighted* average of the per-cohort means
    and covariances (so a large cohort cannot dominate the target). Every block is then whitened by
    its own covariance and re-coloured to the pooled target.

    Args:
        blocks: list of arrays, each (n_i, n_features); all must share n_features.

    Returns:
        list of corrected arrays, one per input block (row order preserved).
    """
    from sklearn.covariance import LedoitWolf
    if len(blocks) < 2:
        raise ValueError("covmatch_multi_mtx needs at least two blocks.")
    if len({b.shape[1] for b in blocks}) != 1:
        raise ValueError("All blocks must have the same number of features (columns).")
    means, covariances = [], []
    for block in blocks:
        fit = LedoitWolf().fit(block)
        means.append(fit.location_)
        covariances.append(fit.covariance_)
    target_mean = np.mean(means, axis=0)
    target_sqrt = _symmetric_sqrt(np.mean(covariances, axis=0))
    corrected = []
    for block, mean, covariance in zip(blocks, means, covariances):
        transform = _symmetric_sqrt(covariance, inverse=True) @ target_sqrt
        corrected.append(np.nan_to_num((block - mean) @ transform + target_mean))
    return corrected


def _symmetric_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Symmetric (inverse) matrix square root via eigendecomposition."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    power = -0.5 if inverse else 0.5
    return vectors @ np.diag(values ** power) @ vectors.T


def _coral_transform(F_S: np.ndarray, F_T: np.ndarray) -> np.ndarray:
    """Core CORAL transform: whiten the source covariance and re-colour it with the target's."""
    mu_S = np.mean(F_S, axis=0)
    mu_T = np.mean(F_T, axis=0)
    F_S_centered = F_S - mu_S

    reg = 1e-8
    C_S = np.cov(F_S_centered, rowvar=False) + reg * np.identity(F_S.shape[1])
    C_T = np.cov(F_T, rowvar=False) + reg * np.identity(F_T.shape[1])

    eigvals_S, eigvecs_S = np.linalg.eigh(C_S)
    eigvals_T, eigvecs_T = np.linalg.eigh(C_T)
    eigvals_S = np.maximum(eigvals_S, 0)
    eigvals_T = np.maximum(eigvals_T, 0)

    C_S_inv_sqrt = eigvecs_S @ np.diag(1.0 / np.sqrt(eigvals_S + 1e-8)) @ eigvecs_S.T
    C_T_sqrt = eigvecs_T @ np.diag(np.sqrt(eigvals_T)) @ eigvecs_T.T

    F_S_transformed = (F_S_centered @ C_S_inv_sqrt) @ C_T_sqrt + mu_T
    return F_S_transformed.astype(F_S.dtype)


def select_transfer_features(source_mtx, target_mtx, source_labels, n_features=None, frac=0.25, transfer_alpha=2.0):
    """
    Rank features by source discriminativeness (ANOVA-F on source labels only) down-weighted
    by their source/target distribution shift (per-feature |mu_s - mu_t| / pooled_sd, using
    target features but not target labels), and keep the top ones.

    n_features: keep a fixed number of features. If None, keep a proportion `frac` of the
    columns (default). Returns the kept column indices (into the full meta-vector).
    """
    from sklearn.feature_selection import f_classif
    score = np.nan_to_num(f_classif(source_mtx, source_labels)[0])
    if transfer_alpha:
        sd = np.sqrt((source_mtx.var(0) + target_mtx.var(0)) / 2) + 1e-8
        score = score / (1 + transfer_alpha * np.abs(source_mtx.mean(0) - target_mtx.mean(0)) / sd)
    k = int(n_features) if n_features is not None else max(1, round(frac * source_mtx.shape[1]))
    return np.argsort(score)[::-1][:min(k, source_mtx.shape[1])]


def covmatch_transfer(source_mtx, target_mtx, source_labels, n_features=None, frac=0.25, transfer_alpha=2.0):
    """
    Domain-shift pipeline: select the transferable features, then covariance-match on them.

    Returns (corrected_source, corrected_target, kept_idx), all restricted to the SELECTED
    features -- classify on these. `kept_idx` records which columns of the original meta-vector
    were used; downstream / UMAP analyses keep the full meta-vector rather than this subset.
    """
    idx = select_transfer_features(source_mtx, target_mtx, source_labels, n_features, frac, transfer_alpha)
    corrected_source, corrected_target = covmatch_mtx(source_mtx[:, idx], target_mtx[:, idx])
    return corrected_source, corrected_target, idx
