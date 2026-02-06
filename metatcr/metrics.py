"""
metatcr.metrics
~~~~~~~~~~~~~~~

This module provides functions to compute various distance and dissimilarity
metrics between repertoire datasets, typically represented as frequency matrices.
"""

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel



# A small constant to prevent division by zero
EPS = 1e-9

__all__ = [
    'compute_kBET', 
    'compute_mmd', 
    'compute_jsd',
    'compute_ce',
    'compute_ilisi',
    'compute_cosine_distance'
]


def compute_kBET(X1: np.ndarray, X2: np.ndarray, k: int = 15) -> float:
    """
    Calculates the kBET rejection rate, a measure of batch mixing.
    A high value (near 1.0) indicates poor mixing, while a low value indicates good mixing.

    Args:
        X1 (np.ndarray): First batch of samples (n_samples1, n_features).
        X2 (np.ndarray): Second batch of samples (n_samples2, n_features).
        k (int): The number of nearest neighbors to consider.

    Returns:
        float: The kBET rejection rate, ranging from 0 to 1.
    """
    X = np.vstack([X1, X2])
    batch_labels = np.array([0] * len(X1) + [1] * len(X2))
    
    try:
        X_std = StandardScaler().fit_transform(X)
        n_components = min(10, X_std.shape[1], X_std.shape[0]-1)
        if n_components < 2: 
            return 1.0
        X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X_std)
    except ValueError:
        return 1.0

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)
    
    rejection_rates = []
    for i in range(len(X)):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_freq = np.sum(neighbor_batches == batch_labels[i])
        expected_freq = k * (np.sum(batch_labels == batch_labels[i]) / len(batch_labels))
        rejection_rates.append(1 if observed_freq > expected_freq else 0)
        
    return np.mean(rejection_rates)


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Computes the unbiased, squared Maximum Mean Discrepancy (MMD^2).
    The result is clamped to be non-negative.

    Args:
        X (np.ndarray): First set of samples (n_samples_X, n_features).
        Y (np.ndarray): Second set of samples (n_samples_Y, n_features).
        sigma (float, optional): The bandwidth of the RBF kernel. If None, it is
                                 estimated as the median of pairwise distances.

    Returns:
        float: The non-negative MMD^2 value.
    """
    if X.shape[0] < 2 or Y.shape[0] < 2: 
        return 0.0
    
    if sigma is None:
        pairwise_dists = np.sqrt(np.sum((X[:, None] - Y[None, :])**2, axis=-1))
        sigma = np.median(pairwise_dists)
        if sigma == 0: 
            sigma = 1.0
            
    gamma = 1.0 / (2 * (sigma**2) + EPS)
    
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    
    m, n = X.shape[0], Y.shape[0]
    
    term1 = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1) + EPS)
    term2 = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1) + EPS)
    term3 = -2 * K_XY.mean()
    
    mmd2_unbiased = term1 + term2 + term3
    
    return max(0, mmd2_unbiased)


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Jensen-Shannon Divergence for two probability distributions (1D arrays).
    Internal helper function.
    """
    p_clipped = np.maximum(p, 0)
    q_clipped = np.maximum(q, 0)
    
    # Normalize
    p_norm = p_clipped / (p_clipped.sum() + EPS)
    q_norm = q_clipped / (q_clipped.sum() + EPS)
    
    m = 0.5 * (p_norm + q_norm)
    
    # scipy.stats.entropy is robust to zeros in the input arrays.
    return 0.5 * (entropy(p_norm, m) + entropy(q_norm, m))


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Cross-Entropy between two probability distributions (1D arrays).
    Internal helper function. H(p, q) = -sum(p(x) * log(q(x)))
    """
    p_clipped = np.maximum(p, 0)
    q_clipped = np.maximum(q, 0)
    
    # Normalize
    p_norm = p_clipped / (p_clipped.sum() + EPS)
    q_norm = q_clipped / (q_clipped.sum() + EPS)
    
    # Compute CE. The clipping ensures `q_norm + EPS` is always positive.
    return -np.sum(p_norm * np.log(q_norm + EPS))


def compute_jsd(mtx1: np.ndarray, mtx2: np.ndarray) -> float:
    """
    Computes the average Jensen-Shannon Divergence between two matrices.
    This is done by calculating the JSD for all pairs of samples between
    mtx1 and mtx2 and averaging the results.

    Args:
        mtx1 (np.ndarray): First dataset (n_samples1, n_features).
        mtx2 (np.ndarray): Second dataset (n_samples2, n_features).

    Returns:
        float: The average JSD between the two matrices.
    """
    if mtx1.ndim == 1: mtx1 = mtx1.reshape(1, -1)
    if mtx2.ndim == 1: mtx2 = mtx2.reshape(1, -1)
    distance_matrix = cdist(mtx1, mtx2, metric=_jensen_shannon_divergence)
    return np.mean(distance_matrix)


# --- MODIFIED --- `compute_ce` now supports symmetric and asymmetric calculation.
def compute_ce(mtx1: np.ndarray, mtx2: np.ndarray, symmetric: bool = True) -> float:
    """
    Computes the average Cross-Entropy (CE) between two matrices.

    By default, computes the symmetric CE. Can be configured to compute the
    asymmetric CE.

    Args:
        mtx1 (np.ndarray): First dataset (n_samples1, n_features).
        mtx2 (np.ndarray): Second dataset (n_samples2, n_features).
        symmetric (bool): If True (default), computes the symmetric CE, defined as
                          the average of CE(mtx1, mtx2) and CE(mtx2, mtx1).
                          If False, computes the asymmetric CE(p, q), where p are
                          samples from `mtx1` (source) and q are from `mtx2`.

    Returns:
        float: The average Cross-Entropy between the two matrices.
    """
    if mtx1.ndim == 1: mtx1 = mtx1.reshape(1, -1)
    if mtx2.ndim == 1: mtx2 = mtx2.reshape(1, -1)

    if symmetric:
        # Calculate CE in both directions and average the results
        ce_12 = np.mean(cdist(mtx1, mtx2, metric=_cross_entropy))
        ce_21 = np.mean(cdist(mtx2, mtx1, metric=_cross_entropy))
        return (ce_12 + ce_21) / 2.0
    else:
        # Calculate asymmetric CE from mtx1 (p) to mtx2 (q)
        distance_matrix = cdist(mtx1, mtx2, metric=_cross_entropy)
        return np.mean(distance_matrix)


def compute_cosine_distance(mtx1: np.ndarray, mtx2: np.ndarray) -> float:
    """
    Computes the average Cosine Distance between two matrices.
    
    Cosine distance is defined as `1 - cosine_similarity`. A value of 0 means
    the vectors are identical in orientation, while 1 means they are orthogonal,
    and 2 means they are diametrically opposed. The average is taken over all
    pairwise comparisons between samples in mtx1 and mtx2.

    Args:
        mtx1 (np.ndarray): First dataset (n_samples1, n_features).
        mtx2 (np.ndarray): Second dataset (n_samples2, n_features).

    Returns:
        float: The average Cosine Distance between the two matrices.
    """
    if mtx1.ndim == 1: mtx1 = mtx1.reshape(1, -1)
    if mtx2.ndim == 1: mtx2 = mtx2.reshape(1, -1)
    
    # 'cosine' metric in cdist computes 1 - cosine_similarity
    distance_matrix = cdist(mtx1, mtx2, metric='cosine')
    return np.mean(distance_matrix)


# ------------------- LISI IMPLEMENTATION -------------------

# The following functions are adapted from harmonypy.
# Source: https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
#
# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software... (GPL License text as in original)

def _compute_inverse_simpson_index(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    tol: float = 1e-5
):
    """Core computation of the Simpson's index for each data point."""
    n_samples = distances.shape[1]
    unique_labels = np.unique(labels)
    simpson = np.zeros(n_samples)
    logU = np.log(perplexity)

    for i in range(n_samples):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf

        for t in range(50):
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P.fill(0)
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P /= P_sum
            
            Hdiff = H - logU
            if abs(Hdiff) < tol: break
            
            if Hdiff > 0:
                betamin = beta
                beta = (beta + betamax) / 2 if np.isfinite(betamax) else beta * 2
            else:
                betamax = beta
                beta = (beta + betamin) / 2 if np.isfinite(betamin) else beta / 2

        for label_category in unique_labels:
            q = labels[indices[:, i]] == label_category
            if np.any(q):
                P_sum_category = np.sum(P[q])
                simpson[i] += P_sum_category * P_sum_category
                
    return 1 / (simpson + EPS)

def compute_ilisi(
    mtx1: np.ndarray, 
    mtx2: np.ndarray, 
    perplexity: float = 30
) -> float:
    """
    Calculates the iLISI distance between two datasets.

    A score near 0 indicates good mixing (low domain gap), and a score near 1
    indicates poor mixing (high domain gap). The larger dataset is downsampled
    to match the smaller one to ensure size-insensitivity.

    Args:
        mtx1 (np.ndarray): First dataset (n_samples1, n_features).
        mtx2 (np.ndarray): Second dataset (n_samples2, n_features).
        perplexity (float): Related to the number of nearest neighbors. Default is 30.

    Returns:
        float: The iLISI distance score [0, 1].
    """
    n1, n2 = mtx1.shape[0], mtx2.shape[0]
    n_neighbors = int(perplexity * 3)

    if n1 <= n_neighbors or n2 <= n_neighbors:
        return 1.0

    if n1 > n2:
        indices = np.random.choice(n1, n2, replace=False)
        mtx1_sub = mtx1[indices]
        mtx2_sub = mtx2
    elif n2 > n1:
        indices = np.random.choice(n2, n1, replace=False)
        mtx1_sub = mtx1
        mtx2_sub = mtx2[indices]
    else:
        mtx1_sub, mtx2_sub = mtx1, mtx2
        
    X_combined = np.vstack((mtx1_sub, mtx2_sub))
    labels = np.array([0] * mtx1_sub.shape[0] + [1] * mtx2_sub.shape[0])
    
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='kd_tree').fit(X_combined)
    distances, indices = knn.kneighbors(X_combined)
    
    distances = distances[:, 1:].T
    indices = indices[:, 1:].T

    try:
        lisi_scores = _compute_inverse_simpson_index(
            distances=distances,
            indices=indices,
            labels=labels,
            perplexity=perplexity
        )
        median_lisi = np.nanmedian(lisi_scores)
        distance = 2.0 - median_lisi
        return np.clip(distance, 0.0, 1.0)

    except Exception as e:
        print(f"[Warning] Could not compute iLISI distance: {e}")
        return 1.0