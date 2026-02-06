# metatcr/integration/coral.py

"""
CORAL (Correlation Alignment) integration algorithm for NumPy arrays.

This module provides a high-level function `coral_mtx` to correct a source
matrix to align with the statistical properties of a target matrix.

Reference:
Sun, B., Feng, J., & Saenko, K. (2016). Return of Frustratingly Easy Domain
Adaptation.
"""

import logging
from typing import Tuple
import numpy as np

# Define what functions are publicly exposed when importing with '*'
__all__ = ['coral_mtx']

# Set up a logger for this module
logger = logging.getLogger(__name__)


def coral_mtx(source_mtx: np.ndarray, target_mtx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms a source matrix to align with a target matrix using CORAL.

    This function changes the source matrix so its covariance matches the
    target matrix. The target matrix is returned unchanged.

    Args:
        source_mtx: The source data matrix (n_samples, n_features).
        target_mtx: The target data matrix (m_samples, m_features).

    Returns:
        A tuple containing:
        - transformed_source (np.ndarray): The modified source matrix.
        - target_mtx (np.ndarray): The original, unchanged target matrix.
    """
    logger.info(
        f"Applying CORAL: Transforming source matrix of shape {source_mtx.shape} "
        f"to align with target matrix of shape {target_mtx.shape}."
    )
    
    if source_mtx.shape[1] != target_mtx.shape[1]:
        raise ValueError("Source and target matrices must have the same number of features (columns).")

    transformed_source = _coral_transform(source_mtx, target_mtx)
    
    logger.info("CORAL transformation complete.")
    
    return transformed_source, target_mtx


def _coral_transform(F_S: np.ndarray, F_T: np.ndarray) -> np.ndarray:
    """
    Core CORAL algorithm implementation.
    """
    # Calculate means
    mu_S = np.mean(F_S, axis=0)
    mu_T = np.mean(F_T, axis=0)

    # Center the source data
    F_S_centered = F_S - mu_S

    # Calculate covariance matrices with regularization
    reg = 1e-8
    C_S = np.cov(F_S_centered, rowvar=False) + reg * np.identity(F_S.shape[1])
    C_T = np.cov(F_T, rowvar=False) + reg * np.identity(F_T.shape[1])

    # Eigendecomposition for matrix square roots
    eigvals_S, eigvecs_S = np.linalg.eigh(C_S)
    eigvals_T, eigvecs_T = np.linalg.eigh(C_T)

    # Ensure eigenvalues are non-negative
    eigvals_S = np.maximum(eigvals_S, 0)
    eigvals_T = np.maximum(eigvals_T, 0)

    # Calculate inverse square root of C_S and square root of C_T
    # Add a small epsilon to sqrt for numerical stability
    C_S_inv_sqrt = eigvecs_S @ np.diag(1.0 / np.sqrt(eigvals_S + 1e-8)) @ eigvecs_S.T
    C_T_sqrt = eigvecs_T @ np.diag(np.sqrt(eigvals_T)) @ eigvecs_T.T

    # Whiten the source data and re-color it with the target covariance
    F_S_whitened = F_S_centered @ C_S_inv_sqrt
    F_S_transformed = F_S_whitened @ C_T_sqrt + mu_T

    return F_S_transformed.astype(F_S.dtype)