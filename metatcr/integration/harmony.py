# metatcr/integration/harmony.py

"""
source: https://github.com/slowkow/harmonypy
harmonypy is a port of the harmony R package by Ilya Korsunsky.

Harmony integration algorithm adapted to work directly with NumPy arrays.

This module provides two high-level functions:
1. `harmony_mtx`: Corrects batch effects between a source and target matrix.
2. `harmonize_numpy`: A general-purpose function to correct batch effects
   from a single matrix and a list of batch labels.
"""

import logging
from functools import partial
# Import Union and Tuple for backward compatibility with Python < 3.10
from typing import Union, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Define what functions are publicly exposed when importing with '*'
__all__ = ['harmony_mtx', 'harmonize_numpy']

# Setup logger for the module.
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.WARNING) ## logging level can be adjusted as needed.

# Add a handler only if one doesn't exist to prevent duplicate messages.
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def harmony_mtx(source_mtx: np.ndarray, target_mtx: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrects batch effects between source and target matrices using Harmony.

    This function treats the source and target matrices as two distinct
    batches. It runs the Harmony algorithm, which adjusts all data points,
    and then returns the corrected versions of the source and target matrices.

    Args:
        source_mtx: The source data matrix (n_samples, n_features).
        target_mtx: The target data matrix (m_samples, m_features).
        **kwargs: Additional keyword arguments to pass directly to the
                  core Harmony algorithm (e.g., theta, lamb, sigma).

    Returns:
        A tuple containing:
        - corrected_source (np.ndarray): The corrected source matrix.
        - corrected_target (np.ndarray): The corrected target matrix.
    """
    # This initial info message is outside the 'harmonypy' logger, so it will still show.
    # To disable it, you would need to control the root logger in your main script.
    # For now, we leave it as it provides context.
    logging.info(
        f"Applying Harmony: Integrating source matrix ({source_mtx.shape}) "
        f"and target matrix ({target_mtx.shape})."
    )

    if source_mtx.shape[1] != target_mtx.shape[1]:
        raise ValueError("Source and target matrices must have the same number of features (columns).")

    # --- Internal Logic ---
    # 1. Store original shape of the source matrix to split the data later
    n_source = source_mtx.shape[0]

    # 2. Combine matrices into a single dataset
    combined_data = np.vstack([source_mtx, target_mtx])

    # 3. Create batch labels internally
    batch_labels = ['source'] * n_source + ['target'] * target_mtx.shape[0]

    # 4. Run Harmony using the existing `harmonize_numpy` function as a backend
    corrected_combined = harmonize_numpy(combined_data, batch_labels, **kwargs)

    # 5. Split the corrected matrix back into source and target components
    corrected_source = corrected_combined[:n_source, :]
    corrected_target = corrected_combined[n_source:, :]

    logging.info("Harmony integration complete.")

    return corrected_source, corrected_target


def harmonize_numpy(
    data_matrix: np.ndarray,
    # Use Union for backward compatibility
    batch_labels: Union[list, np.ndarray],
    **kwargs
) -> np.ndarray:
    """
    Integrates a NumPy data matrix using the Harmony algorithm to correct batch effects.

    This function is a convenient wrapper for the core Harmony implementation,
    designed to work directly with NumPy arrays and avoid a dependency on
    AnnData or scanpy objects.

    Args:
        data_matrix: The data matrix to be integrated, with shape
                     (n_samples, n_features). For example, PCA embeddings.
        batch_labels: A list or array of length n_samples containing the batch
                      identifier for each sample. E.g., ['batch1', 'batch2', ...].
        **kwargs: Additional keyword arguments to be passed to the core
                  `_run_harmony` function, such as theta, lamb, sigma, nclust,
                  max_iter_harmony, etc.

    Returns:
        The Harmony-corrected data matrix, with shape (n_samples, n_features).

    Example:
        >>> from metatcr.integration import harmony
        >>> # 100 samples, 30 principal components
        >>> pca_data = np.random.rand(100, 30)
        >>> batches = ['A'] * 50 + ['B'] * 50
        >>> corrected_pca = harmony.harmonize_numpy(pca_data, batches, theta=2)
    """
    if data_matrix.shape[0] != len(batch_labels):
        raise ValueError(
            "The number of rows in data_matrix (n_samples) must match "
            "the length of batch_labels."
        )

    # The core Harmony function requires a pandas DataFrame with batch information.
    meta_data = pd.DataFrame({'batch': batch_labels})

    # Call the internal core algorithm.
    # Note: _run_harmony handles the matrix transposition internally.
    harmony_out = _run_harmony(
        data_mat=data_matrix,
        meta_data=meta_data,
        vars_use=['batch'],
        **kwargs
    )

    # The core algorithm returns Z_corr with shape (n_features, n_samples).
    # Transpose it back to the standard (n_samples, n_features) format.
    adjusted_matrix = harmony_out.Z_corr.T
    
    return adjusted_matrix


def _run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta = None,
    lamb = None,
    sigma = 0.1, 
    nclust = None,
    tau = 0,
    block_size = 0.05, 
    max_iter_harmony = 10,
    max_iter_kmeans = 20,
    epsilon_cluster = 1e-5,
    epsilon_harmony = 1e-4, 
    plot_convergence = False,
    verbose = True,
    reference_values = None,
    cluster_prior = None,
    random_state = 0
):
    """Internal Harmony runner function."""
    N = meta_data.shape[0]
    
    # The algorithm expects data as (features, cells). Transpose if necessary.
    if data_mat.shape[0] == N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells" 

    if nclust is None:
        nclust = min(round(N / 30.0), 100)

    if isinstance(sigma, float) and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat(theta, phi_n)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat(lamb, phi_n)

    assert len(lamb) == np.sum(phi_n), \
        "each batch variable must have a lambda"

    N_b = phi.sum(axis = 1)
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))
    phi_moe = np.vstack((np.repeat(1, N), phi))
    np.random.seed(random_state)

    # --- MODIFICATION ---
    # Pass the verbose flag down to the _Harmony class constructor
    ho = _Harmony(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state
    )
    return ho

class _Harmony(object):
    # This class holds the state and methods for the Harmony algorithm execution.
    # The implementation is based on the original harmonypy library.
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / np.linalg.norm(self.Z_orig, ord=2, axis=0)

        self.Phi             = Phi
        self.Phi_moe         = Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = Pr_b
        self.B               = self.Phi.shape[0] # number of batch variables
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb            = lamb
        self.sigma           = sigma
        self.sigma_prior     = sigma
        self.block_size      = block_size
        self.K               = K                # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose         = verbose
        self.theta           = theta

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        self.allocate_buffers()
        cluster_fn = partial(_Harmony._cluster_kmeans, random_state=random_state, verbose=self.verbose)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat    = np.zeros((self.K, self.N))
        self.O           = np.zeros((self.K, self.B))
        self.E           = np.zeros((self.K, self.B))
        self.W           = np.zeros((self.B + 1, self.d))
        self.Phi_Rk      = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state, verbose=True):
        if verbose:
            logger.info("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=random_state)
        model.fit(data)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        if verbose:
            logger.info("sklearn.KMeans initialization complete.")
        return km_centroids

    def init_cluster(self, cluster_fn):
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        _entropy = np.sum(_safe_entropy(self.R) * self.sigma[:,np.newaxis])
        x = (self.R * self.sigma[:,np.newaxis])
        y = np.tile(self.theta[:,np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose: logger.info(f"Iteration {i} of {iter_harmony}")
            self.cluster()
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = _moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )
            converged = self.check_convergence(1)
            if converged:
                if verbose: logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break
        if verbose and not converged: logger.info("Stopped before convergence")

    def cluster(self):
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            self.update_R()
            self.compute_objective()
            if i > self.window_size:
                if self.check_convergence(0): break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        self._scale_dist = -self.dist_mat / self.sigma[:,None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = int(np.ceil(1 / self.block_size))
        for b in np.array_split(update_order, n_blocks):
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            self.R[:,b] = self._scale_dist[:,b] * np.dot(
                np.power((self.E + 1) / (self.O + 1), self.theta), self.Phi[:,b]
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)

    def check_convergence(self, i_type):
        obj = self.objective_kmeans if i_type == 0 else self.objective_harmony
        if len(obj) < self.window_size + 1: return False
        
        if i_type == 0:
            obj_old = np.mean(obj[-(self.window_size + 1):-1])
            obj_new = np.mean(obj[-self.window_size:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        else:
            obj_old = obj[-2]
            obj_new = obj[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony


def _safe_entropy(x: np.ndarray) -> np.ndarray:
    """Computes entropy, handling log(0) cases."""
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

def _moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    """Mixture of Experts-style ridge regression to correct for covariates."""
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.linalg.inv(x), np.dot(Phi_Rk, Z_orig.T))
        W[0,:] = 0 # Do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk