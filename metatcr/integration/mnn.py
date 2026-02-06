# metatcr/integration/mnn.py

"""
MNN (Mutual Nearest Neighbors) correction algorithm for NumPy arrays.

This module provides a high-level function `mnn_mtx` to correct batch
effects between a source and a target matrix.

The core algorithm is adapted from the `mnnpy` library, developed by
Chris Cain et al. The original repository can be found at:
- https://github.com/chriscainx/mnnpy

This version has been refactored to remove the dependency on AnnData
and provide a simplified interface consistent with other integration tools,
while preserving the original core computational logic.

A new `hvg` parameter has been added to allow integration to be performed
on a subset of highly variable genes, which can improve both speed and
accuracy.
"""

import logging
import math
import os
from os import cpu_count
from multiprocessing import Pool
from typing import Tuple, Optional
import numpy as np
import pandas as pd  # Added dependency for HVG calculation
from pandas import DataFrame
from scipy.spatial import cKDTree
from scipy.linalg.interpolative import svd as rsvd
from scipy.sparse import issparse
from numba import jit, float32, int32
from metatcr.integration.mnnpy._utils import _adjust_shift_variance

# Define what functions are publicly exposed
__all__ = ['mnn_mtx']

# Set up a logger for this module
logger = logging.getLogger(__name__)


def _find_highly_variable_genes(
    data_matrix: np.ndarray, 
    n_top_genes: int, 
    n_bins: int = 20
) -> np.ndarray:
    """
    Identifies highly variable genes based on normalized dispersion.

    This method is adapted from standard single-cell analysis workflows (e.g.,
    in Seurat, Scanpy). It corrects for the relationship between mean expression
    and variance to find genes that are more variable than expected.

    Args:
        data_matrix (np.ndarray): A combined data matrix (cells, genes),
            preferably log-normalized (e.g., log1p).
        n_top_genes (int): The number of top variable genes to select.
        n_bins (int): The number of bins to group genes by mean expression.

    Returns:
        np.ndarray: An array of integer indices for the selected highly
            variable genes.
    """
    # Step 1: Calculate mean and variance for each gene
    mean_expr = np.mean(data_matrix, axis=0)
    variance_expr = np.var(data_matrix, axis=0)

    # Create a DataFrame to manage calculations
    gene_stats = pd.DataFrame({
        'mean': mean_expr,
        'variance': variance_expr
    })

    # Filter out genes with zero variance as they are not variable
    gene_stats = gene_stats[gene_stats['variance'] > 0]

    # Step 2: Bin genes by their mean expression
    try:
        gene_stats['mean_bin'] = pd.cut(gene_stats['mean'], bins=n_bins, labels=False, include_lowest=True)
    except ValueError:
        logger.warning(
            "Could not create the requested number of bins for HVG selection. "
            "Falling back to variance-based selection."
        )
        sorted_indices = np.argsort(variance_expr)[::-1]
        return sorted_indices[:n_top_genes]

    # Step 3: Calculate the mean and standard deviation of variance within each bin
    bin_stats = gene_stats.groupby('mean_bin')['variance']
    bin_mean_var = bin_stats.transform('mean')
    bin_std_var = bin_stats.transform('std')

    # Step 4: Calculate normalized dispersion (z-score of variance) for each gene
    gene_stats['dispersion'] = (gene_stats['variance'] - bin_mean_var) / (bin_std_var + 1e-6)
    gene_stats['dispersion'].fillna(0, inplace=True)

    # Step 5: Rank genes by their dispersion score and select the top N
    gene_stats.sort_values(by='dispersion', ascending=False, inplace=True)
    hvg_indices = gene_stats.index[:n_top_genes].to_numpy()
    
    return hvg_indices


def mnn_mtx(
    source_mtx: np.ndarray, 
    target_mtx: np.ndarray, 
    hvg: Optional[int] = None, 
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrects batch effects between source and target matrices using MNN.

    This is a wrapper around a NumPy-only version of the `mnnpy` algorithm.
    It takes a source and a target matrix, performs MNN correction, and
    returns the two matrices in the corrected space. Note that the correction
    is asymmetric: the `target_mtx` is corrected to align with the `source_mtx`,
    which serves as the reference.

    If the `hvg` parameter is specified, integration is performed only on the
    subset of highly variable genes. The corrected values for these genes are
    then placed back into the full-dimensional matrix.

    Args:
        source_mtx (np.ndarray): The reference data matrix (n_samples, n_features).
        target_mtx (np.ndarray): The target data matrix to be corrected (m_samples, n_features).
        hvg (Optional[int], optional): The number of highly variable genes to use for
            integration. If None, all genes are used. Defaults to None.
        **kwargs: Additional keyword arguments to pass directly to the core
                  MNN algorithm (e.g., k, sigma, svd_dim, var_adj).

    Returns:
        A tuple containing:
        - corrected_source (np.ndarray): The source (reference) matrix in the integrated space.
        - corrected_target (np.ndarray): The corrected target matrix.
    """
    logger.info(
        f"Applying MNN: Integrating target matrix ({target_mtx.shape}) "
        f"onto source (reference) matrix ({source_mtx.shape})."
    )

    if source_mtx.shape[1] != target_mtx.shape[1]:
        raise ValueError("Source and target matrices must have the same number of features (columns).")

    hvg_indices = None
    source_to_integrate = source_mtx
    target_to_integrate = target_mtx

    # --- HVG Selection and Subsetting Logic ---
    if hvg is not None and hvg > 0:
        if hvg > source_mtx.shape[1]:
            raise ValueError(f"hvg ({hvg}) cannot be greater than the number of features ({source_mtx.shape[1]}).")
        
        logger.info(f"Selecting top {hvg} highly variable genes for integration using normalized dispersion.")
        
        combined_data = np.vstack((source_mtx, target_mtx))
        hvg_indices = _find_highly_variable_genes(combined_data, hvg)
        hvg_indices = np.sort(hvg_indices)

        source_to_integrate = source_mtx[:, hvg_indices]
        target_to_integrate = target_mtx[:, hvg_indices]
        
        logger.info(
            f"Subsetting matrices to shape {source_to_integrate.shape} and "
            f"{target_to_integrate.shape} for integration."
        )
    # --- End of HVG Logic ---

    # The mnn_correct algorithm expects a list of batches.
    datas = [source_to_integrate, target_to_integrate]
    n_features_to_integrate = source_to_integrate.shape[1]
    var_index = [f'var_{i}' for i in range(n_features_to_integrate)]

    # The core function returns a single concatenated matrix
    corrected_combined, _, _ = _mnn_correct_numpy(
        *datas,
        var_index=var_index,
        do_concatenate=True,
        **kwargs
    )

    # --- Reconstruction Logic ---
    if hvg_indices is not None:
        logger.info("Reconstructing full-size matrices from HVG-corrected data.")
        
        # Initialize final matrices as copies of the originals
        # The MNN reference (source) is also modified by normalization, so we need to reconstruct it too.
        corrected_source = source_mtx.copy()
        corrected_target = target_mtx.copy()

        n_source = source_mtx.shape[0]
        corrected_source_hvg = corrected_combined[:n_source, :]
        corrected_target_hvg = corrected_combined[n_source:, :]

        # "Paste" the corrected HVG data back into the full matrices
        corrected_source[:, hvg_indices] = corrected_source_hvg
        corrected_target[:, hvg_indices] = corrected_target_hvg

    else: # No HVG was used, so the corrected datasets are already full-size
        n_source = source_mtx.shape[0]
        corrected_source = corrected_combined[:n_source, :]
        corrected_target = corrected_combined[n_source:, :]
    # --- End of Reconstruction Logic ---

    logger.info("MNN integration complete.")

    return corrected_source, corrected_target, hvg_indices


def _mnn_correct_numpy(*datas, var_index=None, var_subset=None, k=20, sigma=1.,
                       cos_norm_in=True, cos_norm_out=True, svd_dim=None, var_adj=True,
                       compute_angle=False, mnn_order=None, svd_mode='rsvd',
                       do_concatenate=True, n_jobs=None, **kwargs):
    if len(datas) < 2:
        return datas, None, None
    n_batch = len(datas)
    if mnn_order is not None:
        if sorted(mnn_order) != list(range(n_batch)):
            raise ValueError('The argument mnn_order should contain values in 1:' + 'n_batch' + '.')
    
    if n_jobs is None:
        n_jobs = cpu_count()
    n_cols = datas[0].shape[1]
    if var_index is None or len(var_index) != n_cols:
        raise ValueError('The number of vars is not equal to the length of var_index.')
    for i in range(1, n_batch):
        if datas[i].shape[1] != n_cols:
            raise ValueError('The input matrices have inconsistent number of columns.')

    logger.info('Performing cosine normalization...')
    in_batches, out_batches, var_subset_idx, same_set = _transform_input_data(datas, cos_norm_in,
                                                                         cos_norm_out, var_index,
                                                                         var_subset, n_jobs)
    if mnn_order is None:
        mnn_order = list(range(n_batch))
    ref = mnn_order[0]
    ref_batch_in = in_batches[ref]
    if not same_set:
        ref_batch_out = out_batches[ref]
    res_container = [out_batches[ref]]
    mnn_container = [0]
    original_batch = [ref] * ref_batch_in.shape[0]
    logger.info('Starting MNN correct iteration. Reference batch: ' + str(ref))

    for step in range(1, n_batch):
        target = mnn_order[step]
        logger.info('Step ' + str(step) + ' of ' + str(n_batch - 1) + ': processing batch ' + str(target))
        new_batch_in = in_batches[target]
        if not same_set:
            new_batch_out = out_batches[target]
        
        logger.info('  Looking for MNNs...')
        mnn_ref, mnn_new = _find_mutual_nn(data1=ref_batch_in, data2=new_batch_in, k1=k, k2=k,
                                          n_jobs=n_jobs)
        
        logger.info('  Computing correction vectors...')
        correction_in = _compute_correction(ref_batch_in, new_batch_in, np.array(mnn_ref, dtype=np.int32), 
                                           np.array(mnn_new, dtype=np.int32), new_batch_in, sigma)
        if not same_set:
            correction_out = _compute_correction(ref_batch_out, new_batch_out, np.array(mnn_ref, dtype=np.int32), 
                                                np.array(mnn_new, dtype=np.int32), new_batch_in, sigma)
        
        if svd_dim is not None and svd_dim != 0:
            logger.info('  Removing biological components...')
            mnn_ref_u = np.unique(mnn_ref)
            mnn_new_u = np.unique(mnn_new)
            in_span_ref = _get_bio_span(ref_batch_in[mnn_ref_u, :], ndim=svd_dim, svd_mode=svd_mode, **kwargs)
            in_span_new = _get_bio_span(new_batch_in[mnn_new_u, :], ndim=svd_dim, svd_mode=svd_mode, **kwargs)
            correction_in = _subtract_bio(in_span_ref, in_span_new, correction=correction_in)
            if not same_set:
                out_span_ref = _get_bio_span(ref_batch_out[mnn_ref_u, :], ndim=svd_dim, svd_mode=svd_mode, var_subset=var_subset_idx, **kwargs)
                out_span_new = _get_bio_span(new_batch_out[mnn_new_u, :], ndim=svd_dim, svd_mode=svd_mode, var_subset=var_subset_idx, **kwargs)
                correction_out = _subtract_bio(out_span_ref, out_span_new, correction=correction_out, var_subset=var_subset_idx)
        
        if var_adj:
            logger.info('  Adjusting variance...')
            correction_in = _adjust_shift_variance(ref_batch_in, new_batch_in, correction_in, sigma, n_jobs, var_subset=None)
            if not same_set:
                correction_out = _adjust_shift_variance(ref_batch_out, new_batch_out, correction_out, sigma, n_jobs, var_subset=var_subset_idx)

        logger.info('  Applying correction...')
        new_batch_in = new_batch_in + correction_in
        ref_batch_in = np.concatenate((ref_batch_in, new_batch_in))
        if same_set:
            res_container.append(new_batch_in)
        else:
            new_batch_out = new_batch_out + correction_out
            ref_batch_out = np.concatenate((ref_batch_out, new_batch_out))
            res_container.append(new_batch_out)
        
        mnn_container.append(DataFrame({'new cell': mnn_new, 'ref cell': mnn_ref, 'original batch': [original_batch[mnn] for mnn in mnn_ref]}))
        original_batch += [target] * new_batch_in.shape[0]
    
    logger.info('MNN correction complete. Gathering output...')
    reflow_order = [0] * n_batch
    for i in range(n_batch):
        reflow_order[mnn_order[i]] = i
    results_ = [np.array(res_container[i]) for i in reflow_order]
    mnn_list_ = [mnn_container[i] for i in reflow_order]
    angle_list_ = None
    if do_concatenate:
        results_ = np.concatenate(tuple(results_))
    return results_, mnn_list_, angle_list_

def _transform_input_data(datas, cos_norm_in, cos_norm_out, var_index, var_subset, n_jobs):
    datas = [data.toarray().astype(np.float32) if issparse(data) else data.astype(np.float32) for data in datas]
    do_subset = False
    if var_subset is not None:
        if set(var_subset) - set(var_index):
            raise ValueError('Some items in var_subset are not in var_index.')
        if set(var_index) != set(var_subset):
            do_subset = True

    same_set = cos_norm_in == cos_norm_out and not do_subset
    
    if do_subset:
        var_sub_index = [list(var_index).index(var) for var in var_subset]
        in_batches = [data[:, var_sub_index] for data in datas]
    else:
        var_sub_index = None
        in_batches = datas

    if cos_norm_in:
        with Pool(n_jobs) as p:
            in_batches = p.map(_l2_normalize_col, in_batches)

    if cos_norm_out:
        if same_set:
            out_batches = in_batches
        else:
             with Pool(n_jobs) as p:
                out_batches = p.map(_l2_normalize_col, datas)
    else: 
        out_batches = datas
        
    return in_batches, out_batches, var_sub_index, same_set

def _l2_normalize_col(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return mat / norm

@jit(forceobj=True)
def _find_mutual_nn(data1, data2, k1, k2, n_jobs):
    workers = -1 if n_jobs is None else n_jobs
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, workers=workers)[1]
    k_index_2 = cKDTree(data2).query(x=data1, k=k2, workers=workers)[1]
    mutual_1, mutual_2 = [], []
    for index_2 in range(data2.shape[0]):
        # Handle case where k > number of points, cKDTree returns single value
        if k_index_1.ndim == 1:
            neighbors1 = [k_index_1[index_2]] if np.isscalar(k_index_1[index_2]) else k_index_1[index_2]
        else:
            neighbors1 = k_index_1[index_2]
        for index_1 in neighbors1:
            if k_index_2.ndim == 1:
                neighbors2 = [k_index_2[index_1]] if np.isscalar(k_index_2[index_1]) else k_index_2[index_1]
            else:
                neighbors2 = k_index_2[index_1]
            if index_2 in neighbors2:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2

@jit(float32[:, :](float32[:, :], float32[:, :]), nopython=True)
def _kdist(m, n):
    """
    Fast dot product (cosine similarity for normalized vectors) calculation using Numba.
    This implementation uses a manual loop to avoid NumbaPerformanceWarning on non-contiguous arrays.
    """
    assert m.shape[1] == n.shape[1]
    dist = np.zeros((m.shape[0], n.shape[0]), dtype=np.float32)
    num_features = m.shape[1]
    for i in range(m.shape[0]):
        for j in range(n.shape[0]):
            dot_product = 0.0
            for k in range(num_features):
                dot_product += m[i, k] * n[j, k]
            dist[i, j] = dot_product
    return dist

@jit(float32[:, :](float32[:, :], float32[:, :], int32[:], int32[:], float32[:, :], float32), forceobj=True)
def _compute_correction(data1, data2, mnn1, mnn2, data2_or_raw2, sigma):
    vect = data1[mnn1] - data2[mnn2]
    
    mnn_index, mnn_count = np.unique(mnn2, return_counts=True)
    vect_reduced = np.zeros((data2.shape[0], vect.shape[1]), dtype=np.float32)
    for index, ve in zip(mnn2, vect):
        vect_reduced[index] += ve
    
    vect_avg = np.divide(vect_reduced[mnn_index], mnn_count.astype(np.float32)[:, None])
    
    cosine_similarity = _kdist(data2_or_raw2, data2[mnn_index])
    cosine_distance = 1.0 - cosine_similarity
    
    exp_weights = np.exp(-cosine_distance / sigma)
    
    sum_of_weights = np.sum(exp_weights, axis=1, keepdims=True)
    sum_of_weights[sum_of_weights == 0] = 1.0
    
    normalized_weights = exp_weights / sum_of_weights
    
    output = np.dot(normalized_weights, vect_avg)
    
    return output


def _svd_internal(mat, nu, svd_mode, **kwargs):
    mat = mat.astype(np.float64)
    if svd_mode in ('svd', 'rsvd'):
        svd_out = rsvd(mat, eps_or_k=nu, rand=(svd_mode == 'rsvd'))
    else:
        raise ValueError("svd_mode must be 'svd' or 'rsvd' in this implementation.")
    return svd_out[0].astype(np.float32)

def _get_bio_span(exprs, ndim, svd_mode, var_subset=None, **kwargs):
    centred = exprs - np.mean(exprs, axis=0)
    if var_subset is not None:
        centred = centred[:, var_subset]
    
    ndim = min(ndim, *centred.shape)
    basis = _svd_internal(centred.T, ndim, svd_mode, **kwargs)
    
    if var_subset is None:
        return basis
    
    logger.warning("Biological subspace calculation on a subset of variables is simplified.")
    full_basis = np.zeros((exprs.shape[1], ndim), dtype=np.float32)
    full_basis[var_subset, :] = basis
    return full_basis
    
def _subtract_bio(*spans, correction, var_subset=None):
    total_bio_comp = np.zeros_like(correction)
    for span in spans:
        if var_subset is None:
            bio_mag = np.dot(correction, span)
            bio_comp = np.dot(bio_mag, span.T)
        else:
            bio_mag = np.dot(correction[:, var_subset], span[var_subset, :])
            bio_comp = np.zeros_like(correction)
            bio_comp[:, var_subset] = np.dot(bio_mag, span[var_subset, :].T)
        total_bio_comp += bio_comp
    
    correction -= total_bio_comp / len(spans)
    return correction