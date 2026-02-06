## encode the TCR repertoire into meta vectors
import numpy as np
import os
import pandas as pd
import torch
import faiss
import pickle
from metatcr.encoder.tcr2vec_encoder import seqlist2ebd, load_tcr2vec


def load_pkfile(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)

def transform_to_functional_clusters(primary_mtx, mapping_file, k_func):
    # print(f"Transforming primary features (1024) to functional features ({k_func})...")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    mapping = load_pkfile(mapping_file)
    n_samples = primary_mtx.shape[0]
    functional_mtx = np.zeros((n_samples, k_func))
    
    for primary_id, func_id in mapping.items():
        if primary_id < primary_mtx.shape[1] and func_id < k_func:
            functional_mtx[:, func_id] += primary_mtx[:, primary_id]
            
    return functional_mtx


def compute_cluster_assignment(centroids, x):
    '''
    Assign each tcr embedding in x to the closest centroid
    :param centroids: np.array, shape = (n_centroids, d)
    :param x: np.array, shape = (n_samples, d)
    :return: np.array, shape = (n_samples,)
    '''
    assert centroids is not None, "'centroids' file is not found. Should identify reference centroids before assigning repertoire to meta vectors"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()

def kmeans_clustering(X, cluster_num = 50):
    '''
    :param X: np.array, shape = (n_samples, d)
    :param cluster_num: int, number of clusters
    :return: np.array, shape = (n_samples,), np.array, shape = (n_clusters, d)
    '''
    assert np.all(~np.isnan(X)), 'x contains NaN'
    assert np.all(np.isfinite(X)), 'x contains Inf'
    d = X.shape[1]
    kmeans = faiss.Clustering(d, cluster_num)
    kmeans.verbose = bool(0)
    kmeans.niter = 100
    kmeans.nredo = 1

    # otherwise the kmeans implementation sub-samples the training set
    kmeans.max_points_per_centroid = 10000000

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    kmeans.train(X, index)
    # seq_labels = km.labels_
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(cluster_num, d)
    labels = compute_cluster_assignment(centroids, X)

    ## get the obj value
    iteration_stats = kmeans.iteration_stats
    # Get the number of iterations
    num_iterations = iteration_stats.size()

    # Get the stat for the last iteration
    last_iteration_stat = iteration_stats.at(num_iterations - 1)

    return labels, centroids, last_iteration_stat

def seqlist2clst(seq_list, emb_model, cluster_num = 50):
    '''
    :param seq_list: list, list of TCR sequences
    :param emb_model: torch.nn.Module, TCR2vec model as default encoder (in device)
    :param cluster_num: int, number of reference clusters
    '''

    X = seqlist2ebd(seq_list, emb_model=emb_model, keep_pbar=True)
    labels, centroids, _ = kmeans_clustering(X, cluster_num)

    seq2label = dict(zip(seq_list, labels))
    return seq2label, centroids  ## dict, list

def kmeans_traverse_k(X, cluster_range):  ## X is a 2-dim np.array, cluster_range is a list

    all_cluster_labels = []
    all_centers = []
    all_errs = []
    all_imbalances = []
    length = X.shape[0]

    for cluster_num in cluster_range:
        print(f"try clustering with {cluster_num} clusters...")
        labels, centroids, last_stat = kmeans_clustering(X, cluster_num)

        all_cluster_labels.append(labels)
        all_centers.append(centroids)
        all_errs.append(last_stat.obj/length)
        all_imbalances.append(last_stat.imbalance_factor)

    return all_cluster_labels, all_centers, all_errs, all_imbalances


def assign_clsts(seq_list, centroids):  ## ebd_mtx: nd.array, output mtx: tensor

    total_clst_num = len(centroids)

    X = seqlist2ebd(seq_list, emb_model=emb_model, keep_pbar = False)
    assert np.all(~np.isnan(X)), 'x contains NaN'
    assert np.all(np.isfinite(X)), 'x contains Inf'

    labels = compute_cluster_assignment(centroids, X)
    # ebd_dim = X.shape[1]

    total_clst = [i for i in range(total_clst_num)]

    ## get sub mtx when label == clst_id
    bag_features = []
    for clst_id in total_clst:
        clst_ebd_mtx = X[labels == clst_id]
        if clst_ebd_mtx.shape[0] == 0:
            clst_ebd_mtx = np.zeros((1, X.shape[1]))
        clst_ebd_mtx = torch.from_numpy(clst_ebd_mtx)
        bag_features.append(clst_ebd_mtx)

    return bag_features, X.shape[1]

def create_meta_matrix(
    data,
    mapping_file,
    feature_mode="concatenate",
    k_functional=96,
    log_transform_abundance=True,
    diversity_key="diversity_mtx",
    abundance_key="abundance_mtx"
):
    """
    Encodes raw data matrices into a standardized feature matrix for downstream tasks.

    This function performs several key steps:
    1. Transforms primary features (e.g., 1024-dim) into functional cluster features (k-dim)
       using a provided mapping file.
    2. Based on `feature_mode`, processes and returns the specified feature matrix/matrices.
       - Normalizes abundance and diversity matrices to proportions.
       - Optionally applies a log1p transformation to the abundance matrix.

    Args:
        data (dict): Dictionary containing raw data, expected to have keys specified by
                     `abundance_key` and `diversity_key`.
        mapping_file (str): Path to the pickle file that maps primary cluster IDs to
                            functional cluster IDs.
        feature_mode (str): The method for generating features. Supported modes are:
                            - "concatenate": Concatenates processed abundance and diversity matrices.
                            - "abundance": Returns only the processed abundance matrix.
                            - "diversity": Returns only the processed diversity matrix.
        k_functional (int): The target number of functional clusters.
        log_transform_abundance (bool): If True, applies np.log1p to the normalized abundance matrix.
                                      This only affects "abundance" and "concatenate" modes.
        diversity_key (str): The dictionary key for the diversity matrix in the `data` dict.
        abundance_key (str): The dictionary key for the abundance matrix in the `data` dict.

    Returns:
        np.ndarray or None: The final feature matrix, or None if required data keys are missing
                            for the specified feature mode.
    """
    EPS = 1e-9
    PSEUDOCOUNT = 1.0  # Used to normalize to proportions

    # Step 1: Transform to functional cluster space if necessary
    if k_functional == data[abundance_key].shape[1]:
        functional_abun = data.get(abundance_key)
        functional_div = data.get(diversity_key)
    else:
        functional_abun = transform_to_functional_clusters(data[abundance_key], mapping_file, k_functional) if abundance_key in data else None
        functional_div = transform_to_functional_clusters(data[diversity_key], mapping_file, k_functional) if diversity_key in data else None

    processed_abun = None
    processed_div = None

    # Step 2: Process abundance matrix if required by the feature_mode
    if feature_mode in ["abundance", "concatenate"]:
        if functional_abun is None:
            print(f"    - SKIPPING: Required key '{abundance_key}' not found for mode '{feature_mode}'.")
            return None
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalize abundance matrix to proportions
            abun_sum = functional_abun.sum(axis=1, keepdims=True)
            norm_abun = (functional_abun / (abun_sum + EPS)) * PSEUDOCOUNT
            norm_abun = np.nan_to_num(norm_abun)
            
            # Optionally apply log transformation
            if log_transform_abundance:
                processed_abun = np.log1p(norm_abun)
            else:
                processed_abun = norm_abun

    # Step 3: Process diversity matrix if required by the feature_mode
    if feature_mode in ["diversity", "concatenate"]:
        if functional_div is None:
            print(f"    - SKIPPING: Required key '{diversity_key}' not found for mode '{feature_mode}'.")
            return None

        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalize diversity matrix to proportions
            div_sum = functional_div.sum(axis=1, keepdims=True)
            norm_div = functional_div / (div_sum + EPS)
            processed_div = np.nan_to_num(norm_div)

    # Step 4: Return the final matrix based on the feature_mode
    if feature_mode == "abundance":
        return processed_abun
    elif feature_mode == "diversity":
        return processed_div
    elif feature_mode == "concatenate":
        # Both processed_abun and processed_div should have been computed
        return np.hstack([processed_abun, processed_div])
    else:
        raise ValueError(f"Unsupported FEATURE_MODE: '{feature_mode}'. Supported modes are 'concatenate', 'abundance', 'diversity'.")