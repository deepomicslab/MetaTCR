#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script for calculating distance metrics between encoded TCR repertoire datasets.

This script demonstrates how to:
1. Load encoded repertoire data from pickle files
2. Create meta-feature matrices
3. Calculate various distance metrics (kBET, MMD, JSD, CrossEntropy, CosineDistance, iLISI)
4. Display results in a table format
"""

import os
import numpy as np
import pandas as pd
from metatcr.rep2vec import load_pkfile, create_meta_matrix
from metatcr.metrics import (
    compute_kBET,
    compute_mmd,
    compute_jsd,
    compute_ce,
    compute_cosine_distance,
    compute_ilisi
)

# Configuration parameters
K_FUNCTIONAL = 96
FEATURE_MODE = "concatenate"  # Options: "concatenate", "abundance", "diversity"
KBET_K = 15  # Number of nearest neighbors for kBET

# File paths - modify these according to your data
# Example: Load two datasets for comparison
PK_FILE_1 = "./data/processed_data/datasets_mtx_1024/Huth2019.pk"
PK_FILE_2 = "./data/processed_data/datasets_mtx_1024/Emerson2017-Keck.pk"

# Mapping file path - adjust according to your setup
MAPPING_FILE_TEMPLATE = "./data/processed_data/spectral_centroids/centroid_mapping_spectral_k{k}.pk"
MAPPING_FILE_PATH = MAPPING_FILE_TEMPLATE.format(k=K_FUNCTIONAL)

# Dataset names for display
DATASET_NAME_1 = "Huth2019.pk"
DATASET_NAME_2 = "Emerson2017-Keck"


def load_and_process_dataset(pk_file_path, dataset_name):
    """
    Load a dataset from pickle file and create meta-feature matrix.
    
    Args:
        pk_file_path (str): Path to the pickle file containing encoded repertoire data
        dataset_name (str): Name of the dataset for display purposes
    
    Returns:
        np.ndarray or None: Meta-feature matrix, or None if loading fails
    """
    if not os.path.exists(pk_file_path):
        print(f"ERROR: File not found: {pk_file_path}")
        return None
    
    try:
        print(f"\nLoading dataset: {dataset_name}")
        print(f"  File: {pk_file_path}")
        
        # Load the pickle file
        data_dict = load_pkfile(pk_file_path)
        print(f"  Loaded keys: {list(data_dict.keys())}")
        
        # Check if mapping file exists
        if not os.path.exists(MAPPING_FILE_PATH):
            print(f"  WARNING: Mapping file not found at {MAPPING_FILE_PATH}")
            print(f"  Please ensure the mapping file exists or adjust MAPPING_FILE_TEMPLATE")
            return None
        
        # Create meta-feature matrix
        meta_matrix = create_meta_matrix(
            data_dict,
            MAPPING_FILE_PATH,
            feature_mode=FEATURE_MODE,
            k_functional=K_FUNCTIONAL
        )
        
        if meta_matrix is None:
            print(f"  ERROR: Failed to create meta matrix")
            return None
        
        print(f"  Meta matrix shape: {meta_matrix.shape}")
        return meta_matrix
        
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_all_metrics(mtx1, mtx2, dataset1_name, dataset2_name):
    """
    Calculate all distance metrics between two datasets.
    
    Args:
        mtx1 (np.ndarray): Meta-feature matrix of dataset 1
        mtx2 (np.ndarray): Meta-feature matrix of dataset 2
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
    
    Returns:
        dict: Dictionary containing all metric values
    """
    print(f"\n{'='*60}")
    print(f"Calculating distances between {dataset1_name} and {dataset2_name}")
    print(f"{'='*60}")
    print(f"  Dataset 1 shape: {mtx1.shape}")
    print(f"  Dataset 2 shape: {mtx2.shape}")
    
    results = {
        'Dataset1': dataset1_name,
        'Dataset2': dataset2_name,
        'n_samples_1': mtx1.shape[0],
        'n_samples_2': mtx2.shape[0]
    }
    
    # Define all metrics to compute
    metrics_to_compute = {
        'kBET': lambda: compute_kBET(mtx1, mtx2, k=KBET_K),
        'MMD': lambda: compute_mmd(mtx1, mtx2),
        'JSD': lambda: compute_jsd(mtx1, mtx2),
        'CrossEntropy': lambda: compute_ce(mtx1, mtx2),
        'CosineDistance': lambda: compute_cosine_distance(mtx1, mtx2),
        'iLISI': lambda: compute_ilisi(mtx1, mtx2)
    }
    
    # Calculate each metric
    for metric_name, metric_func in metrics_to_compute.items():
        try:
            value = metric_func()
            results[metric_name] = value
            print(f"  {metric_name:15s}: {value:.6f}")
        except Exception as e:
            print(f"  {metric_name:15s}: ERROR - {e}")
            results[metric_name] = np.nan
    
    return results


def main():
    """
    Main function to demonstrate distance metric calculation.
    """
    print("="*60)
    print("MetaTCR Distance Metrics Calculation Demo")
    print("="*60)
    print(f"Feature Mode: {FEATURE_MODE}")
    print(f"K Functional: {K_FUNCTIONAL}")
    print(f"kBET k parameter: {KBET_K}")
    print(f"Mapping File: {MAPPING_FILE_PATH}")
    
    # Check mapping file
    if not os.path.exists(MAPPING_FILE_PATH):
        print(f"\nERROR: Mapping file not found at '{MAPPING_FILE_PATH}'")
        print(f"Please ensure the mapping file exists or adjust MAPPING_FILE_TEMPLATE in the script.")
        return
    
    # Load both datasets
    mtx1 = load_and_process_dataset(PK_FILE_1, DATASET_NAME_1)
    if mtx1 is None:
        print("\nERROR: Failed to load first dataset. Exiting.")
        return
    
    mtx2 = load_and_process_dataset(PK_FILE_2, DATASET_NAME_2)
    if mtx2 is None:
        print("\nERROR: Failed to load second dataset. Exiting.")
        return
    
    # Calculate distances between two datasets
    results = calculate_all_metrics(mtx1, mtx2, DATASET_NAME_1, DATASET_NAME_2)
    
    # Create and display results table
    results_df = pd.DataFrame([results])
    print(f"\n{'='*60}")
    print("Distance Metrics Results")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

