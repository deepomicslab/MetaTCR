#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script for batch effect correction between encoded TCR repertoire datasets.

This script demonstrates how to:
1. Load encoded repertoire data from pickle files
2. Create meta-feature matrices
3. Apply batch correction methods (Covariance matching, Harmony, MNN, Scanorama)
4. Evaluate correction effectiveness using distance metrics (kBET, JSD)
5. Display results in a table format
"""

import os
import numpy as np
import pandas as pd
from metatcr.rep2vec import load_pkfile, create_meta_matrix
from metatcr.integration import coral_mtx, harmony_mtx, mnn_mtx, scanorama_mtx
from metatcr.metrics import compute_kBET, compute_jsd

# Configuration parameters - same as step3
K_FUNCTIONAL = 96
FEATURE_MODE = "concatenate"  # Options: "concatenate", "abundance", "diversity"
KBET_K = 15  # Number of nearest neighbors for kBET

# File paths - modify these according to your data
# Example: Load two datasets for batch correction
PK_FILE_1 = "./data/processed_data/datasets_mtx_1024/Huth2019.pk"
PK_FILE_2 = "./data/processed_data/datasets_mtx_1024/Emerson2017-Keck.pk"

# Mapping file path - adjust according to your setup
MAPPING_FILE_TEMPLATE = "./data/processed_data/spectral_centroids/centroid_mapping_spectral_k{k}.pk"
MAPPING_FILE_PATH = MAPPING_FILE_TEMPLATE.format(k=K_FUNCTIONAL)

# Dataset names for display
DATASET_NAME_1 = "Huth2019"
DATASET_NAME_2 = "Emerson2017-Keck"

# Integration method parameters
MNN_K = 15  # Number of neighbors for MNN
MNN_HVG = 50  # Number of highly variable genes for MNN
SCANORAMA_K = 5  # Number of neighbors for Scanorama
SCANORAMA_SIGMA = 15  # Sigma parameter for Scanorama
SCANORAMA_ALPHA = 0.1  # Alpha parameter for Scanorama
SCANORAMA_HVG = 50  # Number of highly variable genes for Scanorama

EPS = 1e-9  # Small constant to prevent division by zero


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


def calculate_metrics(mtx1, mtx2, dataset1_name, dataset2_name):
    """
    Calculate distance metrics between two datasets.
    
    Args:
        mtx1 (np.ndarray): Meta-feature matrix of dataset 1
        mtx2 (np.ndarray): Meta-feature matrix of dataset 2
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
    
    Returns:
        dict: Dictionary containing metric values
    """
    results = {}
    
    # Calculate JSD (requires non-negative values)
    try:
        mtx1_pos = np.maximum(mtx1, EPS)
        mtx2_pos = np.maximum(mtx2, EPS)
        jsd_value = compute_jsd(mtx1_pos, mtx2_pos)
        results['JSD'] = jsd_value
    except Exception as e:
        print(f"  ERROR calculating JSD: {e}")
        results['JSD'] = np.nan
    
    # Calculate kBET
    try:
        kbet_value = compute_kBET(mtx1, mtx2, k=KBET_K)
        results['kBET'] = kbet_value
    except Exception as e:
        print(f"  ERROR calculating kBET: {e}")
        results['kBET'] = np.nan
    
    return results


def apply_batch_correction(source_mtx, target_mtx, method_name):
    """
    Apply batch correction using the specified method.
    
    Args:
        source_mtx (np.ndarray): Source matrix to be corrected
        target_mtx (np.ndarray): Target matrix (reference)
        method_name (str): Name of the correction method
    
    Returns:
        tuple: (corrected_source, corrected_target) or None if failed
    """
    try:
        if method_name == "Covariance matching":
            return coral_mtx(source_mtx=source_mtx, target_mtx=target_mtx)
        elif method_name == "Harmony":
            return harmony_mtx(source_mtx=source_mtx, target_mtx=target_mtx)
        elif method_name == "MNN":
            return mnn_mtx(source_mtx=source_mtx, target_mtx=target_mtx, k=MNN_K, hvg=MNN_HVG)
        elif method_name == "Scanorama":
            return scanorama_mtx(
                source_mtx=source_mtx, 
                target_mtx=target_mtx, 
                k=SCANORAMA_K, 
                sigma=SCANORAMA_SIGMA, 
                alpha=SCANORAMA_ALPHA, 
                hvg=SCANORAMA_HVG
            )
        else:
            print(f"  ERROR: Unknown method '{method_name}'")
            return None
    except Exception as e:
        print(f"  ERROR applying {method_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to demonstrate batch effect correction.
    """
    print("="*60)
    print("MetaTCR Batch Effect Correction Demo")
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
    
    # Calculate baseline metrics (before correction)
    print(f"\n{'='*60}")
    print("Baseline Metrics (Before Correction)")
    print(f"{'='*60}")
    print(f"  Dataset 1 shape: {mtx1.shape}")
    print(f"  Dataset 2 shape: {mtx2.shape}")
    
    baseline_metrics = calculate_metrics(mtx1, mtx2, DATASET_NAME_1, DATASET_NAME_2)
    print(f"  JSD:  {baseline_metrics.get('JSD', np.nan):.6f}")
    print(f"  kBET: {baseline_metrics.get('kBET', np.nan):.6f}")
    
    # Prepare results storage
    results_list = []
    
    # Add baseline results
    baseline_result = {
        'Method': 'Without integration',
        'JSD': baseline_metrics.get('JSD', np.nan),
        'kBET': baseline_metrics.get('kBET', np.nan)
    }
    results_list.append(baseline_result)
    
    # Define correction methods to test
    correction_methods = [
        "Covariance matching",
        "Harmony",
        "MNN",
        "Scanorama"
    ]
    
    # Apply each correction method
    print(f"\n{'='*60}")
    print("Applying Batch Correction Methods")
    print(f"{'='*60}")
    
    for method_name in correction_methods:
        print(f"\n--- {method_name} ---")
        
        # Apply correction
        # Note: We correct mtx1 (source) to align with mtx2 (target)
        correction_result = apply_batch_correction(mtx1, mtx2, method_name)
        
        if correction_result is None:
            print(f"  Method failed, skipping...")
            result = {
                'Method': method_name,
                'JSD': np.nan,
                'kBET': np.nan
            }
            results_list.append(result)
            continue
        
        # Unpack corrected matrices
        if isinstance(correction_result, (tuple, list)):
            corrected_source = correction_result[0]
            corrected_target = correction_result[1]
        else:
            corrected_source = correction_result
            corrected_target = mtx2
        
        print(f"  Corrected source shape: {corrected_source.shape}")
        print(f"  Corrected target shape: {corrected_target.shape}")
        
        # Calculate metrics after correction
        corrected_metrics = calculate_metrics(
            corrected_source, 
            corrected_target, 
            f"{DATASET_NAME_1}_corrected",
            f"{DATASET_NAME_2}_corrected"
        )
        
        print(f"  JSD after correction:  {corrected_metrics.get('JSD', np.nan):.6f}")
        print(f"  kBET after correction: {corrected_metrics.get('kBET', np.nan):.6f}")
        
        # Calculate improvement
        jsd_improvement = baseline_metrics.get('JSD', np.nan) - corrected_metrics.get('JSD', np.nan)
        kbet_improvement = baseline_metrics.get('kBET', np.nan) - corrected_metrics.get('kBET', np.nan)
        
        print(f"  JSD improvement:  {jsd_improvement:.6f} (lower is better)")
        print(f"  kBET improvement: {kbet_improvement:.6f} (lower is better)")
        
        result = {
            'Method': method_name,
            'JSD': corrected_metrics.get('JSD', np.nan),
            'kBET': corrected_metrics.get('kBET', np.nan),
            'JSD_improvement': jsd_improvement,
            'kBET_improvement': kbet_improvement
        }
        results_list.append(result)
    
    # Create and display results table
    results_df = pd.DataFrame(results_list)
    print(f"\n{'='*60}")
    print("Batch Correction Results Summary")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # # Save results to CSV
    # output_file = "batch_correction_results.csv"
    # results_df.to_csv(output_file, index=False, float_format='%.6f')
    # print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNote: Lower values for JSD and kBET indicate better batch mixing.")
    print("      Positive improvement values indicate successful correction.")


if __name__ == "__main__":
    main()

