#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate distance metrics between encoded TCR repertoire datasets.

Datasets, mapping and output paths are all given on the command line, so nothing
in this file needs to be edited. With two datasets a single pair is scored; with
more than two, every pairwise combination is scored.

Metrics: kBET, MMD, JSD, SCE, CosineDistance, iLISI.

Examples
--------
# 1) demo defaults (two bundled datasets)
python step3.calculate_dataset_distance.py

# 2) two explicit datasets (full paths)
python step3.calculate_dataset_distance.py \
    --datasets data/processed_data/datasets_mtx_1024_downstream/Huth2019.pk \
               data/processed_data/datasets_mtx_1024_downstream/Emerson2017.pk

# 3) several datasets by name -> all pairwise distances
python step3.calculate_dataset_distance.py \
    --data-dir data/processed_data/datasets_mtx_1024_downstream \
    --datasets Huth2019 Emerson2017 Wang2022
"""

import argparse
import itertools
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
    compute_ilisi,
)

DEFAULT_DATA_DIR = "./data/processed_data/datasets_mtx_1024_downstream"
DEFAULT_MAPPING_TEMPLATE = "./data/processed_data/spectral_mappings/centroid_mapping_spectral_seed0_k{k}.pk"
DEFAULT_DATASETS = ["Huth2019", "Emerson2017"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distance metrics between encoded TCR repertoire datasets.")
    parser.add_argument(
        "--datasets", nargs="+", metavar="PK", default=None,
        help="Two or more dataset .pk files (full paths, or bare names resolved "
             "under --data-dir). Two datasets score one pair; more score every "
             "pair. Default: bundled demo pair.")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory used to resolve bare dataset names (default: %(default)s).")
    parser.add_argument(
        "--mapping", default=None,
        help="Spectral mapping .pk (default: seed0 mapping for the chosen --k).")
    parser.add_argument("--k", type=int, default=128,
                        help="Number of functional clusters (default: %(default)s).")
    parser.add_argument("--feature-mode", default="concatenate",
                        choices=["concatenate", "abundance", "diversity"],
                        help="Meta-feature construction (default: %(default)s).")
    parser.add_argument("--kbet-k", type=int, default=15,
                        help="Nearest neighbors for kBET (default: %(default)s).")
    parser.add_argument("--output-dir", default="./demo_result",
                        help="Where the result table is written (default: %(default)s).")
    return parser.parse_args()


def resolve_pk(arg, data_dir):
    """Accept a full .pk path or a bare dataset name resolved under data_dir."""
    for cand in (arg, f"{arg}.pk", os.path.join(data_dir, arg), os.path.join(data_dir, f"{arg}.pk")):
        if os.path.exists(cand):
            return cand
    return arg  # not found; the loader below reports a clear error


def dataset_label(pk_path):
    """Display name for a dataset: its file basename without the .pk suffix."""
    return os.path.splitext(os.path.basename(pk_path))[0]


def load_and_process_dataset(pk_file_path, dataset_name, mapping_path, feature_mode, k):
    """Load one dataset pickle and build its meta-feature matrix."""
    if not os.path.exists(pk_file_path):
        print(f"ERROR: File not found: {pk_file_path}")
        return None

    try:
        print(f"\nLoading dataset: {dataset_name}")
        print(f"  File: {pk_file_path}")

        data_dict = load_pkfile(pk_file_path)
        print(f"  Loaded keys: {list(data_dict.keys())}")

        meta_matrix = create_meta_matrix(
            data_dict, mapping_path, feature_mode=feature_mode, k_functional=k)
        if meta_matrix is None:
            print("  ERROR: Failed to create meta matrix")
            return None

        print(f"  Meta matrix shape: {meta_matrix.shape}")
        return meta_matrix

    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_all_metrics(mtx1, mtx2, dataset1_name, dataset2_name, kbet_k):
    """Calculate all distance metrics between two datasets."""
    print(f"\n{'='*60}")
    print(f"Calculating distances between {dataset1_name} and {dataset2_name}")
    print(f"{'='*60}")
    print(f"  Dataset 1 shape: {mtx1.shape}")
    print(f"  Dataset 2 shape: {mtx2.shape}")

    results = {
        'Dataset1': dataset1_name,
        'Dataset2': dataset2_name,
        'n_samples_1': mtx1.shape[0],
        'n_samples_2': mtx2.shape[0],
    }

    metrics_to_compute = {
        'kBET': lambda: compute_kBET(mtx1, mtx2, k=kbet_k),
        'MMD': lambda: compute_mmd(mtx1, mtx2),
        'JSD': lambda: compute_jsd(mtx1, mtx2),
        'SCE': lambda: compute_ce(mtx1, mtx2),
        'CosineDistance': lambda: compute_cosine_distance(mtx1, mtx2),
        'iLISI': lambda: compute_ilisi(mtx1, mtx2),
    }

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
    args = parse_args()
    mapping_path = args.mapping or DEFAULT_MAPPING_TEMPLATE.format(k=args.k)
    dataset_args = args.datasets or DEFAULT_DATASETS

    print("=" * 60)
    print("MetaTCR Distance Metrics Calculation")
    print("=" * 60)
    print(f"Feature Mode: {args.feature_mode}")
    print(f"K Functional: {args.k}")
    print(f"kBET k parameter: {args.kbet_k}")
    print(f"Mapping File: {mapping_path}")

    if len(dataset_args) < 2:
        print("\nERROR: provide at least two datasets with --datasets.")
        return

    if not os.path.exists(mapping_path):
        print(f"\nERROR: Mapping file not found at '{mapping_path}'")
        print("Please ensure the mapping file exists or pass --mapping.")
        return

    # Resolve, load and cache each dataset once.
    pk_paths = [resolve_pk(a, args.data_dir) for a in dataset_args]
    names = [dataset_label(p) for p in pk_paths]
    matrices = []
    for pk_path, name in zip(pk_paths, names):
        mtx = load_and_process_dataset(
            pk_path, name, mapping_path, args.feature_mode, args.k)
        if mtx is None:
            print(f"\nERROR: Failed to load dataset '{name}'. Exiting.")
            return
        matrices.append(mtx)

    # Score every pair (one pair when exactly two datasets are given).
    rows = [
        calculate_all_metrics(matrices[i], matrices[j], names[i], names[j], args.kbet_k)
        for i, j in itertools.combinations(range(len(matrices)), 2)
    ]

    results_df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("Distance Metrics Results")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "step3_distance_metrics.csv")
    results_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
