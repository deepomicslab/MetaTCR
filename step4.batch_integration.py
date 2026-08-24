#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrate encoded TCR repertoire datasets and score cohort mixing.

Two integration modes, selected with --mode:
  * pairwise : align a source dataset to a target dataset (exactly two datasets).
  * multi    : jointly integrate several datasets at once (two or more).

Datasets and paths are supplied on the command line; nothing in this file needs
to be edited. With no arguments, both modes run on the bundled demo datasets.

Mixing is scored with kBET and maximum mean discrepancy (MMD); for both, a lower
value means the cohorts are better mixed (less study-associated separation).

Examples
--------
# demo: both modes on the bundled datasets
python step4.batch_integration.py

# pairwise: source aligned to target
python step4.batch_integration.py --mode pairwise \
    --datasets data/processed_data/datasets_mtx_1024_downstream/Huth2019.pk \
               data/processed_data/datasets_mtx_1024_downstream/Emerson2017.pk

# multi: several datasets by name (resolved under --data-dir)
python step4.batch_integration.py --mode multi \
    --data-dir data/processed_data/datasets_mtx_1024_downstream \
    --datasets Huth2019 Heather2015 Wang2022 Wright2026
"""

import argparse
import csv
import itertools
import os

import numpy as np

from metatcr.rep2vec import load_pkfile, create_meta_matrix
from metatcr.integration import (
    covmatch_mtx,
    covmatch_multi_mtx,
    harmony_mtx,
    harmony_multi_mtx,
    mnn_mtx,
    scanorama_mtx,
)
from metatcr.metrics import compute_kBET, compute_mmd

DEFAULT_DATA_DIR = "./data/processed_data/datasets_mtx_1024_downstream"
DEFAULT_MAPPING_TEMPLATE = "./data/processed_data/spectral_mappings/centroid_mapping_spectral_seed0_k{k}.pk"
DEFAULT_PAIR = ["Huth2019", "Emerson2017"]                          # source, target
DEFAULT_MULTI = ["Huth2019", "Heather2015", "Wang2022", "Wright2026"]

# Runtime configuration; main() fills these from the parsed arguments.
K = 128
MAPPING = DEFAULT_MAPPING_TEMPLATE.format(k=K)
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = "./demo_result"
KBET_K = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrate encoded TCR repertoire datasets and score cohort mixing.")
    parser.add_argument(
        "--mode", choices=["pairwise", "multi"], default=None,
        help="pairwise (source -> target, two datasets) or multi (joint, two or "
             "more). Default: run both demos on the bundled datasets.")
    parser.add_argument(
        "--datasets", nargs="+", metavar="PK", default=None,
        help="Dataset .pk files (full paths, or bare names resolved under "
             "--data-dir). pairwise expects exactly two (source then target); "
             "multi expects two or more.")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory used to resolve bare dataset names (default: %(default)s).")
    parser.add_argument(
        "--mapping", default=None,
        help="Spectral mapping .pk (default: seed0 mapping for the chosen --k).")
    parser.add_argument("--k", type=int, default=128,
                        help="Number of functional clusters (default: %(default)s).")
    parser.add_argument("--kbet-k", type=int, default=15,
                        help="Nearest neighbors for kBET (default: %(default)s).")
    parser.add_argument("--output-dir", default="./demo_result",
                        help="Where result tables are written (default: %(default)s).")
    return parser.parse_args()


def resolve_pk(arg):
    """Accept a full .pk path or a bare dataset name resolved under DATA_DIR."""
    for cand in (arg, f"{arg}.pk", os.path.join(DATA_DIR, arg), os.path.join(DATA_DIR, f"{arg}.pk")):
        if os.path.exists(cand):
            return cand
    return arg


def dataset_label(arg):
    """Display name for a dataset: its file basename without the .pk suffix."""
    return os.path.splitext(os.path.basename(resolve_pk(arg)))[0]


def to_meta(name_or_path):
    """Load one dataset pickle and build its meta-feature matrix (concatenate)."""
    data = load_pkfile(resolve_pk(name_or_path))
    return create_meta_matrix(data, MAPPING, feature_mode="concatenate", k_functional=K)


def scores(a, b):
    """kBET and MMD between two cohorts (lower = better mixed)."""
    return compute_kBET(a, b, k=KBET_K), compute_mmd(a, b)


def write_rows(filename, rows):
    """Write a (method, kBET, MMD) table under the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as handle:
        csv.writer(handle).writerows([("method", "kBET", "MMD"), *rows])
    print(f"saved -> {path}")


def paired_batch(datasets):
    """Pairwise integration of two datasets (source aligned to target)."""
    src_name, tgt_name = dataset_label(datasets[0]), dataset_label(datasets[1])
    print(f"\n=== pairwise integration ({src_name} -> {tgt_name}) ===")
    source, target = to_meta(datasets[0]), to_meta(datasets[1])

    kbet, mmd = scores(source, target)
    print(f"{'Without integration':22s} kBET={kbet:.4f}  MMD={mmd:.4f}")
    rows = [("Without integration", kbet, mmd)]

    methods = {
        "Covariance Matching": lambda: covmatch_mtx(source_mtx=source, target_mtx=target),
        "Harmony": lambda: harmony_mtx(source_mtx=source, target_mtx=target),
        "MNN": lambda: mnn_mtx(source_mtx=source, target_mtx=target),
        "Scanorama": lambda: scanorama_mtx(source_mtx=source, target_mtx=target),
    }
    for name, run in methods.items():
        try:
            result = run()  # each method returns the corrected (source, target) first
            kbet, mmd = scores(result[0], result[1])
            print(f"{name:22s} kBET={kbet:.4f}  MMD={mmd:.4f}")
            rows.append((name, kbet, mmd))
        except Exception as exc:
            print(f"{name:22s} FAILED: {exc}")
            rows.append((name, float("nan"), float("nan")))

    write_rows("step4_paired_batch.csv", rows)


def multi_batch(datasets):
    """Multi-batch integration of several datasets at once.

    Covariance Matching and Harmony (two methods that work relatively well here)
    are shown as examples. Mixing is the mean kBET/MMD over all cohort pairs.
    """
    names = [dataset_label(d) for d in datasets]
    print(f"\n=== multi-batch integration ({len(names)} datasets: {', '.join(names)}) ===")
    blocks = [to_meta(d) for d in datasets]

    def mean_scores(cohorts):
        pairs = [scores(cohorts[i], cohorts[j])
                 for i, j in itertools.combinations(range(len(cohorts)), 2)]
        return np.mean([k for k, _ in pairs]), np.mean([s for _, s in pairs])

    kbet, mmd = mean_scores(blocks)
    print(f"{'Without integration':22s} kBET={kbet:.4f}  MMD={mmd:.4f}")
    rows = [("Without integration", kbet, mmd)]

    # Covariance Matching: align every cohort to an equal-cohort pooled target.
    covmatched = covmatch_multi_mtx(blocks)
    kbet, mmd = mean_scores(covmatched)
    print(f"{'Covariance Matching':22s} kBET={kbet:.4f}  MMD={mmd:.4f}")
    rows.append(("Covariance Matching", kbet, mmd))

    # Harmony: joint multi-batch integration (PCA-50 default; same core as harmony_mtx).
    harmonized = harmony_multi_mtx(blocks, verbose=False)
    kbet, mmd = mean_scores(harmonized)
    print(f"{'Harmony':22s} kBET={kbet:.4f}  MMD={mmd:.4f}")
    rows.append(("Harmony", kbet, mmd))

    write_rows("step4_multi_batch.csv", rows)


def main():
    global K, MAPPING, DATA_DIR, OUTPUT_DIR, KBET_K
    args = parse_args()
    K = args.k
    MAPPING = args.mapping or DEFAULT_MAPPING_TEMPLATE.format(k=K)
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    KBET_K = args.kbet_k

    if not os.path.exists(MAPPING):
        print(f"ERROR: Mapping file not found at '{MAPPING}'. Pass --mapping to override.")
        return

    # No mode: run both demos on the bundled defaults (backward-compatible).
    if args.mode is None:
        if args.datasets:
            print("ERROR: --datasets requires --mode pairwise or --mode multi.")
            return
        paired_batch(DEFAULT_PAIR)
        multi_batch(DEFAULT_MULTI)
        return

    if args.mode == "pairwise":
        datasets = args.datasets or DEFAULT_PAIR
        if len(datasets) != 2:
            print("ERROR: --mode pairwise needs exactly two datasets (source target).")
            return
        paired_batch(datasets)
    else:  # multi
        datasets = args.datasets or DEFAULT_MULTI
        if len(datasets) < 2:
            print("ERROR: --mode multi needs at least two datasets.")
            return
        multi_batch(datasets)


if __name__ == "__main__":
    main()
