import os
import tqdm
import numpy as np
import pandas as pd
import time
import torch
import glob
import configargparse
from metatcr.utils.utils import save_pk, load_pkfile
from metatcr.encoder.tcr2vec_encoder import load_tcr2vec
from metatcr.utils.data_process import process_dataset_to_metavec, list_filepaths, merge_part_files



def check_and_confirm_overwrite(output_dir, dataset_name):
    """
    Check if the final output file already exists. If so, show its contents' shape/length 
    and ask for overwrite confirmation.

    Args:
        output_dir (str): The directory where the final file would be saved.
        dataset_name (str): The name of the dataset, used for the filename.

    Returns:
        bool: True if processing should proceed, False if it should be cancelled.
    """
    final_output_file = os.path.join(output_dir, f"{dataset_name}.pk")

    if os.path.exists(final_output_file):
        print(f"Warning: Output file already exists: {final_output_file}")
        try:
            existing_data = load_pkfile(final_output_file)
            print("  Contents of the existing file:")
            
            # Iterate through the dictionary to print details of each key-value pair
            for key, value in existing_data.items():
                if isinstance(value, np.ndarray):
                    print(f"    - Key: '{key}', Type: numpy.ndarray, Shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"    - Key: '{key}', Type: list, Length: {len(value)}")
                else:
                    # For any other data types, just show the type
                    print(f"    - Key: '{key}', Type: {type(value).__name__}")

        except Exception as e:
            print(f"  - Could not read contents from existing file due to an error: {e}")

        response = input("Do you want to overwrite this file? (yes/no): ").lower().strip()
        if response == 'yes':
            print("Confirmed. The existing file will be overwritten upon completion.")
            return True
        else:
            print("Operation cancelled by user.")
            return False
    return True
'''
=== for positive and negative samples usage example:
python step2.dataset_to_meta_matrix.py \
    --neg_dir /path/to/negative/samples \
    --pos_dir /path/to/positive/samples \
    --dataset_name MyCancerDataset \
    --tcr2vec_path ./pretrained_models/TCR2vec_120


=== custom usage example:
python step2.dataset_to_meta_matrix.py \
    --neg_dir /path/to/neg_data \
    --dataset_name CustomDataset \
    --output_dir /path/to/my/results \
    --centroids_path /path/to/my/centroids.pk \
    --batch_size 50 \
    --tcr2vec_path ./pretrained_models/TCR2vec_120

'''

def main():
    """Main function to parse arguments and run the processing pipeline."""
    parser = configargparse.ArgumentParser(
        description="Convert TCR repertoire files into meta-feature matrices using pre-trained models."
    )
    
    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='A unique name for the dataset being processed.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--neg_dir', type=str, help='Directory containing negative samples (.tsv files).')
    group.add_argument('--unlabeled_dir', type=str, help='Directory containing unlabeled samples (.tsv files).')

    # This allows --pos_dir to be used with --neg_dir but not with --unlabeled_dir
    parser.add_argument('--pos_dir', type=str, help='Directory containing positive samples (.tsv files).')

    # Optional arguments
    parser.add_argument('--centroids_path', type=str, default="./data/processed_data/1024_primary_centroids.pk",
                        help='Path to the pre-computed centroids .pk file.')
    parser.add_argument('--output_dir', type=str, default="./data/processed_data/datasets_mtx_1024",
                        help='Directory to save the output feature matrices.')
    parser.add_argument('--tcr2vec_path', type=str, default='./pretrained_models/TCR2vec_120',
                        help='Path to the pre-trained TCR2vec model directory.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of files to process per batch.')

    args = parser.parse_args()
    
    # Input validation
    if args.unlabeled_dir and args.pos_dir:
        parser.error("--pos_dir cannot be used with --unlabeled_dir.")
    if not args.neg_dir and not args.pos_dir and not args.unlabeled_dir:
        parser.error("At least one input directory must be provided: --neg_dir, --pos_dir, or --unlabeled_dir.")

    # Check for existing output and confirm overwrite before proceeding
    if not check_and_confirm_overwrite(args.output_dir, args.dataset_name):
        return  # Exit if user cancels


    # Load models
    emb_model = load_tcr2vec(args.tcr2vec_path)
    centroids = load_pkfile(args.centroids_path)

    # Prepare data paths and labels
    data_paths = []
    data_labels = []

    if args.neg_dir:
        neg_paths = list_filepaths(args.neg_dir)
        data_paths.extend(neg_paths)
        data_labels.extend([0] * len(neg_paths))
    
    if args.pos_dir:
        pos_paths = list_filepaths(args.pos_dir)
        data_paths.extend(pos_paths)
        data_labels.extend([1] * len(pos_paths))

    if args.unlabeled_dir:
        unlabeled_paths = list_filepaths(args.unlabeled_dir)
        data_paths.extend(unlabeled_paths)
        data_labels.extend(["NA"] * len(unlabeled_paths))

    if not data_paths:
        print("No .tsv files found in the provided directories. Exiting.")
        return

    # Run the processing pipeline
    process_dataset_to_metavec(
        data_paths, data_labels, centroids, emb_model,
        args.dataset_name, args.output_dir, args.batch_size
    )

    # Merge part files if any were created
    merge_part_files(args.output_dir)
    print("Processing complete.")


if __name__ == "__main__":
    main()