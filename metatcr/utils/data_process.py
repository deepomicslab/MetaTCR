import numpy as np
import os
import pandas as pd
import random
import tqdm
import glob
from metatcr.rep2vec import compute_cluster_assignment, seqlist2ebd
from metatcr.utils.utils import save_pk, load_pkfile
from collections import defaultdict

random.seed(0)

# freq_col = 'frequencyCount (%)'  ## col name of frequency or counts
# tcr_col = 'aminoAcid'  ## col name of full length full_seq or aminoAcid


def list_filepaths(directory):
    """
    Recursively find all .tsv files in a directory.
    This function already searches through ALL subdirectories, no matter how deep.
    """
    files = []
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found - {directory}")
        return files

    # os.walk() iterates through the directory tree, including the top directory
    # and all of its subdirectories at any level.
    for root, _, filenames in os.walk(directory,followlinks=True):
        print("- Searching in dir", root)

        for filename in filenames:
            if filename.endswith('.tsv'):
                # os.path.join correctly combines the current directory path and the filename.
                files.append(os.path.join(root, filename))
    return files

def get_basenames(file_paths):
    """Extract basenames from a list of file paths."""
    return [os.path.basename(filepath) for filepath in file_paths]


def csv2meta_feature(filelist, centroids, emb_model, tcr_col='full_seq', freq_col='count (templates/reads)'):
    """
    Convert a list of TCR repertoire files into meta-feature matrices (diversity and abundance).

    Args:
        filelist (list): List of paths to .tsv files.
        centroids (np.array): Pre-computed cluster centroids.
        emb_model (TCR2vec): Loaded TCR2vec model.
        tcr_col (str): Column name for TCR sequences.
        freq_col (str): Column name for frequency/counts.

    Returns:
        tuple: A tuple containing cluster assignments, diversity matrix, and abundance matrix.
    """
    freq_features = np.zeros((len(filelist), len(centroids)))
    freqsum_features = np.zeros((len(filelist), len(centroids)))
    all_clst_ids = []

    with tqdm.tqdm(total=len(filelist), desc="Processing batch") as pbar:
        for idx, file in enumerate(filelist):
            pbar.update(1)
            try:
                df = pd.read_csv(file, sep="\t")
                if tcr_col not in df.columns or freq_col not in df.columns:
                    print(f"Warning: Skipping file {file} due to missing columns '{tcr_col}' or '{freq_col}'.")
                    continue
                
                seqlist = df[tcr_col].to_list()
                X = seqlist2ebd(seqlist, emb_model, keep_pbar=False)
                seqlabels = compute_cluster_assignment(centroids, X)

                label_counts_df = pd.DataFrame({"label": seqlabels, "freq": df[freq_col]})
                label_counts_sum = label_counts_df.groupby("label")["freq"].sum()

                for clst_id in range(len(centroids)):
                    freqsum_features[idx, clst_id] = label_counts_sum.get(clst_id, 0)
                    freq_features[idx, clst_id] = np.sum(seqlabels == clst_id)
                # all_clst_ids.extend(seqlabels)
                all_clst_ids.append(seqlabels)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return all_clst_ids, freq_features, freqsum_features


def process_dataset_to_metavec(filelist, all_smplabels, centroids, emb_model, dataset_name, out_dir, batch_size=100):
    """
    Process an entire dataset in batches and save the results.
    This version supports resuming from the last completed batch.

    Args:
        filelist (list): Complete list of file paths for the dataset.
        all_smplabels (list): Corresponding labels for each file.
        centroids (np.array): Cluster centroids.
        emb_model (object): Loaded TCR2vec model.
        dataset_name (str): Name for the output files.
        out_dir (str): Directory to save the output .pk files.
        batch_size (int): Number of files to process in each batch.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Processing dataset: {dataset_name}")
    print(f"Total number of files: {len(filelist)}")
    all_sample_list = get_basenames(filelist)

    # --- Resume Logic: Check for existing part files to find where to start ---
    start_index = 0
    # Find all part files for the current dataset
    part_files = glob.glob(os.path.join(out_dir, f"{dataset_name}_part*.pk"))
    
    if part_files:
        completed_part_nums = []
        for f in part_files:
            try:
                # Extract part number from filename, e.g., 'MyDataset_part12.pk' -> 12
                part_num_str = os.path.basename(f).split('_part')[1].split('.pk')[0]
                completed_part_nums.append(int(part_num_str))
            except (IndexError, ValueError):
                # This handles cases where a file might be named incorrectly or doesn't fit the pattern
                print(f"Warning: Could not parse part number from unexpected file: {f}")
                continue
        
        if completed_part_nums:
            last_completed_part = max(completed_part_nums)
            # The next batch to process starts right after the last completed one.
            # The starting index 'i' for the next part is `last_completed_part * batch_size`.
            start_index = last_completed_part * batch_size
            
            num_files_to_skip = start_index
            if num_files_to_skip > 0 and num_files_to_skip < len(filelist):
                print(f"Resuming from after part {last_completed_part}.")
                print(f"Skipping the first {num_files_to_skip} files that have already been processed.")
            elif num_files_to_skip >= len(filelist):
                print("All parts for this dataset appear to be processed. Nothing to do.")
                # If all work is done, we can exit the function early.
                return
    
    # --- Main Processing Loop (now starts from the calculated start_index) ---
    for i in range(start_index, len(filelist), batch_size):
        batch_files = filelist[i:i + batch_size]
        sample_list = all_sample_list[i:i + batch_size]
        sample_labels = all_smplabels[i:i + batch_size]

        # The part number is calculated based on the current index `i`
        part_num = (i // batch_size) + 1
        print(f"\n--- Processing Batch {part_num} (Files {i+1} to {min(i+batch_size, len(filelist))}) ---")

        # Assuming csv2meta_feature is defined elsewhere and works as intended
        all_clst_ids, diversity_mtx, abundance_mtx = csv2meta_feature(batch_files, centroids, emb_model)

        result_dict = {
            "diversity_mtx": diversity_mtx,
            "abundance_mtx": abundance_mtx,
            "sample_names": sample_list,
            "cluster_labels": all_clst_ids,
            "sample_labels": sample_labels,
        }

        # Suffix logic remains the same, but now it correctly reflects the part number
        suffix = '' if len(filelist) <= batch_size else f'_part{part_num}'
        output_file = os.path.join(out_dir, dataset_name + suffix + '.pk')
        print(f"Saving batch results to: {output_file}")
        save_pk(output_file, result_dict)


def merge_part_files(directory):
    """Merge batched .pk files into a single file per dataset prefix."""
    files = glob.glob(os.path.join(directory, '*_part*.pk'))
    if not files:
        return

    prefix_dict = defaultdict(list)
    for file in files:
        prefix = os.path.basename(file).split('_part')[0]
        prefix_dict[prefix].append(file)

    for prefix, part_files in prefix_dict.items():
        if len(part_files) <= 1:
            continue

        print(f"Merging {len(part_files)} part-files for prefix: {prefix}")
        merged_dict = {
            "diversity_mtx": [], "abundance_mtx": [], "sample_names": [],
            "cluster_labels": [], "sample_labels": [],
        }

        for file in sorted(part_files):
            result_dict = load_pkfile(file)
            merged_dict["diversity_mtx"].append(result_dict["diversity_mtx"])
            merged_dict["abundance_mtx"].append(result_dict["abundance_mtx"])
            merged_dict["sample_names"].extend(result_dict["sample_names"])
            merged_dict["cluster_labels"].extend(result_dict["cluster_labels"])
            merged_dict["sample_labels"].extend(result_dict["sample_labels"])

        merged_dict["diversity_mtx"] = np.concatenate(merged_dict["diversity_mtx"], axis=0)
        merged_dict["abundance_mtx"] = np.concatenate(merged_dict["abundance_mtx"], axis=0)

        output_file = os.path.join(directory, prefix + '.pk')
        print(f"Saving merged results to: {output_file}")
        save_pk(output_file, merged_dict)

        # Clean up part files after merging
        for file in part_files:
            os.remove(file)
            print(f"Removed part file: {file}")

def read_filelist(filepath_txt, return_smplist=False, file_cut_size=None):
    """
    Reads a file or list of files containing file paths, validates them, and returns a clean list.
    This corrected version safely handles file validation and empty lines.
    """
    all_lines = []
    
    # Handle both a single filepath string and a list of filepaths
    filepaths_to_read = filepath_txt if isinstance(filepath_txt, list) else [filepath_txt]
    
    for f_path in filepaths_to_read:
        try:
            with open(f_path, "r") as f:
                # Read all lines and strip whitespace from each
                lines = [line.strip() for line in f.readlines()]
                # Filter out any empty lines that might result from blank lines in the file
                all_lines.extend([line for line in lines if line])
        except FileNotFoundError:
            print(f"Warning: The file list '{f_path}' was not found. Skipping.")
            continue

    # Use a list comprehension to create a NEW list containing only valid file paths
    # This avoids modifying the list while iterating over it.
    valid_filelist = []
    for filepath in all_lines:
        if os.path.isfile(filepath):
            valid_filelist.append(filepath)
        else:
            print(f"Warning: file '{filepath}' does not exist. Removing from list.")
    
    # Apply file cut size if specified
    if file_cut_size and len(valid_filelist) > file_cut_size:
        random.shuffle(valid_filelist)
        valid_filelist = valid_filelist[:file_cut_size]

    if return_smplist:
        smp_list = [os.path.basename(f).replace('_filt_full.tsv', '') for f in valid_filelist]
        return valid_filelist, smp_list
    
    return valid_filelist


def read_filelist_from_dir(directory, format = ".tsv", suffix="_filt_full.tsv", file_cut_size=None):
    # Initialize empty lists for the file paths and case ids
    file_list = []
    case_list = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has the given suffix
        if filename.endswith(format):
            # Add the full file path to the file list
            file_list.append(os.path.join(directory, filename))
            # Remove the suffix to get the case id, and add it to the case list
            case_id = filename[:-len(suffix)]
            case_list.append(case_id)
    if file_cut_size:
        ## random select files from the filelist
        random.shuffle(file_list)
        file_list = file_list[:file_cut_size]
        case_list = [os.path.basename(f).replace(suffix, '') for f in file_list]

    return file_list, case_list

def file2seqlist(f, cut_size = 10000, tcr_col = 'aminoAcid', freq_col = 'frequencyCount (%)'):
    df = pd.read_csv(f, sep="\t", engine='c')
    df.sort_values(by=freq_col, ascending=False, inplace=True)
    df = df[df[tcr_col] != 'Failure']
    df = df[:cut_size]
    return df[tcr_col].to_list()


def merge_seq_list(filelist, cut_size = 10000, tcr_col = 'aminoAcid', multi_process=True):

    all_seqs = []

    if not multi_process:
        for file in filelist:
            all_seqs += file2seqlist(file, cut_size=cut_size, tcr_col = tcr_col)
    else:
        import multiprocessing as mp
        from functools import partial

        pool = mp.Pool(processes=mp.cpu_count())
        partial_func = partial(file2seqlist, cut_size=cut_size, tcr_col = tcr_col)
        results = pool.map(partial_func, filelist)
        pool.close()
        pool.join()
        all_seqs = sum(results, [])
    all_seqs = list(set(all_seqs))
    return all_seqs


def count_to_frequency(embeddings):
    row_sums = np.sum(embeddings, axis=1, keepdims=True)
    frequency_embeddings = embeddings / row_sums

    return frequency_embeddings


def rewrite_labels(pk_file_path, new_labels, output_path=None):
    """
    Reads a dataset .pk file, overwrites its sample labels, and saves it.

    Args:
        pk_file_path (str): Path to the source .pk file.
        new_labels (list): A new list of labels. Its length must match the number of samples.
        output_path (str, optional): Path to save the modified file. 
                                     If None, overwrites the original file. Defaults to None.
    """
    if not os.path.exists(pk_file_path):
        raise FileNotFoundError(f"Source file not found: {pk_file_path}")

    if output_path is None:
        output_path = pk_file_path
        print(f"Output path not specified. Will overwrite original file: {pk_file_path}")

    data = load_pkfile(pk_file_path)

    # --- Validation ---
    num_samples = len(data.get("sample_names", []))
    if len(new_labels) != num_samples:
        raise ValueError(
            f"Label mismatch: The number of new labels ({len(new_labels)}) does not match "
            f"the number of samples in the file ({num_samples})."
        )

    print(f"Updating labels for {os.path.basename(pk_file_path)}...")
    data["sample_labels"] = new_labels
    
    save_pk(data, output_path)
    print(f"Successfully saved updated dataset to: {output_path}")


def split_dataset_by_label(source_pk_path, output_dir, neg_subset_name="Normal", pos_subset_name="Tumor"):
    """
    Splits a dataset .pk file into two subsets based on binary labels (0 and 1).

    Args:
        source_pk_path (str): Path to the full dataset .pk file.
        output_dir (str): Directory to save the split subset files.
        neg_subset_name (str): Suffix for the subset with label 0. Defaults to "Normal".
        pos_subset_name (str): Suffix for the subset with label 1. Defaults to "Tumor".
    """
    if not os.path.exists(source_pk_path):
        print(f"Warning: Source file not found, skipping: {source_pk_path}")
        return

    print(f"Splitting dataset: {os.path.basename(source_pk_path)}")
    source_dict = load_pkfile(source_pk_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Find indices for negative (0) and positive (1) samples ---
    labels = np.array(source_dict["sample_labels"])
    neg_indices = np.where(labels == 0)[0]
    pos_indices = np.where(labels == 1)[0]
    
    print(f"  - Found {len(neg_indices)} negative samples and {len(pos_indices)} positive samples.")

    # --- Process and save the negative subset (Normal) ---
    if len(neg_indices) > 0:
        neg_dict = {}
        for key, value in source_dict.items():
            if isinstance(value, np.ndarray):
                neg_dict[key] = value[neg_indices]
            elif isinstance(value, list):
                neg_dict[key] = [value[i] for i in neg_indices]
            else: # Copy other metadata as is
                neg_dict[key] = value
        
        assert len(neg_dict["sample_names"]) == neg_dict["diversity_mtx"].shape[0], "Shape mismatch in negative subset!"
        
        base_name = os.path.basename(source_pk_path).replace('.pk', '')
        output_path = os.path.join(output_dir, f"{base_name}_{neg_subset_name}.pk")
        
        # *** CORRECTED LINE ***
        save_pk(output_path, neg_dict) 
        
        print(f"  -> Saved negative subset to: {output_path}")
    else:
        print("  - No negative samples (label 0) found to create a subset.")

    # --- Process and save the positive subset (Tumor) ---
    if len(pos_indices) > 0:
        pos_dict = {}
        for key, value in source_dict.items():
            if isinstance(value, np.ndarray):
                pos_dict[key] = value[pos_indices]
            elif isinstance(value, list):
                pos_dict[key] = [value[i] for i in pos_indices]
            else: # Copy other metadata as is
                pos_dict[key] = value
        
        assert len(pos_dict["sample_names"]) == pos_dict["diversity_mtx"].shape[0], "Shape mismatch in positive subset!"

        base_name = os.path.basename(source_pk_path).replace('.pk', '')
        output_path = os.path.join(output_dir, f"{base_name}_{pos_subset_name}.pk")
        
        # *** CORRECTED LINE ***
        save_pk(output_path, pos_dict)
        
        print(f"  -> Saved positive subset to: {output_path}")
    else:
        print("  - No positive samples (label 1) found to create a subset.")


def merge_meta_dicts(list_of_dicts):
    """
    Merges a list of data dictionaries into a single dictionary.
    This is a file-free adaptation of the merge_part_files logic.

    Args:
        list_of_dicts (list): A list of dictionaries, where each dictionary
                              has the standard data structure.

    Returns:
        dict: A single merged dictionary.
    """
    if not list_of_dicts:
        return {}

    print(f"Merging {len(list_of_dicts)} dictionaries in memory...")
    
    # Initialize a dictionary to hold the merged data
    merged_dict = {
        "diversity_mtx": [], "abundance_mtx": [], "sample_names": [],
        "cluster_labels": [], "sample_labels": [],
    }

    # Aggregate data from each dictionary in the list
    for result_dict in list_of_dicts:
        merged_dict["diversity_mtx"].append(result_dict["diversity_mtx"])
        merged_dict["abundance_mtx"].append(result_dict["abundance_mtx"])
        merged_dict["sample_names"].extend(result_dict["sample_names"])
        merged_dict["cluster_labels"].extend(result_dict["cluster_labels"])
        merged_dict["sample_labels"].extend(result_dict["sample_labels"])

    # Concatenate the numpy arrays
    merged_dict["diversity_mtx"] = np.concatenate(merged_dict["diversity_mtx"], axis=0)
    merged_dict["abundance_mtx"] = np.concatenate(merged_dict["abundance_mtx"], axis=0)
    
    print("In-memory merge successful.")
    return merged_dict
