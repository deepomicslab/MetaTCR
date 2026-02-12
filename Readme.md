# MetaTCR: A Framework for Analyzing Batch Effects in TCR Repertoire Datasets

MetaTCR is a computational framework designed to standardize disparate T-cell Receptor (TCR) repertoires and systematically correct for batch effects to enable robust downstream analysis. The framework transforms variable-length TCR repertoire data into fixed-dimensional meta-vectors by projecting individual repertoires onto a standardized reference space, facilitating large-scale integration and batch correction.

## Framework Overview

The MetaTCR framework consists of four main stages:

<p align="center">
  <img src="workflow/metatcr_workflow.png" alt="MetaTCR Workflow" width="800"/>
</p>

**Stage 1: Constructing a universal TCR space.** A reference database is curated from multiple TCR repertoires and hierarchical clustering is performed to establish functional TCR centroids.

**Stage 2: Projecting repertoires into the universal space.** Individual repertoires are encoded by mapping their clonotypes to reference centroids, generating standardized meta-feature matrices.

**Stage 3: Framework evaluation.** The framework's performance is evaluated using simulated and real-world data to assess metric accuracy and batch correction efficacy.

**Stage 4: Application for biological discovery.** The framework is applied to downstream tasks including batch effect identification, dataset integration, and biological discovery from corrected data.

## Installation

### Prerequisites

- Python >= 3.8

### Dependencies

Most dependencies are handled automatically by `setup.py`. However, you must install PyTorch manually according to your CUDA version.

- **PyTorch**: >= 1.9.1 (We tested on 1.9 and 1.13)
  - Please install the appropriate version for your system from [pytorch.org](https://pytorch.org/).

Other dependencies (automatically installed):
- numpy, pandas, scipy, sklearn, matplotlib, seaborn, tqdm
- tape_proteins, faiss-gpu, umap-learn, configargparse, biopython, etc.

### Install MetaTCR

1. Install Cython (required for building extensions):
```bash
pip install cython==3.1.5
```

2. Install MetaTCR package:
```bash
cd /path/to/MetaTCR
pip install .
```

Alternatively, you can install directly from the source:
```bash
pip install -e .
```

## Data Availability

The processed metadata and MetaTCR-encoded intermediate results are available at:
- **Zenodo**: https://zenodo.org/records/18265157

This repository includes:
- Processed TCR reference database
- Meta-vector representations of repertoires
- Other intermediate analysis results

### Download Requirements

Depending on where you start in the pipeline, different files are required from Zenodo:

| Starting Step | Required File | Path to Place | Note |
| :--- | :--- | :--- | :--- |
| **Step 1: Clustering** | `TCR_reference_database.full_length.txt` | `./data/raw_data/` | Only needed if you want to reconstruct the reference database from scratch. |
| **Step 2: Encoding** | (None) | - | We already provide pre-computed centroids at `./data/processed_data/`. You only need your own repertoire files. |
| **Step 3/4: Analysis** | `*.pk` (Encoded datasets) | `./data/processed_data/datasets_mtx_1024/` | Needed if you want to replicate our analysis results directly. |

For detailed information about data files, please refer to the Zenodo repository.

### Data Preprocessing Requirements

Before encoding repertoires with MetaTCR, raw TCR repertoire data must undergo quality control and preprocessing:

1. **Quality Control**: 
   - Filter out entries with CDR3β chain lengths shorter than 10 amino acids
   - Remove sequences containing stop codons
   - Retain only amino acid sequences beginning with cysteine (C) and ending with phenylalanine (F)
   - Select the most abundant clones (up to 10,000 per repertoire)

2. **Required Data Fields**:
   Each repertoire file must contain the following information:
   - CDR3 sequence (amino acid)
   - V gene annotation
   - J gene annotation
   - Clone frequency/count

3. **Full-length Sequence Reconstruction**:
   For repertoires containing only CDR3+V+J information, full-length TCR sequences can be reconstructed using the provided script:
   - **Script**: `pre_process_scripts/cdr3_to_full_seq_mod.py` ([original code reference](pre_process_scripts/cdr3_to_full_seq_mod.py#L9-L10))
   - **Usage example**: See `pre_process_scripts/demo_generate_TCR_fullseq.sh`
   
   The script reconstructs full TCR sequences by aligning CDR3 sequences with V and J gene segments from IMGT reference sequences.

4. **Input Data Format**:
   Processed repertoire files should be in TSV format with columns including `aminoAcid` (CDR3), `vMaxResolved` (V gene), `jMaxResolved` (J gene), `frequencyCount`, and `full_seq` (full-length sequence). 
   
   Example processed data format can be found in `demo_data/Emerson2017_demo/`, which serves as the input format for MetaTCR repertoire encoding.

## Pre-trained Models

### TCR2vec Model

MetaTCR uses the TCR2vec model to encode TCR clonotypes into numerical vectors. The pre-trained TCR2vec model should be placed in:

```
pretrained_models/TCR2vec_120/
```

**Model Download**: The pre-trained TCR2vec model can be downloaded from:
- **Google Drive**: https://drive.google.com/file/d/1Nj0VHpJFTUDx4X7IPQ0OGXKlGVCrwRZl/view?usp=sharing

*Note: The code supports automatic downloading if the model is missing. To use this feature, please install `gdown` (`pip install gdown`). Otherwise, you can download the model manually from the link above.*

**Model Source**: The TCR2vec model is based on the work from:
- **GitHub Repository**: https://github.com/jiangdada1221/TCR2vec

After downloading, extract the model files to `pretrained_models/TCR2vec_120/`. The directory should contain:
- `pytorch_model.bin`
- `args.json`
- `config.json`

For more details about the TCR2vec model, please refer to the README in `pretrained_models/TCR2vec_120/Readme.md`.

## Usage

### Basic Workflow

You can start from any step depending on your needs. For most users encoding their own data, **start from Step 2** using the provided pre-computed centroids.

1. **Step 1: Generate TCR Functional Clusters** (Optional)
   
   *Only required if you want to build a custom reference database from scratch.*
   ```bash
   python step1.generate_TCR_functional_clusters.py
   ```

2. **Step 2: Encode Repertoires to Meta-vectors**

   Use the pre-computed centroids (included in `./data/processed_data/`) to encode your own repertoire data.


   **For unlabeled data** (no positive/negative labels):

   The script automatically searches for all repertoire files (`.tsv`) in the directory and subdirectories. The generated MetaTCR dictionary will not contain positive/negative labels.

   ```bash
   python step2.rawdata_to_meta_encoding.py --unlabeled_dir demo_data/Emerson2017_demo --dataset_name Emerson2017_demo --tcr2vec_path ./pretrained_models/TCR2vec_120
   ```

   **For labeled data** (with separate positive and negative sample paths):
   ```bash
   python step2.rawdata_to_meta_encoding.py --pos_dir demo_data/Emerson2017_demo/CMVpos/ --neg_dir demo_data/Emerson2017_demo/CMVneg/ --dataset_name Emerson2017_demo --tcr2vec_path ./pretrained_models/TCR2vec_120
   ```

3. **Measure Quantitative Metrics**:
```bash
python step3.measure_quantitative_metrics.py
```

4. **Correct Batch Effects**:
```bash
python step4.correct_batch_effect.py
```

For detailed usage instructions and parameter descriptions, please refer to the individual script help messages:
```bash
python step2.rawdata_to_meta_encoding.py --help
```

## Citation

If you use MetaTCR in your research, please cite:

```
[Citation information to be added]
```

## License

This project is licensed under the GPL-3.0 License.

## Contact

For questions and issues, please contact:
- **Author**: Miaozhe Huo
- **Email**: miaozhhuo2-c@my.cityu.edu.hk

## Acknowledgments

MetaTCR builds upon the TCR2vec model for TCR sequence encoding. We acknowledge the original TCR2vec authors for their valuable contribution.

