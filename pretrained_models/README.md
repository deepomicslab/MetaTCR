# TCR2vec Pre-trained Model

This directory contains the pre-trained TCR2vec model used by MetaTCR for encoding TCR clonotypes into numerical vectors.

## Model Description

TCR2vec is a transformer-based deep learning model designed for embedding T-cell Receptor (TCR) sequences into numerical vectors. The model is pre-trained using **MLM** (Masked Language Modeling) and **SPM** (Similarity Preservation Modeling) tasks, enabling it to transform amino acid sequences of TCRs into a similarity-preserved embedding space with contextual understanding.

### Key Features

- **Embedding Dimension**: 120-dimensional vectors
- **Architecture**: Transformer-based model (ProteinBERT)
- **Training**: Multi-task pre-training with MLM and SPM objectives
- **Output**: Fixed-dimensional embeddings that preserve sequence similarity

## Model Files

The pre-trained TCR2vec model requires the following files to be present in this directory:

1. **`pytorch_model.bin`**: The pre-trained model weights
2. **`args.json`**: Model configuration arguments
3. **`config.json`**: Model architecture configuration

**All three files are required** for the model to function correctly. If any file is missing, the model will not load properly.

## Download Instructions

The pre-trained TCR2vec model can be downloaded from:

**Google Drive**: https://drive.google.com/file/d/1Nj0VHpJFTUDx4X7IPQ0OGXKlGVCrwRZl/view?usp=sharing

### Manual Download

1. Download the model archive from the Google Drive link above
2. Extract the archive to this directory (`pretrained_models/TCR2vec_120/`)
3. Verify that all three required files (`pytorch_model.bin`, `args.json`, `config.json`) are present

### Automatic Download

If the model files are not found when running MetaTCR, the framework will attempt to automatically download and extract the model. However, manual download is recommended for better control and to avoid potential download issues.

## Model Source and Citation

The TCR2vec model is developed by:

- **Repository**: https://github.com/jiangdada1221/TCR2vec
- **License**: GPL-3.0

If you use the TCR2vec model in your research, please cite the original TCR2vec publication. For citation information, please refer to the [TCR2vec GitHub repository](https://github.com/jiangdada1221/TCR2vec).

## Usage in MetaTCR

The TCR2vec model is automatically loaded by MetaTCR when encoding TCR repertoires. The model path is specified in the encoding scripts:

```python
--tcr2vec_path pretrained_models/TCR2vec_120
```

The model is used to:
1. Encode individual TCR clonotypes (CDR3 sequences with V/J segments) into 120-dimensional embeddings
2. Project these embeddings onto functional cluster centroids
3. Generate standardized meta-vector representations of entire repertoires

## Technical Details

- **Input Format**: TCR sequences (amino acid strings) with V/J gene information
- **Tokenization**: Uses TAPE tokenizer with IUPAC vocabulary
- **Device**: Automatically uses GPU if available, otherwise falls back to CPU
- **Batch Processing**: Supports batch processing for efficient encoding of large datasets

## Verification

To verify that the model is correctly installed, you can check for the presence of all required files:

```bash
ls -lh pretrained_models/TCR2vec_120/
```

You should see:
- `pytorch_model.bin` (model weights file, typically several hundred MB)
- `args.json` (configuration file)
- `config.json` (architecture configuration file)

If all files are present, the model is ready to use with MetaTCR.
