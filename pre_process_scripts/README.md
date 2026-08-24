# Pre-processing: full-length TCRβ reconstruction

MetaTCR encodes repertoires from the **full-length** TCRβ amino-acid sequence (`full_seq`).
If your repertoire files only carry CDR3 + V gene + J gene, use `cdr3_to_full_seq.py` here to
reconstruct `full_seq` before running `step2.dataset_to_meta_matrix.py`.

The reconstruction aligns the CDR3 to the IMGT V- and J-segment amino-acid sequences and
extends it in both directions, so `full_seq = V region + CDR3 + J region`. A row is marked
`Failure` when the V segment is not found or the reconstruction is shorter than 50 aa. Every
input column is kept; only a `full_seq` column is added (or overwritten if already present).

## Contents

| file | purpose |
|---|---|
| `cdr3_to_full_seq.py` | the reconstruction script (adapted from TITAN's `cdr3_to_full_seq.py`); parses the V/J FASTA once per worker and processes files in parallel |
| `TCR_gene_segment_data/` | IMGT V/J segment FASTA the reconstruction reads |
| `demo_filt_input.tsv` | a small filt-style input (CDR3 + V + J, no `full_seq`) |
| `demo_generate_TCR_fullseq.sh` | runnable demo that adds `full_seq` to the demo input |

## Usage

```bash
python cdr3_to_full_seq.py your_repertoire_filt.tsv [more_files ...] \
    --segment-dir TCR_gene_segment_data \    # dir with V_/J_segment_sequences.fasta
    --v-header vMaxResolved \                 # V-gene column header
    --j-header jMaxResolved \                 # J-gene column header
    --cdr3-header aminoAcid \                 # CDR3 column header
    --out-dir OUTDIR \                        # output dir (default: alongside each input)
    --in-suffix _filt.tsv --out-suffix _filt_full.tsv \
    --procs 16                                # parallel workers
```

Each output is the input table with a `full_seq` column added (TSV). Add `--drop-failures`
to drop rows whose reconstruction failed (default keeps them marked `Failure`).

Quick demo (uses the bundled demo input and FASTA):

```bash
bash demo_generate_TCR_fullseq.sh
```

Dependency: `biopython` (`Bio.pairwise2`, `Bio.SeqIO`).
