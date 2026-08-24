#!/usr/bin/env bash
# Demo: reconstruct full-length TCRbeta sequences from a filt-style TSV that has
# only CDR3 + V gene + J gene. It adds a `full_seq` column (what the MetaTCR
# repertoire encoding expects). Run from anywhere:  bash demo_generate_TCR_fullseq.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

python "$HERE/cdr3_to_full_seq.py" "$HERE/demo_filt_input.tsv" \
    --segment-dir "$HERE/TCR_gene_segment_data" \
    --v-header vMaxResolved --j-header jMaxResolved --cdr3-header aminoAcid \
    --out-dir "$HERE" --in-suffix .tsv --out-suffix _with_fullseq.tsv \
    --procs 1

echo "wrote -> $HERE/demo_filt_input_with_fullseq.tsv  (input columns + a new 'full_seq')"
