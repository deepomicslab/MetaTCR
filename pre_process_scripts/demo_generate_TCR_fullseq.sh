for dir in ./raw_repertoire_data_path/dataset_name/; do
  for file in "$dir"*.tsv; do
    output_dir="$(basename "$dir")/$(basename "$file")"
    python cdr3_to_full_seq_mod_mp.py ./TCR_gene_segment_data/ $file vMaxResolved jMaxResolved aminoAcid "${output_dir%.tsv}_full.tsv"
  done
done

