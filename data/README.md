# `data/`

Small, ready-to-use assets ship in this repository; the large files are hosted on Zenodo
(see *Data Availability* in the main README).

## Shipped in the repository

| Path | What it is |
|---|---|
| `demo_data/Emerson2017_demo/` | A few example repertoire files (TSV) to try `step2` on. |
| `processed_data/primary_k1024/1024_primary_centroids.pk` | The 1,024 reference cluster centroids — repertoires are encoded against these. |
| `processed_data/spectral_mappings/` | Primary → functional cluster maps (`k = 8…1024`; default `k = 128`). |
| `antigen/` | McPAS-derived antigen-specificity validation set (see `antigen/README.md`). |
| `metadata/` | Per-study sample metadata (`<Study>.csv`, plus `datasets_type.csv`). |

## Downloaded from Zenodo

Each archive extracts to a folder whose name differs from the target path, so move its
contents into the subdirectories below (the main README lists the exact `mv` commands):

| Archive | Move its contents to |
|---|---|
| `database.tar.gz` | `MetaTCR_reference_fullseq.txt` → `data/database/`; `reference_embeddings/` → `data/reference_database/` |
| `repertoire_data.tar.gz` | `data/repertoire_data/` |
| `encoding.tar.gz` | `data/processed_data/datasets_mtx_1024_downstream/` |
