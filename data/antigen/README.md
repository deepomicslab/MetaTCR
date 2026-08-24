# Antigen-specificity validation data (McPAS-TCR–derived)

McPAS-TCR-derived records were used only for supporting antigen-specificity validation and the
semi-synthetic EBV spike-in analysis; they were **not** included in construction of the MetaTCR
reference database. For antigen validation, the five most represented epitope groups were
balanced, encoded with TCR2vec and assigned to the current MetaTCR primary clusters. For the EBV
experiment, McPAS records annotated with Epstein–Barr virus pathology formed the sampling pool
for clonotype replacement.

## File

`McPAS_antigen_validation_currentclusters.csv` — **2,164** human TCRβ records, the balanced
antigen-validation set (top-5 epitope groups, ~433 records each), each assigned to its **current**
MetaTCR 1,024-cluster reference (seed-123). Only the columns needed to reproduce the analyses are
kept; the remaining McPAS metadata (MHC, tissue, T-cell type, patient / publication details, etc.)
is not redistributed here.

| column | description |
|---|---|
| `CDR3.beta.aa` | CDR3β amino-acid sequence (from McPAS-TCR) |
| `TRBV` / `TRBJ` | V / J gene (from McPAS-TCR) |
| `full_seq` | full-length TCRβ reconstructed from CDR3 + V + J (TCR2vec input) |
| `Epitope.peptide` | epitope group (one of the five balanced groups) |
| `primary_cluster_id` | assigned MetaTCR primary cluster (0–1023), current seed-123 reference |

Epitope groups: `CRVLCCYVL`, `GILGFVFTL`, `GLCTLVAML`, `LPRRSGAAGA` (433 each), `NLVPMVATV` (432).
`primary_cluster_id` is the only MetaTCR-derived column; all sequence / epitope fields originate
from McPAS-TCR.

## Redistribution & citation

McPAS-TCR provides a full-database download on its official page, requires citation, and notes a
last update of 2022-09-10; the original paper describes it as free access. We could not find an
explicit redistribution license, so **no separate CC-BY license is attached to this file**. If you
reuse it, please obtain McPAS records from — and cite — the official source:

- McPAS-TCR official page: http://friedmanlab.weizmann.ac.il/McPAS-TCR/
- Tickotsky, N., Sagiv, T., Prilusky, J., Shifrut, E., & Friedman, N. (2017). *McPAS-TCR: a
  manually curated catalogue of pathology-associated T cell receptor sequences.* Bioinformatics,
  33(18), 2924–2929.

## Integrity

```
McPAS_antigen_validation_currentclusters.csv  sha256  f679de90ed7a84e1517945709edf420f463d939262054f34d9eca6d194fdc97f
```
