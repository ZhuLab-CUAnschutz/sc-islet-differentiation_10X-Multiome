# TF motif hit calls — sc-islet differentiation, 22 cell types

ChromBPNet + Fi-NeMo TF binding site calls, motif catalog **v1.1** (74 motifs).
Shared 2026-08-17.

## Contents

```
hits/<cell_type>.hits.bed.gz    22 files, coordinate-sorted BED6+1, hg38
motif_catalog.tsv               motif_name <-> ST## <-> TF label (74 rows)
catalog.meme                    the 74 motif PPMs
hit_counts_matrix.tsv           motif x cell type hit counts
hit_counts_long.tsv             same, long format
```

## BED columns

| # | column | notes |
|---|--------|-------|
| 1 | `chrom` | hg38 |
| 2 | `chromStart` | 0-based, trimmed motif core |
| 3 | `chromEnd` | |
| 4 | `name` | motif label, e.g. `ST64_ISL1`, `ST25_FEV`, `ST05_sub1_HNF4A_G` |
| 5 | `score` | Fi-NeMo `hit_importance` (float, not the 0–1000 BED convention) |
| 6 | `strand` | |
| 7 | `motif_name` | original catalog id (`Average_NNN` / `ST##_subN`), for joining back |

Column 4 is the field to filter on. `motif_catalog.tsv` maps it to `short_id`,
TF, curator category and motif width.

## Cell types

`DE`, `PGT1-3`, `PFG1-2`, `PP1-2`, `ENP_phase1`, `early_ENP`, `late_ENP`,
`early_SC_beta`, `late_SC_beta`, `early_SC_alpha`, `late_SC_alpha`,
`early_SC_EC`, `late_SC_EC`, `SC_delta_GHRL`, `proliferating_endocrine`,
`exocrine`, `liver`, `FB_FLT1`.

## How the hits were called

Per cell type, on that cell type's own ChromBPNet **counts-head** contribution
scores over its own peak set:

1. `finemo extract-regions-chrombpnet-h5 -w 400`
2. `finemo call-hits` against all 74 catalog motifs at once, `lambda = 0.8`

The scan is **competitive** — all 74 motifs compete to reconstruct the
contribution track, so a hit is a motif instance that was needed to explain the
observed importance, not a PWM match. Two consequences worth knowing:

- Hit counts are not comparable to a PWM scan; they are much sparser.
- Where two catalog motifs are near-identical, the assignment between them is
  arbitrary. See the caveats below.

## Caveats on the motifs you asked about

**ISL1 appears twice in the catalog.**

| label | what it is | total hits (all CTs) |
|---|---|---|
| `ST36_sub2_ISL1` | clean ISL1 homeodomain, from the manual split of the ST36 cluster | 36,552 |
| `ST64_ISL1` | curator annotation reads "PAX6 (or ISL1)" — ambiguous between the two | 5,738 |

These are different motifs with different hit sets. `ST64_ISL1` is the one the
pairwise-synergy calls fired on; `ST36_sub2_ISL1` is the unambiguous ISL1.
Both are included.

**FEV appears three times:** `ST08_FEV` (1,701 hits), `ST25_FEV` (80,459),
`ST62_FEV` (53,437). They are separate clusters, not duplicates.

**HNF4G and RARB are ~93% identical DR1 motifs.** `ST05_sub1_HNF4A_G` and
`ST05_sub2_RARB` came from splitting one cluster, and the competitive scan
cannot reliably resolve which of the two explains a given site. Treat any
ST05_sub hit as "an HNF4/RAR-type DR1 site", not as HNF4G specifically.

**HNF4G hits are concentrated in liver / foregut, not beta.** In
`early_SC_beta` there are 3 `ST05_sub1_HNF4A_G` hits genome-wide (and 0 for
`ST05_sub2_RARB`); the motif peaks in `liver`. Check
`hit_counts_matrix.tsv` before building an analysis on ST05 in an endocrine
cell type.

## Provenance

Catalog and hit calls:
`results/3_single_task_models/motifs/v1.1/` in
`adamklie/sc-islet-differentiation_10X-Multiome`.
