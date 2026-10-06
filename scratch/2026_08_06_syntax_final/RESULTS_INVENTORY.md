# NRNB results inventory — synergy work

Where everything computed so far lives. Sandbox root (all paths relative to it):

    carter:/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg/

SLURM note: always `ssh nrnb-login.ucsd.edu 'bash -lc "sbatch ..."'` (login shell has the SLURM
binaries); never compute on the login node.

## results/ — marginalization outputs (npz grids + summary__*.tsv)

| dir | what it is | pairs × CT | status |
|---|---|---|---|
| `results/full` | **primary-CT** all-pairs screen (2211 keep-set pairs, each at its one primary CT) | 2211 × 1 | ✅ complete (44 summaries) — **this + `subs` gave the 582** |
| `results/subs` | the 6 sub-motifs' primary-CT pairs (added after keep-set) | 417 × 1 | ✅ complete (12 summaries) |
| `results/syn_allct` | the 582 synergistic pairs re-optimized across **all 22 CT** | 582 × 22 | ✅ complete (66 summaries) — the per-CT profiles |
| `results/null` | **global** null: 42 lineage-mismatched pairs × 3 CT (DE, early_SC_beta, early_SC_EC) | 42 × 3 | ✅ complete → Δ>0.504, Z>5.38 |
| `results/null_allct` | **per-CT** null: same 42 pairs × all 22 CT | 42 × 22 | ✅ complete (22 summaries) → per-CT d95/z95 |
| `results/mt` | **multitask** (CRESTED) all-pairs × all 22 CT, one forward pass | 2628 × 22 | ✅ complete (summaries only, **no grids saved**) |
| `results/mt_null` | MT per-CT null | 42 × 22 | ✅ complete |
| `results/candidate` | 22 **hero** pairs × 22 CT, full fidelity (n=100, **1-bp**), incl homodimers | 22 × 22 | ✅ complete |
| `results/deeplift` | DeepLIFT confirmation of the 582 (frac-in / concentration) | 582 | ✅ complete |
| `results/ctspec` | cell-type-specificity intermediate | — | ✅ |
| `results/full_allct` | **BROKEN** 2-bp all-pairs × all-CT run — every chunk TIMED OUT at 2h, partial npz, **0 summaries** | 2628 × 22 | ❌ cancelled 2026-07-27, superseded by the 1-bp rerun |
| `results/{bench,correct,smoke,mt_bench,mt_smoke,dl_example,dl_sweep}` | benchmark / smoke / attribution-sweep scratch | — | scratch |

## inputs/ — pair manifests

- `pairs_full_sorted.tsv` (2211) / `pairs_subs_sorted.tsv` (417) — primary-CT screen
- `pairs_syn_allCT.tsv` (12804 = 582×22) — the syn_allct profiles
- `pairs_null.tsv` (42) — the lineage-mismatched null grid (7×6); `pairs_null_allct_sorted.tsv` (42×22)
- `pairs_allpairs_allct_sorted.tsv` (57816 = 2628×22) — the broken full_allct manifest (CT-sorted)
- `pairs_candidate.tsv`, `pairs_orfix_allct.tsv` (91-pair orientation-fix, now moot), smoke manifests

## key derived tables / figures

- `figures/` — all PNGs + `synergistic_pairs_allCT_long.tsv` (582 × 22 master table)
- `specificity/` — per-CT null thresholds, specificity_by_ct_long, specificity_scores, cross_model_st_vs_mt
- `mt/` — `mt_calls_long.tsv` (2628×22 MT matrix), mt_pair_summary, mt per-CT null
- `per_ct_tables/`, `per_ct_tables_mt/` — per-CT synergy tables (single-task 582 / MT all-pairs)

## motif catalog (source of CWMs)

    carter:/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/motifs/v1.1/
      cwm/cwms.npz        collapsed signed CWMs (what marginalization inserts, argmax consensus)
      metadata.tsv        74 motifs, curator_tf / curator_category (Base / *composite / obligate homodimer)
      catalog.meme        the PPMs (IC logos)

Local copy of catalog metadata: `syntax_2026-07-25/catalog_v1.1_metadata.tsv`.

## THE RERUN (planned, not yet launched) — 1-bp all-pairs × all-CT

- **Set:** all 73 motifs (drop only ST53). **2,701 pairs** = 2,628 heterotypic + **73 homodimers (A×A)**.
- **Resolution:** 1-bp (`gap_step=1`, gaps 0–200), n=64 backgrounds.
- **Trimming:** stricter — insert-consensus trim raised to match the CWM logos (see HANDOFF item 1).
- **Manifest:** pair-major (each pair's 22 CT rows contiguous) so completed chunks = fully-profiled pairs.
- **Chunking/wall:** sized to finish inside a 12h wall (the old run died at a 2h wall); %6 throttle (good-citizen GPU use).
- **Motif exclusion:** none at run time — composite/dimer filtering applied downstream as a rule.
- Output will land in `results/full_allct_1bp/` (new), aggregated to a 2701 × 22 matrix.
