# DE 4-TF synergy — EOMES / MIXL1 / SOX17 / FOXH1 (2026-09-06)

Direct pairwise-synergy test of the four DE TFs Han sent, in the DE ChromBPNet model only, at
full paper fidelity. Two insert sets: (1) catalog v1.1 CWMs for EOMES/MIXL1/SOX17 + JASPAR
FOXH1; (2) all four from JASPAR. Report: `REPORT.md` / `report.html`.

Cluster copy: `narrows-login.sdsc.edu:/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_09_06_de_synergy/`
(GPU job 1047148, carter-gpu, one A30). Local mirror: `sandbox/2026_09_06_de_synergy/`.

## Layout
```
inputs/   jaspar/*.jaspar (raw PFMs), motifs_de4.npz (7 inserts), pairs_de4.tsv (19-row run
          manifest), pairs_by_set.tsv (20 rows = set x pair), inserts.tsv, null_DE_summary.tsv
          (42-pair DE lineage-mismatched null, copied from 2026_07_19_syntax_pairwise_marg/results/null)
scripts/  build_inputs.py -> preflight.py -> run_de4.sh (SLURM) -> analyze_de4.py -> plot_de4.py
          marginalize_pairs.py = verbatim copy of the 2026_07_19 core script (md5 bd55b873...)
          plot_logos.py [--match_orientation]  (fig0 / fig0b)
results/de4/  single-task summary__DE.tsv + one .npz grid per pair
results/mt_de4/ and mt_null_full/  multitask 19×22 target and 42×22 null summaries
tables/   single-task tables plus mt_calls_long.tsv, mt_pairs_de.tsv, mt_null_thresholds.txt
figures/  fig0_insert_logos, fig0b_insert_logos_oriented, fig1_delta_by_pair, fig2_distance_curves,
          fig3_inserts, fig4_homodimers, fig5_multitask_delta_by_pair  (PNG + PDF)
```

## Motifs

| TF | catalog v1.1 | catalog insert | JASPAR | JASPAR insert (IC ≥ 0.2 bit trim) | strand vs catalog |
|---|---|---|---|---|---|
| EOMES | ST18 (Base, w=8) | `TTCACACC` | [MA0800.1](https://jaspar.elixir.no/matrix/MA0800.1/) EOMES, HT-SELEX, 13 bp | `AAGGTGTGAA` (10 bp) | reverse complement (`TTCACACCTT`) |
| MIXL1 | ST36_sub1 (Base, w=12) | `GGTGCTAATCAG` | [MA0662.1](https://jaspar.elixir.no/matrix/MA0662.1/) MIXL1, HT-SELEX, 10 bp | `TAATTA` (6 bp) | palindromic core, same either way |
| SOX17 | ST66 (Partial, half of obligate homodimer, w=7) | `AGAATGC` | [MA0078.1](https://jaspar.elixir.no/matrix/MA0078.1/) Sox17 (mouse), SELEX, 9 bp | `CATTGTC` (7 bp) | reverse complement (`GACAATG`) |
| FOXH1 | none | — | [MA0479.1](https://jaspar.elixir.no/matrix/MA0479.1/) FOXH1, ChIP-seq, 11 bp | `CAATCCACA` (9 bp) | n/a (used in both sets) |

- JASPAR PFMs were fetched from `https://jaspar.elixir.no/api/v1/matrix/<ID>/?format=jaspar` and kept verbatim in `inputs/jaspar/`.
- Trim rule: keep the span from the first to the last position with information content ≥ 0.2 bits, then argmax per position (the single-motif PPM-trim convention). MIXL1 loses its low-IC flanks and is left with the 6-bp homeodomain core `TAATTA`; flagged as a caveat in the report.
- The catalog CWM inserts are produced by the same `cwm_consensus_ohe` (argmax of the signed CWM, flanks below 10 % of peak importance dropped) as every prior synergy run, so their widths (8/12/7) match `calls_long.tsv` from the first pass.
- Strand is irrelevant to the run itself (all orientations are swept); `fig0b_insert_logos_oriented` flips the JASPAR logo to the strand that best matches the catalog consensus (ungapped alignment of consensus strings, `plot_logos.py --match_orientation`) purely for visual comparison. `fig0_insert_logos` shows the inserts as stored.

## Run parameters (identical to the hero-pair run and to the DE null)
| parameter | value |
|---|---|
| model | single-task ChromBPNet, DE, fold_0, `chrombpnet_nobias.h5` |
| backgrounds | n=100, GC-matched inaccessible (`filtered.nonpeaks.bed` of the DE model), 2114 bp, seed 1234 |
| readout | predicted ln counts; ΔA, ΔB, ΔJ vs background; ΔS = ΔA + ΔB |
| catalog inserts | argmax of signed CWM (`cwms.npz` v1.1), trimmed at `imp_frac=0.1`: ST18 `TTCACACC`, ST36_sub1 `GGTGCTAATCAG`, ST66 `AGAATGC` |
| JASPAR inserts | PFM trimmed to IC ≥ 0.2 bit, argmax: MA0800.1 EOMES `AAGGTGTGAA`, MA0662.1 MIXL1 `TAATTA`, MA0078.1 SOX17 `CATTGTC`, MA0479.1 FOXH1 `CAATCCACA` |
| gaps | inner-edge 0–200 bp, 1-bp uniform grid |
| orientations | FF/FR/RF/RR (distinct), FF/FR/RF (homodimer); palindrome dedup (none here) |
| optimum | argmax mean ΔJ over arrangements; Δ = ΔJ(opt) − ΔS |
| stats | Wilcoxon signed-rank joint vs additive per background; maxZ of optimum vs all arrangements; bootstrap (200) of the hard call; soft = Δ>0.15 at 20–150 bp; frac_pos |
| calling | Δ > max(0.15, DE-null p95) & p<1e-3 → synergistic; & maxZ > max(4, DE-null z95) → hard |
| pairs | 6 heterotypic + 4 homodimers per set; 19 unique (FOXH1×FOXH1 shared) |
| compute | `--partition=carter-gpu --account=carter-gpu --gres=gpu:a30:1`, 4 CPU, 32 GB, batch 256 |

## Reproduce
```
python scripts/build_inputs.py                       # local: JASPAR trim + npz + manifests
rsync -az --exclude .DS_Store --exclude __pycache__ . narrows:$R/
ssh narrows 'bash -lc "cd $R && $PY scripts/preflight.py && sbatch scripts/run_de4.sh"'
rsync -az narrows:$R/results/ results/               # then, locally:
python scripts/analyze_de4.py && python scripts/plot_de4.py && python scripts/plot_logos.py [--match_orientation]
```

## Multitask replication

Job 1047592 reran the 19 target pairs and 42 null pairs through CRESTED multitask checkpoint
`finetuned_v2/checkpoints/28.keras`, scoring all 22 heads and reporting DE. Both used n=100,
gaps 0–200 by 1 bp, all orientations, seed 1234. Unlike the single-task run, this established
pipeline uses shared dinucleotide-shuffled peak backgrounds and emits raw predicted counts.
Accordingly, multitask calls use the separately rerun multitask DE null (Δ p95 = 161.37 counts;
maxZ p95 = 6.38), and cross-model figures compare Δ / matched-null p95 rather than raw Δ.

Post-flight: target 418/418 rows, null 924/924 rows, exactly 22 heads per pair, no duplicate
pair-head rows or non-finite numeric values, independently reproduced percentiles, and matching
SHA-256 checksums between the cluster and local summaries.
