# sox32 enhancer — cross-cell-type ChromBPNet analysis

Does the per-cell-type ChromBPNet ensemble reproduce the **sox32** (zebrafish SOX17
analog) enhancer reporter readout (ON in DE → OFF in PGT), and does it recover the
collaborator-reported **EOMES / T (TBXT) / FOXH1** binding?

## Construct (`sox32_enh.fa`, 2114 bp = ChromBPNet input window)
| feature | coords | evidence |
|---|---|---|
| 5′ flank / vector | 0–507 | GC-rich |
| **sox32 enhancer** | **507–1607** | central ~1.1 kb, AT-rich |
| minimal promoter | 1607–1647 | TATA `TATATAA` @1614 |
| Kozak / eGFP CDS | 1641 / 1647–2114 | `GCCACC` + `ATGGTGAGC…` |

All feature coords, paths, motif DBs, and colors live in **`config.py`** (single
source of truth). The enhancer stays at native coords (507–1607) in every variant,
so motif/feature positions are shared throughout.

## Modules
- `config.py` — paths, lengths, construct features (`feature_of`), DBs, TF colors, mpl setup
- `chrombpnet_locus.py` — load models, predict (strand-avg counts+profile), DeepLIFT/SHAP
  contributions, sequence ops (`ohe`, `dinuc_shuffle`, `place_at_offset`, `keep_center_shuffle_flanks`)
- `plot_locus.py` — `LocusResults`, seqlet calling/annotation, contribution matrices, all plotting
- `fimo.py` — MEME-suite FIMO build/run/parse (feature-annotated)

## Pipeline (run under `eugene_tools`)
| step | what | compute |
|---|---|---|
| `01_predict.py/.sh` | predict + count contributions → `outputs/<tag>.npz` (variants: `construct`, `enh_dinuc`) | **GPU** |
| `02_panels.py` | accessibility bar + contribution panels (native Vierstra seqlet annotation) + seqlet table (feature-tagged) | CPU |
| `03_fimo_scan.py` | FIMO EOMES/T/FOXH1 scan + hit-map vs features | CPU |
| `04_motif_relationships.py` | 3 TF logos + nearest Vierstra archetype + browsable cluster/family HTML | CPU |
| `05_contrib_trajectory.py` | per-FIMO-hit contribution across the trajectory (abs + z-score + lines + construct-vs-enh_dinuc) | CPU |
| `06_context_sweep.py/.sh` | context sweep, dinuc-shuffled non-kept bits: `--mode slide` or `keepcenter` | **GPU** |
| `07_construct_map.py` | annotated construct map (features + GC% + FIMO sites) | CPU |

`run_all.sh` documents/drives the order. GPU steps are `sbatch`ed (prefer
`--gres=gpu:rtx5000:1` when the a30 queue is busy); CPU steps run on the head node.

## Variants
- **construct** — the 2114 bp window as-is.
- **enh_dinuc** — keep the central enhancer (507–1607), dinucleotide-shuffle the
  flanking vector/promoter/reporter (the native-like read; GC-rich reporter context removed).
