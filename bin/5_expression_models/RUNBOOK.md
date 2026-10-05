# RUNBOOK — running the seq→expression POC on NRNB (carter)

Concrete cluster steps to take the scaffold in this stage from "syntax-checks" to real
numbers. Never compute on the login node — everything below runs via `srun`/`sbatch` on
`carter-compute` / `carter-gpu`.

## Cluster facts (verified 2026-07-25)

| Thing | Value |
|---|---|
| Data (RNA) | `/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/rna_matrix.h5ad` (2.7 GB) |
| Genome | `/cellar/users/aklie/data/ref/genomes/hg38/hg38.fa` (3 GB) |
| Repo on cluster | `/cellar/users/aklie/projects/islet_organoid_differentiation` (== `/carter/...`) |
| Base env | conda **`gRelu`** (`/carter/users/aklie/opt/miniconda3/envs/gRelu`) — gReLU 1.0, torch 2.0.0+cu118, lightning 2.2.5, captum, anndata, modisco-lite |
| Missing pkgs | **decima, peft, tangermeme** (+ maybe scanpy/pysam) |
| GPU | `carter-gpu`: 2× nodes with `a30:4`, 1× node with `rtx5000:8` |

## 0. One-time setup

**a) Get this code onto the cluster.** Commit locally and `git pull` on the cluster, or
rsync the stage over:
```
rsync -av --exclude __pycache__ --exclude .DS_Store \
  bin/5_expression_models/ \
  hpc:/cellar/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/bin/5_expression_models/
# and the new target builder:
rsync -av bin/2_process_data/build_pseudobulk_expression.py \
  hpc:/cellar/.../bin/2_process_data/
```

**b) Env.** Decima isn't in `gRelu`. **Clone it** (don't mutate the shared `gRelu` env —
other work depends on it), then add the missing pieces:
```
srun -p carter-compute --mem 16G -c 4 --pty bash
conda create -n decima --clone gRelu          # ~minutes
conda activate decima
pip install decima peft tangermeme scanpy pysam
python -c "import decima, grelu, peft, torch; print(decima.__version__, torch.__version__)"
```
⚠️ `gRelu`'s torch is 2.0.0+cu118 (older). If `pip install decima` tries to pull a newer
torch, either pin torch or build a fresh env from `scripts/setup/create_env.sh` instead.

**c) Point the SLURM wrappers at the env.** In `scripts/*/*.sh`, replace the
`source .../.venv-torch/bin/activate` line with:
```
source /carter/users/aklie/opt/miniconda3/etc/profile.d/conda.sh && conda activate decima
```

## 1. Resolve the `# VERIFY:` spots (do this FIRST, interactively)

Decima/grelu API names in the scripts were read from source, not run. Confirm them once
in the `decima` env (light imports, on a compute node):
```
srun -p carter-compute --mem 16G -c 4 --pty bash ; conda activate decima
python
>>> import decima, grelu, inspect
>>> [x for x in dir(decima)]                     # model loader name?
>>> import decima.preprocess as pp; print(dir(pp))# extract_gene_input / load_gene_metadata?
>>> from decima.model.loss import TaskWisePoissonMultinomialLoss   # loss path?
```
Checklist of spots to confirm/patch (grep `# VERIFY` in scripts):
- `train/data.py`: `extract_gene_input`, `load_gene_metadata`
- `train/model.py`: `load_decima_model`, trunk attribute path, `ConvHead`, `BorzoiModel` loader, stem-widening, `TaskWisePoissonMultinomialLoss`, LoRA target-module names
- `zeroshot/run_decima_zeroshot.py`: `load_decima_model`, `load_track_metadata`, `predict_genes`
- `preprocess/build_windows_and_folds.py`: `load_gene_metadata`, fold-label values
- `attribution/attribute.py`: `attribute_gene`, `scan_sequences`
- fold labels `TRAIN/VAL/TEST_FOLDS` in `data.py` and `run_decima_zeroshot.py`

## 2. Step 1 — target (CPU)
```
sbatch/gpu.sh -p carter-compute --mem 64G -c 8 :  python ../2_process_data/build_pseudobulk_expression.py
                                                   python scripts/target/qc_pseudobulk_target.py
```

## 3. Step 2 — folds (CPU)
```
python scripts/preprocess/build_windows_and_folds.py
```

## 4. Curate `scripts/zeroshot/state_mapping.tsv`
Print Decima's real `cell_type` values, then edit the mapping:
```
python -c "import pandas as pd; from <decima track meta loader> import *; print(sorted(set(meta.cell_type)))"
```

## 5. Step 3 — zero-shot floor (GPU, a30)
```
gpu.sh -s scripts/zeroshot/decima_zeroshot.sh -j decima_zeroshot -m 64G -t 04:00:00 -c 4 -g a30:1
cat results/5_expression_models/scorecard.csv   # zeroshot_decima + mean_expression rows
```

## 6. Steps 4–5 — probe arms → escalate (GPU, a30)
```
gpu.sh -s scripts/train/train.sh -j decima_probe -g a30:1 -- --arm decima --regime probe
gpu.sh -s scripts/train/train.sh -j borzoi_probe -g a30:1 -- --arm borzoi --regime probe
# evaluate each -> scorecard, then escalate the winner:
gpu.sh -s scripts/train/train.sh -j decima_lora  -g a30:1 -- --arm decima --regime lora
python scripts/evaluate/evaluate_model.py --ckpt <best> --arm decima --regime lora
```

## 7. Step 6 — attribution (only if metrics pass)
```
gpu.sh -s scripts/attribution/attribute.sh -j decima_attr -g a30:1 -- --ckpt <best> --arm decima --regime lora
```

## Gate
Zero-shot floor (Step 5 → scorecard) is the decision point: if off-the-shelf Decima
already tracks our pseudobulks, great baseline; the probes/escalation then have to beat it
on **both** across-gene and across-track. Don't run Steps 6 until a trained arm clears the bar.
