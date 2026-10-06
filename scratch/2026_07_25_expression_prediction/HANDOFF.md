# Sequence → Expression (Decima-style) — Handoff

**Date:** 2026-07-27 · **Status:** wound down, all jobs cancelled. Zero-shot baseline **done**; Decima-style Borzoi fine-tune **fully built and ready to train**, but not yet run (a30 GPUs were saturated by other jobs).

---

## 1. Goal

Predict **cell-type-specific gene expression directly from DNA sequence** for the islet organoid differentiation dataset, by fine-tuning **Borzoi in the Decima style** (Borzoi trunk + gene-mask channel + avg-pooled scalar-per-cell-state head, Poisson+multinomial loss). The **deliverable is the fine-tuned model** — the zero-shot step is only the baseline it must beat.

Locked decisions (from a /grill-me session): feasibility POC; compare Decima- vs Borzoi-warm-start with **zero-shot Decima as the floor**; **22 pooled cell-type** pseudobulk tracks (summed counts); success = beat the floor on **both** across-gene *and* across-track correlation + marker recovery; reuse **Decima's own Borzoi folds** (train/val/test in the metadata AnnData). Full plan: `~/.claude/plans/to-date-we-have-hashed-quail.md`. Method background: `islet_organoid_differentiation/docs/papers/decima_and_related.md`.

---

## 2. Result so far — zero-shot Decima floor (DONE)

Files in this folder: `scorecard.{csv,png}`, `zeroshot_across_gene.csv`, `zeroshot_across_track.csv`, `zeroshot_track_provenance.csv`, `figures/step1/*` (target QC), `figures/step3/*` (zero-shot).

| arm | across-gene r (median) | across-track r (median) | marker AUROC |
|---|---|---|---|
| **zero-shot Decima** | 0.68 | **0.08** | ~0.05 (noisy) |
| mean-expression baseline | 0.95 | ~0 (NaN by construction) | 0.50 |

**Key finding:** off-the-shelf Decima captures the *general* expression program (across-gene ~0.68) but has **essentially no cell-type specificity** for our states (across-track ~0.08). Root cause, confirmed from Decima's atlas metadata: **Decima collapses all pancreatic islet endocrine cells into one `enteroendocrine cell` label** — so our beta/alpha/EC/delta map to the *same* atlas tracks and can't be distinguished zero-shot (the `figures/step3/track_similarity_heatmap.png` is flat, no diagonal). **This is exactly the gap fine-tuning our 22 resolved states should fill** — the strongest possible motivation for the fine-tune. `figures/step1/markers_heatmap.png` confirms our pseudobulk target is biologically sound (INS→beta, GCG→alpha, TPH1/LMX1A→EC, SST/GHRL→delta).

Zero-shot needs **no GPU** — decima 0.7.2 ships precomputed predictions in `DecimaResult.load().anndata.layers['preds']`.

---

## 3. The fine-tune recipe (decima's own trainer — use this, not hand-rolled)

decima 0.7.2 API (verified on cluster):
- **Model:** `decima.model.decima_model.DecimaModel(n_tasks=22, mask=True, init_borzoi=True, replicate=0)` — Borzoi trunk + gene-mask 5th channel + `ConvHead(n_tasks=22, in_channels=1920, pool_func="avg")`. `init_borzoi=True` pulls Borzoi weights via **wandb** (`grelu/borzoi/human_state_dict_fold{replicate}`, ~709 MB) — **works** (wandb already logged in as `adamklie`). ~171 M params.
- **Trainer:** `decima.model.lightning.LightningModel(name, model_params, train_params, data_params).train_on_dataset(train_ds, val_ds)` — builds a Lightning `Trainer` (accelerator="gpu", precision="16-mixed", `ModelCheckpoint(monitor="val_loss", save_top_k=1)`), full-param Adam. `default_train_params = {lr:4e-5, batch_size:4, max_epochs:15, total_weight:1e-4, disease_weight:0.01, accumulate_grad_batches:1, clip:0.0, ...}`.
- **Data:** `decima.data.write_hdf5(file, ad, genome="hg38")` → HDF5 with `sequences/masks/labels/tasks`, from an AnnData whose `.var` has `chrom/start/end/strand/gene_mask_start/gene_mask_end/dataset` and `.X` = labels, `.obs.index` = task names. Then `decima.data.dataset.HDF5Dataset(key, h5_file, ad=ad_train)` with `key` in {"train","val","test"}.
- **Loss:** `decima.model.loss.TaskWisePoissonMultinomialLoss`.

**Our training AnnData** = our 22 pseudobulk tracks + summed counts grafted onto **Decima's gene coordinates/mask/split** (Decima's metadata AnnData `.var` already has everything). Built by `build_train_data.py`.

---

## 4. Exact cluster state (NRNB / carter)

- **Env:** conda `decima` (`/carter/users/aklie/opt/miniconda3/envs/decima`) = clone of `gRelu` + `pip install decima peft tangermeme scanpy pysam`. **torch `2.13.0+cu126`** (NOT the cu130 that decima pulls by default — cu130 needs a CUDA-13 driver; the a30 driver is 565.57 = CUDA 12.7). `setuptools` pinned `<80` (newer drops `pkg_resources`, breaking decima import). decima 0.7.2, grelu 1.1.0.
- **Data (all under** `/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/`**):**
  - `results/2_process_data/pseudobulk_expression.h5ad` — 22 tracks × 36,601 genes, summed counts (built by `bin/2_process_data/build_pseudobulk_expression.py`).
  - `results/5_expression_models/train_anndata.h5ad` — 22 × 18,391, our counts + Decima gene coords/split.
  - `results/5_expression_models/islet_decima.h5` — **38 GB training HDF5** (sequences+masks+labels+tasks, genome="hg38"). Expensive to rebuild; keep unless you need disk.
  - `results/5_expression_models/scorecard.csv` + `figures/` — zero-shot results.
- **Scripts:** in `~/` on cluster: `build_train_data.py`, `finetune.py`, `finetune_job.sh`, `run_decima_zeroshot.py`, etc. Also committed in repo `sc-islet-differentiation_10X-Multiome/bin/5_expression_models/scripts/` (note: repo copies of the *training* scripts predate the API rewrite — the authoritative training scripts are the `~/` copies; sync them into the repo before relying on it).
- **HF cache:** `/carter/users/aklie/data/decima_cache` (Decima weights + metadata AnnData, ~GBs).
- **Jobs:** all mine cancelled. Nothing running.

---

## 5. How to resume the fine-tune

```bash
# on an a30 (rtx5000 GPUs are Turing → unusable with torch 2.13, "CUDA unknown error")
conda activate decima          # torch cu126 env, ready
sbatch ~/finetune_job.sh        # full-data, 6 epochs, 1 a30, batch1 + accumulate8, 12h limit
                                # checkpoints per val improvement -> partial run still usable
squeue -u aklie -n islet_finetune
tail -f /carter/users/aklie/finetune_<jobid>.log   # look for "CUDA OK", trainable params, val_loss
```

`finetune.py` fail-fasts if CUDA isn't available (catches the a30-quirk case in seconds). It also takes `--max_train_genes 300 --max_epochs 2` for a fast **smoke run** to validate the pipeline end-to-end before the long run — recommended as the first thing to try.

**Reality:** full fine-tune = ~14.5k train genes × 524 kb context at batch 1 on one a30 ≈ **GPU-hours per epoch**. Use the smoke run first; consider multi-GPU DDP on one a30 node (`--devices 4`) or fewer epochs to keep it tractable.

---

## 6. Evaluation (once a checkpoint exists)

Predict our 22 tracks for the held-out **test** genes with the fine-tuned model, compute across-gene + across-track correlation + marker recovery (reuse `bin/5_expression_models/scripts/_utils.py` — `summarize`, `update_scorecard`), and add an `islet_ft` row to `scorecard.csv` next to `zeroshot_decima` and `mean_expression`. **Win = beat the floor on across-track (>0.08) and marker recovery.** (`evaluate_model.py` in the repo is the scaffold but was written against the hand-rolled model; adapt it to load the decima `LightningModel.load_from_checkpoint` + predict via `forward(gene_list)`.)

---

## 7. Gotchas / lessons (don't relearn these)

1. **torch/CUDA:** decima pulls torch cu130 (needs CUDA-13 driver); a30 driver is 12.7 → must repin `torch==2.13.0+cu126` **with `--extra-index-url https://pypi.org/simple`** (else pip backtracks forever). Re-pin `setuptools<80` afterward (torch bumps it and drops `pkg_resources`).
2. **rtx5000 GPUs unusable** with torch 2.13 (Turing sm_75 → "CUDA unknown error"). Use **a30** only. (8 idle rtx5000s are a tease — would need an older torch env, e.g. 2.7+cu126, to use them.)
3. **`/cellar` vs `/carter`:** compute nodes mount `/carter`, not `/cellar` (same storage). Use `/carter/...` paths in scripts. Also `projects/islet_organoid_differentiation/sc-islet...` is a symlink to `data/datasets/sc-islet...`.
4. **Genome for `write_hdf5`:** use `genome="hg38"` (genomepy), **not** the raw fasta path (the fasta path produced "Input string is not a valid DNA sequence" on some genes).
5. **SSH:** rapid short-lived ssh connections trip the login node's sshd **MaxStartups throttling** (connections hang → look like an outage). Use **one persistent `ControlMaster`** connection and multiplex.
6. **SLURM output:** submit with `sbatch` + `-o` log file and poll the file; interactive `srun` piped through ssh loses output under contention and orphans allocations.

---

## 8. Quick file index

- **This folder** (`~/Desktop/ucsd/expression_prediction_2026_07_25/`): zero-shot scorecard + figures + this handoff.
- **Plan:** `~/.claude/plans/to-date-we-have-hashed-quail.md`
- **Method notes:** `.../islet_organoid_differentiation/docs/papers/decima_and_related.md` (+ `Decima_NatMethods_2026/` PDFs)
- **Code stage:** `.../sc-islet-differentiation_10X-Multiome/bin/5_expression_models/` (README, RUNBOOK, scripts)
- **Project memory:** `~/.claude/projects/-Users-adamklie-Desktop-ucsd/memory/seq2expression-decima-poc.md`
