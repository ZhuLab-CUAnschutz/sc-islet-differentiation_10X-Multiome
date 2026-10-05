# 5_expression_models — sequence → expression (Decima recipe)

Predict **cell-type-specific gene expression directly from DNA sequence** by adapting
the **Decima** recipe (Borzoi trunk, 524 kb context, gene-mask channel, average-pooled
scalar-per-cell-state head, Poisson+multinomial loss) to the islet organoid
differentiation dataset. This is a **feasibility POC**: can the recipe learn our
cell-type-specific expression well enough to beat honest baselines?

See `docs/papers/decima_and_related.md` for the method background and the plan file
`~/.claude/plans/to-date-we-have-hashed-quail.md` for the full design + decisions.

## Locked decisions

| Fork | Decision |
|---|---|
| Goal | Feasibility POC — beat honest baselines, then decide whether to invest further |
| Base model | Compare **Decima-warm-start** vs **Borzoi-warm-start**; **zero-shot Decima** = floor |
| Targets | **22 cell types, replicates pooled** (one summed-counts pseudobulk track each) |
| Success bar | **Both** across-gene *and* across-track correlation, **plus** marker recovery |
| FT regime | **Head-only probe** first on both arms → escalate the winner only if it shows signal |
| Gene split | Reuse **Decima's Borzoi folds**; evaluate all arms on Decima's held-out **test** genes |

## Status

All six steps **scaffolded** (code written, syntax-checked). Not yet run — needs the
env built + the `# VERIFY:` decima/grelu API spots confirmed on the cluster.

- [x] Step 1 — pseudobulk target + QC figures
- [x] Step 2 — gene windows + gene-mask + Decima fold assignment
- [x] Step 3 — zero-shot Decima baseline + figures + scorecard
- [x] Step 4 — head-only probes (Decima vs Borzoi warm-start)
- [x] Step 5 — escalate the winner (LoRA / full FT) — same entrypoint, `--regime`
- [x] Step 6 — attribution smoke test

## Note from the Decima paper (bears on the probe → escalate regime)

Decima's own ablation (Nat Methods, Supp. Table 2) found that **freezing the Borzoi
trunk and training only the head performed poorly** (as did random init) — full
fine-tuning from the pretrained foundation was essential. Implication for our
head-only probes:

- **Borzoi-arm probe** is expected to be *weak* — it's the exact frozen-Borzoi setup
  Decima found underperforms. Keep it as the honest cheap-rung diagnostic, but don't
  read a weak result as "the recipe fails."
- **Decima-arm probe** is the promising cheap rung — its trunk is *already
  expression-trained*, so a frozen Decima trunk + small 22-head can plausibly work
  where frozen Borzoi doesn't.
- Plan the escalation as **LoRA or full fine-tune**; on A30 24 GB, LoRA is the
  realistic path to a full-trunk update. Reference: per-pseudobulk r=0.80 /
  per-gene r=0.58 on 1,811 test genes (the bars our arms are measured against).

## Environment

Decima/gReLU is **PyTorch** — a *separate* env from the CREsted `.venv-keras`.
Create it once on the cluster:

```bash
bash scripts/setup/create_env.sh      # builds the grelu/decima env
```

## Layout

```
scripts/
  setup/      create_env.sh                # one-time env build (cluster)
  target/     qc_pseudobulk_target.py       # Step 1 figures (runnable, no decima dep)
  preprocess/ build_windows_and_folds.py    # Step 2 windows + gene-mask + folds (decima API)
  zeroshot/   state_mapping.tsv             # our 22 states -> candidate Decima atlas tracks (CURATE)
              run_decima_zeroshot.py        # Step 3 predict + metrics + figures + scorecard
              decima_zeroshot.sh            # SLURM GPU wrapper
  train/      data.py                       # gene-window + gene-mask + 22-target DataModule
              model.py                      # SeqExpr: {decima,borzoi} x {probe,lora,full}
              train.py  train.sh            # Steps 4-5 entrypoint (--arm --regime) + SLURM
  evaluate/   evaluate_model.py             # test-gene metrics/figures + scorecard row
  attribution/attribute.py  attribute.sh    # Step 6 specificity attribution smoke test
  _utils.py                                 # shared metrics + plotting + scorecard
```

Steps 4-5 share one entrypoint. The compare-two-arms POC is four training runs:
```
python train/train.py --arm decima --regime probe   # promising cheap rung
python train/train.py --arm borzoi --regime probe   # expected weak (frozen-Borzoi ablation)
# escalate the better arm:
python train/train.py --arm <winner> --regime lora   # A30-friendly full-trunk update
```
Evaluate each with `evaluate/evaluate_model.py --ckpt <best> --arm <a> --regime <r>`
(appends its row to the shared scorecard). Attribution runs only if metrics pass.

The all-gene summed-counts **target** is built one stage up by
`bin/2_process_data/build_pseudobulk_expression.py` (generalizes the TF-subset script)
→ `results/2_process_data/pseudobulk_expression.h5ad`.

## Run order (all on carter-gpu unless noted)

```bash
# Step 1 — build target (CPU, in 2_process_data) then QC figures
python ../2_process_data/build_pseudobulk_expression.py
python scripts/target/qc_pseudobulk_target.py

# Step 2 — windows + gene-mask + Decima folds (CPU/GPU)
python scripts/preprocess/build_windows_and_folds.py

# Step 3 — zero-shot Decima baseline (GPU)
#   1) curate scripts/zeroshot/state_mapping.tsv first
gpu.sh -s scripts/zeroshot/decima_zeroshot.sh -j decima_zeroshot -m 64G -t 04:00:00 -c 4 -g a30:1
```

## Outputs

Figures → `results/5_expression_models/figures/<step>/`.
The running **scorecard** (arm × {across-gene r, across-track r, marker AUROC} vs
baselines) → `results/5_expression_models/scorecard.{csv,png}`, appended to at each step.

## ⚠️ Verify against the installed decima version

The Step 2–3 scripts call the `decima` / `grelu` APIs (model loading, `aggregate_anndata`,
`assign_borzoi_folds`, `VariantDataset`, `ConvHead`, attribution). Function names/signatures
were read from the Decima source but **must be checked against the version pip-installed on
the cluster** — spots that need confirming are marked `# VERIFY:` in the scripts.
