# Synergy rerun + analysis plan

Project-local plan (mirrors `~/.claude/plans/`). See `HANDOFF.md` (cluster state) and
`RESULTS_INVENTORY.md` (where prior results live). This supersedes the earlier 2-bp plan.

## Why we're rerunning

The all-pairs × all-CT run (`full_allct`) was misconfigured (2 h wall vs ~4.3 h/chunk → every chunk
timed out, 0 usable summaries) — cancelled. We're relaunching at **full paper fidelity** so the result
is defensible and matches Betty Liu et al. Nature 2026, where our prior screens had cut corners for speed.

## The run spec — FIRST PASS (fast, expandable)

| parameter | value | notes |
|---|---|---|
| motifs | **73** (all; drop only ST53 "no expression") | no composite/dimer exclusion at run time — filter downstream as a rule |
| pairs | **2,701** = C(73,2)=2,628 heterotypic **+ 73 homodimers (A×A)** | homodimers newly included (paper tests them; 3 orientations) |
| cell types | 22 single-task ChromBPNet models | |
| spacing | **1-bp**, **gaps 0–50** (`gap_step=1 maxdist=50`) | data: 99.9% of synergy optima have gap ≤50bp (median 2, 95th=24, max 84) |
| backgrounds | **n = 32** (first pass) | paper uses 100; expand later. 32 is a fast first look |
| orientation | fixed logic (NRNB code) | same→FF/FR/RF (3), distinct→4, palindrome dedup — matches paper |
| trimming | **stricter — match the CWM logos (~`imp_frac=0.3`)** | paper hand-removes uninformative flanks; verify insert-length ≈ catalog `width` before launch |
| null / calling | in-matrix empirical null + BH q<0.05 | retires the contaminated 42-pair null |

**Compute:** ~4× cheaper than paper-fidelity (n=32 vs 100, gaps 0–50 vs 0–200) → **~150 GPU-hr ≈ ~1 day**
at %6 throttle. `carter-gpu` has infinite wall + no QOS GPU cap, so wall time is unconstrained.

**Expansion path (later, if warranted):** n=32→100, gaps 0–50→0–200. Same manifest, rerun the pairs of
interest (or all) at full fidelity — the first pass tells us which pairs/CTs deserve it.

## Manifest + chunking (the fix for the timeouts)

- **Pair-major manifest:** each pair's 22 CT rows contiguous, so a completed chunk = **fully-profiled
  pairs** (all 22 CT) → we can prototype/analyze incrementally as chunks land.
- Chunk by **whole pairs** (~15–20 pairs × 22 CT per chunk), **12 h wall** (huge margin vs the ~needed
  runtime), `%6` array throttle. Requires a small tweak to chunk on pair boundaries (not flat rows).
- Output → `results/full_allct_1bp/` (new dir); each chunk writes its `summary__*.tsv` on completion.

## Downstream pipeline (prototype on complete MT matrix now; swap in 1-bp when it lands)

1. **Aggregate** → 2,701 × 22 matrices (Δ, maxZ, wilcoxon_p) + `category_A/B` + `involves_composite` flag
   (so composite/dimer pairs can be dropped with one filter, per the "filter later" rule).
2. **Call soft/hard** per CT via in-matrix empirical null (fit on central bulk) + BH q<0.05.
3. **Filter** to pairs soft/hard in ≥1 CT.
4. **Cluster all** on row-z(Δ*) — Ward + silhouette k — fixed-dev-order and clustered columns.
5. **Endocrine (11) vs non-endocrine (11)** split (`cell_type_metadata.tsv` `endocrine` col); recluster
   within each block; tag pairs endocrine- / non-endocrine- / shared.
6. **Figures PNG + PDF** → `syntax_final/figures/`.

## Methods page

Build `syntax_final/1_methodological_overview.html` (style of `syntax_2026_07_26/1_methodological_overview.html`),
explicit about every number above, growing as the run progresses.

## Verification

- Insert-consensus length ≈ catalog `width` (trim matches logos) — check before launch.
- Run completes: 2,701 × 22 rows, no missing pairs/CTs, every chunk has a summary.
- Positive control FOXA2×OTX2 (both Base) comes out synergistic; homodimers behave sanely.
- Endocrine/non-endocrine split visible; every PDF opens and matches its PNG.

## Status
- ✅ Cancelled broken `full_allct`, `orfix` (moot at 1-bp), stale `bash`.
- ✅ Orientation bug already fixed in NRNB code (verified); local mirror stale → sync.
- ⏳ Prep 1-bp submit script (pair-major, 12h wall, n=100, gap_step=1) — **show before launching**.
- ⏳ Prototype pipeline on MT matrix in parallel.
