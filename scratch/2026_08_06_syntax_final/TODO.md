# Synergy rerun — TODO

Living checklist. See `PLAN.md` (full spec), `HANDOFF.md` (cluster state), `RESULTS_INVENTORY.md` (prior results).

## Locked decisions
- [x] **Motif set:** all 73 (drop only ST53). No composite/dimer exclusion at run time — filter downstream.
- [x] **Pairs:** 2,701 = 2,628 heterotypic + **73 homodimers** (A×A newly included).
- [x] **Resolution:** 1-bp, **gaps 0–50** (data: 99.9% of synergy optima ≤50bp gap; max 84).
- [x] **Backgrounds:** **n=32** first pass (paper uses 100; expand later).
- [x] **Orientation:** fixed logic confirmed on NRNB (same→FF/FR/RF, distinct→4). Local mirror stale → sync.
- [x] **Trimming:** verified insert consensus == report CWM logo (cwms.npz pre-trimmed @30%). No change needed.
- [x] **Calling:** in-matrix empirical null + BH q<0.05 (downstream).
- [x] **Chunking:** pair-major (complete pairs per chunk → incremental prototyping). 12h wall, %6 throttle.

## Done
- [x] Cancelled broken `full_allct` (2-bp, timed out), `orfix` (moot at 1-bp), stale `bash`, 5× `sw_*`.
- [x] Wrote HANDOFF.md, PLAN.md, RESULTS_INVENTORY.md, this TODO.md.
- [x] Verified 50bp cutoff, orientation fix, trimming match, motif/CT counts.

## First pass — to launch
- [ ] Build **pair-major manifest** `pairs_fp_pairmajor.tsv` (2,701 pairs × 22 CT, homodimers included).
- [ ] Split into pair-boundary chunk files (~30 pairs each) → `inputs/fp_chunks/`.
- [ ] Write submit script `run_firstpass.sh` (n=32, maxdist=50, gap_step=1, 12h wall, array %6, out `results/full_fp/`).
- [ ] **Show script to user, then launch.**
- [ ] Sync fixed `marginalize_pairs.py` local ↔ NRNB (local mirror is stale on orientation).

## Downstream (prototype on MT now; swap in first-pass results as they land)
- [ ] `aggregate` → 2,701 × 22 matrix (Δ, maxZ, wilcoxon_p) + category_A/B + involves_composite flag.
- [ ] `call_syntax` — in-matrix empirical null (central-bulk fit) + BH q<0.05 → soft/hard per (pair,CT).
- [ ] Filter to synergistic ≥1 CT; cluster (row-z Δ*, Ward, silhouette k), fixed + clustered cols.
- [ ] Endocrine (11) vs non-endocrine (11) split + recluster within each; tag pairs.
- [ ] Figures PNG **+ PDF** → `syntax_final/figures/`.
- [ ] `1_methodological_overview.html` in `syntax_final/` — explicit numbers, grows with the run.

## Tier 2 — expansion (after discovery)
- [ ] Take pairs called synergistic in the first pass → rerun at **gaps 0–200, n=100** (paper fidelity).
- [ ] Merge tier-2 high-fidelity results for the synergistic set into the final tables/figures.
