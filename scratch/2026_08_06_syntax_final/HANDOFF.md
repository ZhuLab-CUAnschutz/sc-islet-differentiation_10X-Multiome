# Handoff — syntax synergy work (as of 2026-07-27)

Quick note on cluster state, what was cancelled, and what still needs picking up. Companion to
`PLAN.md` in this folder (the Monday deliverable plan).

## Cluster jobs

### Cancelled 2026-07-27 (can be resumed)
Five DeepLIFT attribution-sweep **animation** jobs, all running `run_dl_sweep2.sh`
(job ids 13323060–13323064: `sw_hard, sw_soft, sw_ctA, sw_ctB, sw_orient`). These render the
"motif B sliding across motif A, attribution logo morphing" GIF idea (example pair PDX1×FOXA2,
ST52×ST65 liver). Figure-polish, not needed for the Monday deliverables → cancelled to free GPUs for
`full_allct`.
- **To resume:** `scripts/deeplift_sweep.py` + `scripts/run_dl_sweep2.sh` in the cluster sandbox
  `.../sandbox/2026_07_19_syntax_pairwise_marg/`. Re-`sbatch` when GPUs are free and we actually want
  the animation figures.

### Left running / intact (do NOT touch)
- `full_allct` (13322971) — **the Monday linchpin**: all 2628 pairs × 22 cell types, single-task
  ChromBPNet. ~16% done on 2026-07-27, GPU-limited. Everything in `PLAN.md` Phases 1–4 waits on this.
- `orfix` (13323328) — orientation-bug re-run of the 91 affected pairs (2nd-motif-palindrome dedup fix).
- `cbp_vscore` / `pr_vscore` (13323485/6) — **different project** (variant-effects sandbox
  `2026_07_25_variant_effects/`), unrelated to synergy. Leave alone.

## Deferred / open items to pick up

1. **CWM vs PWM trim mismatch (flagged, not fixed).** Synergy inserts the CWM consensus trimmed at
   **10% of peak** importance (`cwm_consensus_ohe`, `imp_frac=0.1` in `marginalize_pairs.py`), but the
   report **logos** and the stored `cwms.npz` are trimmed at **30%**, and the single-motif
   marginalization uses a **PPM with IC≥0.2-bit** trim entirely. So what we insert for synergy carries
   lower-importance flank the displayed logo doesn't. Decision: keep 0.1 for now (don't disturb the
   running full_allct); unify later (e.g. `imp_frac=0.3` to match logos, or drive both single-motif and
   pairwise off a shared width descriptor in `metadata.tsv`). Unification needs re-marginalizing.

2. **Null contamination (documented, quantified).** The 42-pair lineage-mismatched null is
   ~14% contaminated (6/42 pairs are themselves synergistic, TEAD4/NEUROD1 combos), which inflated the
   per-CT bars conservatively (up to +48% in DE). Full detail in
   `../syntax_2026-07-25/NULL_CALIBRATION_PROVENANCE.md`. The Monday pipeline retires this null in favor
   of the in-matrix empirical null.

3. **DeepLIFT sweep animations** (see cancelled jobs above) — resume if we want the morphing figures.

## Where things live
- Cluster sandbox: `carter:/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg/`
- Local working dir: `~/Desktop/ucsd/syntax_2026-07-25/` (scripts, methods notes, specificity outputs)
- Deliverable bundle (07-26 reports): `~/Desktop/ucsd/syntax_2026_07_26/`
- This folder `~/Desktop/ucsd/syntax_final/` — Monday deliverables (figures PNG+PDF) land in `figures/`.
- SLURM: always `ssh nrnb-login.ucsd.edu 'bash -lc "sbatch ..."'` (login shell; non-login ssh drops
  the SLURM binaries from PATH). Never compute on the login node.
