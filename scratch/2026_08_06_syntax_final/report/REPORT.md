# Synergy collaborator bundle — first pass

Generated from `syntax_final/` (single-task ChromBPNet, 22 cell types).

- Pairs: 2701 (2,628 heterotypic + 73 homodimers); motifs: 73 (catalog v1.1 minus ST53).
- Sweep: 1-bp, gaps 0–50, n=32 backgrounds. Calling: in-matrix empirical null + BH, q<0.05.
- Synergistic pairs (≥1 CT): 660. Pair×CT calls: 2180 (403 hard).

## Files
- `index.html` — landing page.
- `1_methodological_overview.html` — methods, with figures.
- `2_all_pairs_report.html` — interactive all-pairs table + per-pair detail (live FDR/Δ/frac_pos).
- `3_synergy_by_cell_type.html` — synergistic pairs browsable per cell type.
- `3_synergy_by_cell_type.xlsx`, `data/*.tsv` — tables.
- `figures/` — clustermap + orientation/gap QC.

Model predictions, not experimental measurements. First pass; a later tier re-runs called pairs at n=100, gaps 0–200.
