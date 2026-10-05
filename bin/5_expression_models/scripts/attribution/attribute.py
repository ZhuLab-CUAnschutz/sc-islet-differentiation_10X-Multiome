#!/usr/bin/env python
"""Step 6 - attribution smoke test on a trained arm (only run if metrics pass).

For a few design-target genes, attribute *cell-type-specific* expression (Decima's
specificity transform: on-target track vs off-target tracks) and check the highlighted
sequence recovers known islet TF motifs in the right cell type.

  python attribute.py --ckpt <path> --arm decima --regime lora

⚠️ Attribution + motif scan are decima/grelu/captum/tangermeme API; VERIFY.
"""

import argparse
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, ".."))
sys.path.append(os.path.join(HERE, "..", "train"))
from model import SeqExpr  # noqa: E402

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
FIG_DIR = f"{DATA_DIR}/results/5_expression_models/figures/step6"

# gene -> the cell type whose specificity we attribute (its known TF context)
DESIGN_GENES = {
    "INS": "late_SC_beta",     # expect RFX, PDX1, NKX6-1, MAFA, NEUROD1, ISL1
    "GCG": "late_SC_alpha",    # expect ARX, MAFB, ISL1
    "TPH1": "late_SC_EC",      # expect LMX1A, and EC-lineage motifs
}
# HOCOMOCO/known islet TF motifs to scan attributions against (VERIFY motif source path).
ISLET_TFS = ["RFX3", "PDX1", "NKX6-1", "PAX6", "NEUROD1", "ISL1", "MAFA", "MAFB", "ARX", "LMX1A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["decima", "borzoi"], required=True)
    ap.add_argument("--regime", choices=["probe", "lora", "full"], required=True)
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    model = SeqExpr.load_from_checkpoint(args.ckpt, arm=args.arm, regime=args.regime).eval().cuda()

    # VERIFY: use Decima's attributer with the "specificity" transform (on- vs off-target
    # tasks), Captum InputXGradient. Signature per decima.interpret.attributer.
    from decima.interpret import attribute_gene  # type: ignore  # VERIFY
    from grelu.interpret.motifs import scan_sequences  # type: ignore  # VERIFY

    track_order = _track_order()
    for gene, ct in DESIGN_GENES.items():
        on_idx = track_order.index(ct)
        off_idx = [i for i in range(len(track_order)) if i != on_idx]
        attr = attribute_gene(model, gene, on_tasks=[on_idx], off_tasks=off_idx,
                              method="inputxgradient")  # VERIFY
        _plot_attr_track(gene, ct, attr, f"{FIG_DIR}/{gene}_{ct}_attr.png")
        hits = scan_sequences(attr_seq=attr, motifs=ISLET_TFS)  # VERIFY
        hits.to_csv(f"{FIG_DIR}/{gene}_{ct}_motif_hits.csv", index=False)
        print(f"{gene} ({ct}): {len(hits)} motif hits -> {FIG_DIR}")


def _track_order():
    import anndata as ad

    pb = ad.read_h5ad(f"{DATA_DIR}/results/2_process_data/pseudobulk_expression.h5ad")
    return list(pb.obs_names)


def _plot_attr_track(gene, ct, attr, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_pos = np.asarray(attr).sum(axis=0) if np.ndim(attr) > 1 else np.asarray(attr)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(per_pos, lw=0.4)
    ax.set_title(f"{gene} specificity attribution ({ct} vs others)")
    ax.set_xlabel("position in 524 kb window")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
