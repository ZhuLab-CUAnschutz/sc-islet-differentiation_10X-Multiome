#!/usr/bin/env python
"""
06_context_sweep.py — how predicted accessibility changes with sequence context,
with the non-kept ('non-centered') bits DINUCLEOTIDE-shuffled. Two modes:

  slide      : keep the WHOLE enhancer (config.ENHANCER) and slide it through the
               receptive field (offset 0=left .. center .. right); flanks dinuc-shuffled.
               -> tests positional sensitivity.
  keepcenter : keep a central window of width W (the enhancer is the central ~1.1 kb)
               and dinuc-shuffle the flanks; sweep W from full construct down to a core.
               -> isolates the enhancer as the flanking minimal promoter is removed.

Saves long CSV (celltype, mode, x, seed, counts, construct_ref) and plots it.
`--plot-only` re-plots from an existing CSV (head-node safe). GPU for the run.
Run under eugene_tools.
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import plot_locus as pl
import chrombpnet_locus as cl


def build(seq, mode, x, seed):
    if mode == "slide":
        enh = seq[config.ENHANCER[0]:config.ENHANCER[1]]
        return cl.place_at_offset(enh, int(x), config.INPUT_LEN, pad="dinuc", seed=seed)[0]
    return cl.keep_center_shuffle_flanks(seq, int(x), pad="dinuc", seed=seed)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["slide", "keepcenter"], default="keepcenter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--celltypes", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--xs", type=int, nargs="+", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        name, seq = cl.read_fasta(config.FASTA)
        if args.xs is None:
            if args.mode == "slide":
                args.xs = list(np.unique(np.linspace(0, config.INPUT_LEN - (config.ENHANCER[1]-config.ENHANCER[0]), 9).astype(int)))
            else:
                args.xs = [2114, 1700, 1400, 1100, 900, 700, 500, 300]
        cts = args.celltypes or cl.available_celltypes()
        print(f"[plan] mode={args.mode} xs={args.xs} {len(cts)} cells seeds={args.seeds}", flush=True)
        rows = []
        for ct in cts:
            model = cl.load_model(ct)
            ref = cl.predict(model, seq)["counts"]
            for x in args.xs:
                for sd in args.seeds:
                    s = seq if (args.mode == "keepcenter" and x >= len(seq)) else build(seq, args.mode, x, sd)
                    rows.append({"celltype": ct, "mode": args.mode, "x": int(x), "seed": int(sd),
                                 "counts": cl.predict(model, s)["counts"], "construct_ref": ref})
            print(f"  {ct} done", flush=True)
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[done] wrote {args.out}", flush=True)

    df = pd.read_csv(args.out)
    mode = df["mode"].iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    pl.plot_context_sweep(df, mode, ax_abs=a1, ax_norm=a2)
    fig.suptitle(f"sox32 context sweep [{mode}] — non-kept bits dinucleotide-shuffled", fontsize=12)
    fig.tight_layout()
    base = f"figures/sox32_context_sweep_{mode}"
    fig.savefig(base + ".pdf", bbox_inches="tight"); fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {base}.png")


if __name__ == "__main__":
    main()
