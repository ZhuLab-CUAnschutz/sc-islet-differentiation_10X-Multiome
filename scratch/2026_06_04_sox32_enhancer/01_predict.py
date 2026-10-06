#!/usr/bin/env python
"""
01_predict.py — predict accessibility + count contributions across cell types.

Two sequence variants (both keep the enhancer at its native 507-1607 coords, so
all motif/feature coordinates are shared):
  construct  : the 2114 bp window as-is (enhancer + minimal promoter + eGFP)
  enh_dinuc  : keep the central enhancer, dinucleotide-shuffle the flanking
               vector/promoter/reporter — the 'native-like' read with the
               GC-rich reporter context removed (motifs destroyed, dinuc kept)

Saves one .npz with, per (variant, celltype): counts, profile, pred, attr (4,L),
plus the one-hot per variant. GPU SLURM job. Run under eugene_tools.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import chrombpnet_locus as cl


def build_variants(seq):
    enh_w = config.ENHANCER[1] - config.ENHANCER[0]
    cseq, _ = cl.keep_center_shuffle_flanks(seq, enh_w, pad="dinuc", seed=0)
    return {"construct": seq, "enh_dinuc": cseq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=config.FASTA)
    ap.add_argument("--celltypes", nargs="+", default=None)
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--no-contribs", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name, seq = cl.read_fasta(args.fasta)
    assert len(seq) == config.INPUT_LEN, f"expected {config.INPUT_LEN} bp, got {len(seq)}"
    cts = args.celltypes or cl.available_celltypes()
    variants = build_variants(seq)
    print(f"[plan] {name} | {len(cts)} cell types x {list(variants)}", flush=True)

    out = {"name": name, "celltypes": np.array(cts), "variants": np.array(list(variants)),
           "input_len": config.INPUT_LEN, "output_len": config.OUTPUT_LEN}
    for v, vseq in variants.items():
        out[f"seq__{v}"] = vseq
        out[f"ohe__{v}"] = cl.ohe(vseq)[0]
        out[f"meta_offset__{v}"] = 0            # enhancer stays at native coords

    for ct in cts:
        model = cl.load_model(ct)
        for v, vseq in variants.items():
            p = cl.predict(model, vseq)
            out[f"counts__{v}__{ct}"] = p["counts"]
            out[f"profile__{v}__{ct}"] = p["profile"].astype("float32")
            out[f"pred__{v}__{ct}"] = p["pred"].astype("float32")
            if not args.no_contribs:
                out[f"attr__{v}__{ct}"] = cl.count_contributions(model, vseq, n_shuffles=args.n_shuffles)
        print(f"  {ct}: " + " ".join(f"{v}={out[f'counts__{v}__{ct}']:.0f}" for v in variants), flush=True)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
