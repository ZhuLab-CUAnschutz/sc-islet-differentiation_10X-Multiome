#!/usr/bin/env python
"""REF/ALT variant scoring for the multi-task CREsted peak-regression 'design oracle'.

CREsted 1.6.1 ships no variant scorer, so we roll one: for each SNP, extract a
seq_len window centered on the variant, one-hot-encode REF and ALT with crested's
own encoder (channel order must match training), batch-predict the 22-head
accessibility vector for each allele, and report the per-head effect
delta = pred_ALT - pred_REF (the model's targets are normalized/log-scaled, so a
difference is the log-fold-analog — the peak-reg counterpart of ChromBPNet's logfc).

Output: a wide TSV, one row per variant, columns:
  chr end allele1 allele2 id  {group1} {group2} ... {group22}   (each = delta)
plus a companion *.refalt.tsv with the raw per-head REF and ALT predictions.

Head→group mapping is taken from the sorted bigwig filenames (the exact order
crested.import_bigwigs assigns to model output columns).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def get_head_order(bigwigs_dir):
    bws = sorted(glob.glob(os.path.join(bigwigs_dir, "*.bw")))
    groups = [os.path.basename(b)[:-3] for b in bws]  # strip .bw
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bed", required=True, help="combined scoring bed: chr start end a1 a2 id")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--bigwigs_dir", required=True, help="defines the 22-head column order")
    ap.add_argument("--seq_len", type=int, default=2114)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="tensorflow")
    args = ap.parse_args()

    os.environ["KERAS_BACKEND"] = args.backend
    import keras
    import crested
    from crested.utils import one_hot_encode_sequence

    genome = crested.Genome(args.genome, args.chrom_sizes)
    crested.register_genome(genome)

    groups = get_head_order(args.bigwigs_dir)
    print(f"[peakreg] {len(groups)} heads: {groups}")
    assert len(groups) == 22, f"expected 22 heads, got {len(groups)}"

    # 28.keras was saved by keras 2 (tf.keras); its BatchNormalization config carries
    # `renorm*` kwargs that keras 3.13 rejects. keras 3 resolves this built-in by module
    # path (bypassing custom_objects), so patch the class __init__ directly to drop them.
    _bn = keras.layers.BatchNormalization
    _orig_init = _bn.__init__
    def _patched_init(self, *a, **k):
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            k.pop(key, None)
        _orig_init(self, *a, **k)
    _bn.__init__ = _patched_init

    model = keras.models.load_model(args.checkpoint, compile=False)
    n_out = model.output_shape[-1]
    assert n_out == len(groups), f"model has {n_out} heads but {len(groups)} bigwigs"

    bed = pd.read_csv(args.bed, sep="\t", header=None,
                      names=["chr", "start", "end", "allele1", "allele2", "id"])
    half = args.seq_len // 2  # 1057; variant base sits at index `half`

    # Build REF and ALT one-hot arrays (window centered on the variant base;
    # bed 'end' is the 1-based variant position), then batch-predict.
    ref_oh, alt_oh, ok = [], [], []
    for r in bed.itertuples():
        center = r.end
        start = center - 1 - half
        seq = genome.fetch(chrom=r.chr, start=start, end=start + args.seq_len).upper()
        if (len(seq) != args.seq_len or r.allele1 not in "ACGT" or r.allele2 not in "ACGT"):
            ok.append(False)
            continue
        alt_seq = seq[:half] + r.allele2 + seq[half + 1:]
        ref_oh.append(one_hot_encode_sequence(seq, expand_dim=False))
        alt_oh.append(one_hot_encode_sequence(alt_seq, expand_dim=False))
        ok.append(True)
    ok = np.array(ok)
    print(f"[peakreg] scoring {ok.sum()}/{len(bed)} variants ({(~ok).sum()} skipped: edge/non-ACGT)")

    ref_oh = np.asarray(ref_oh, dtype=np.float32)
    alt_oh = np.asarray(alt_oh, dtype=np.float32)
    ref_pred = model.predict(ref_oh, batch_size=args.batch_size, verbose=1)
    alt_pred = model.predict(alt_oh, batch_size=args.batch_size, verbose=1)
    delta = alt_pred - ref_pred  # (n_ok, 22)

    scored = bed[ok].reset_index(drop=True)
    base = scored[["chr", "end", "allele1", "allele2", "id"]]

    wide = pd.concat([base, pd.DataFrame(delta, columns=groups)], axis=1)
    wide.to_csv(args.out, sep="\t", index=False)

    refalt = base.copy()
    for j, g in enumerate(groups):
        refalt[f"{g}_ref"] = ref_pred[:, j]
        refalt[f"{g}_alt"] = alt_pred[:, j]
    refalt.to_csv(args.out.replace(".tsv", "") + ".refalt.tsv", sep="\t", index=False)

    print(f"[peakreg] wrote {args.out} ({len(wide)} rows x {len(groups)} heads)")


if __name__ == "__main__":
    main()
