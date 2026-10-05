#!/usr/bin/env python
"""Build the all-pairs x all-CT manifest for marginalize_pairs.py and split it into pair-major chunks.

Pair set = every unordered pair of kept catalog motifs, including homodimers (A x A).
Kept motifs = catalog v1.1 minus curator_keep == "No expression" (drops ST53 -> 73 motifs,
2,701 pairs = C(73,2) + 73). The 6 ST05/ST36 sub-motifs have curator_keep = NaN because they
postdate curation; filtering on curator_keep == "keep" would silently drop them, so we only
exclude explicitly.

Pair-major chunking: each chunk file holds whole pairs x all CTs, so a finished SLURM task means
fully profiled pairs (no partial pairs to stitch). One chunk = one array task in slurm/run_pairs.sh.

A/B order matters (A is always placed upstream, so it defines FF/FR/RF/RR and the A__B file names).
Default is lexical. To line a new run up with existing results, pass --orient_like with any table
that has idA/idB columns (e.g. the first-run tables/calls_long.tsv): pairs found there keep its order.

Outputs: <out>/pairs_allpairs_allct.tsv (idA, idB, ct) and <out>/chunks/chunk_NNN.tsv
"""
import os, argparse, itertools
import pandas as pd


def main(a):
    meta = pd.read_csv(a.meta, sep="\t")
    drop = set(meta.loc[meta.curator_keep.astype(str).isin(a.exclude), "short_id"])
    ids = sorted(set(meta.short_id) - drop)
    if a.motifs:
        ids = sorted(set(ids) & set(a.motifs.split(",")))
    cts = pd.read_csv(a.cell_types, sep="\t").sort_values("display_order").cell_type.tolist()
    if a.cts:
        cts = [c for c in cts if c in a.cts.split(",")]

    pairs = list(itertools.combinations_with_replacement(ids, 2)) if a.homodimers \
        else list(itertools.combinations(ids, 2))
    if a.orient_like:
        prev = pd.read_csv(a.orient_like, sep="\t", usecols=["idA", "idB"]).drop_duplicates()
        flip = set(zip(prev.idB, prev.idA)) - set(zip(prev.idA, prev.idB))
        n_flip = sum((A, B) in flip for A, B in pairs)
        pairs = [(B, A) if (A, B) in flip else (A, B) for A, B in pairs]
        print(f"[manifest] --orient_like: flipped {n_flip} pairs to match {a.orient_like}")
    print(f"[manifest] {len(ids)} motifs (dropped {sorted(drop)}), {len(pairs)} pairs x {len(cts)} CT")

    os.makedirs(os.path.join(a.out, "chunks"), exist_ok=True)
    rows = [(A, B, ct) for A, B in pairs for ct in cts]
    pd.DataFrame(rows, columns=["idA", "idB", "ct"]).to_csv(
        os.path.join(a.out, "pairs_allpairs_allct.tsv"), sep="\t", index=False)

    n = 0
    for i in range(0, len(pairs), a.pairs_per_chunk):
        n += 1
        sub = [(A, B, ct) for A, B in pairs[i:i + a.pairs_per_chunk] for ct in cts]
        pd.DataFrame(sub, columns=["idA", "idB", "ct"]).to_csv(
            os.path.join(a.out, "chunks", f"chunk_{n:03d}.tsv"), sep="\t", index=False)
    print(f"[manifest] wrote {len(rows)} rows, {n} chunks of <= {a.pairs_per_chunk} pairs "
          f"-> set the SLURM array to 1-{n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="catalog metadata.tsv (short_id, curator_keep)")
    p.add_argument("--cell_types", required=True, help="config/cell_type_metadata.tsv")
    p.add_argument("-o", "--out", required=True, help="e.g. results/6_synergy/inputs")
    p.add_argument("--exclude", nargs="+", default=["No expression"], help="curator_keep values to drop")
    p.add_argument("--motifs", default=None, help="optional comma-separated short_id subset")
    p.add_argument("--cts", default=None, help="optional comma-separated cell-type subset")
    p.add_argument("--orient_like", default=None, help="TSV with idA,idB whose A/B order to reuse")
    p.add_argument("--no_homodimers", dest="homodimers", action="store_false")
    p.add_argument("--pairs_per_chunk", type=int, default=30)
    main(p.parse_args())
