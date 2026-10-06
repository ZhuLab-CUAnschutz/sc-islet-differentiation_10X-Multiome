#!/usr/bin/env python
"""Build the inserts and pair manifest for a FOCUS pairwise screen: a handful of TFs run at full
fidelity in one cell type, rather than the all-pairs sweep.

This is the step that produces the pairwise calls the higher-order pipeline consumes
(build_higher_order_inputs.py --calls). Run order:
    build_focus_inputs.py -> slurm/run_focus_pairs.sh -> analyze_focus_pairs.py
    -> build_higher_order_inputs.py -> slurm/run_triples.sh

The focus set is a spec TSV with columns:
    id      insert key, e.g. ST18 or J_FOXH1
    tf      TF name
    source  "catalog" (take the signed CWM verbatim from the catalog cwms.npz) or a JASPAR
            matrix id such as MA0479.1 (read inputs/jaspar/<id>.jaspar)
    sets    comma-separated set labels this insert belongs to

Running the same TF from both the catalog and JASPAR is the point of "sets": it shows whether a
synergy call survives changing the motif representation. Each set is expanded to all heterotypic
pairs plus every homodimer.

Within a set, A/B order follows spec row order; A is placed upstream, so that order fixes the
orientation labels and the npz filenames. Pass --orient_like <existing pairs table> to reuse a
previous run's ordering.

Outputs: <out>/motifs_focus.npz, inserts.tsv, pairs_by_set.tsv (analysis join),
         pairs_run.tsv (de-duplicated run manifest for marginalize_pairs.py)

Generalized from the DE 4-TF screen in scratch/2026_09_06_de_synergy (EOMES, MIXL1, SOX17, FOXH1).
"""
import os, re, argparse, itertools
import numpy as np, pandas as pd

IC_MIN = 0.2   # bits; matches the single-motif PPM-trim convention


def read_jaspar(path):
    rows = {}
    for line in open(path):
        m = re.match(r"^([ACGT])\s*\[([^\]]*)\]", line.strip())
        if m:
            rows[m.group(1)] = np.array(m.group(2).split(), float)
    return np.stack([rows[b] for b in "ACGT"], 1)          # (W,4) counts


def ic_bits(pfm):
    p = pfm / pfm.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 + np.where(p > 0, p * np.log2(p), 0).sum(1)


def trimmed_consensus_ohe(pfm, ic_min=IC_MIN):
    """IC-trim a JASPAR PFM, then take the argmax consensus as a one-hot (W,4).

    A one-hot array has uniform per-position |importance|, so marginalize_pairs' own
    cwm_consensus_ohe() trims nothing further and its argmax returns this consensus unchanged.
    """
    ic = ic_bits(pfm)
    keep = np.where(ic >= ic_min)[0]
    core = pfm[keep.min():keep.max() + 1]
    idx = core.argmax(1)
    ohe = np.zeros((len(idx), 4), np.float32)
    ohe[np.arange(len(idx)), idx] = 1
    return ohe, "".join("ACGT"[i] for i in idx), ic


def cwm_consensus(cwm, imp_frac=0.1):
    """Mirror of marginalize_pairs.cwm_consensus_ohe, for reporting the inserted string."""
    W4 = cwm if cwm.shape[1] == 4 else cwm.T
    imp = np.abs(W4).sum(1)
    keep = np.where(imp >= imp_frac * imp.max())[0]
    core = W4[keep.min():keep.max() + 1]
    return "".join("ACGT"[i] for i in core.argmax(1))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    spec = pd.read_csv(a.spec, sep="\t")
    for col in ("id", "tf", "source", "sets"):
        if col not in spec.columns:
            raise SystemExit(f"--spec needs a {col} column; got {list(spec.columns)}")

    cw = np.load(a.cwms)
    out, inserts = {}, []
    for r in spec.itertuples():
        if str(r.source).lower() == "catalog":
            if r.id not in cw:
                raise SystemExit(f"{r.id} not in {a.cwms}")
            out[r.id] = cw[r.id]
            cons = cwm_consensus(cw[r.id])
            inserts.append({"id": r.id, "tf": r.tf,
                            "source": "catalog CWM (argmax, imp_frac=0.1)",
                            "consensus": cons, "width": len(cons)})
        else:
            pfm = read_jaspar(os.path.join(a.jaspar_dir, f"{r.source}.jaspar"))
            ohe, cons, ic = trimmed_consensus_ohe(pfm)
            out[r.id] = ohe
            full = "".join("ACGT"[i] for i in pfm.argmax(1))
            inserts.append({"id": r.id, "tf": r.tf,
                            "source": f"JASPAR {r.source} (IC>={IC_MIN} bit trim)",
                            "consensus": cons, "width": len(cons),
                            "untrimmed_consensus": full, "untrimmed_width": len(full),
                            "ic_bits": ",".join(f"{x:.2f}" for x in ic)})

    np.savez_compressed(os.path.join(a.out, "motifs_focus.npz"), **out)
    ins = pd.DataFrame(inserts)
    ins.to_csv(os.path.join(a.out, "inserts.tsv"), sep="\t", index=False)
    print(ins[["id", "tf", "consensus", "width", "source"]].to_string(index=False))

    # every set -> all heterotypic pairs + every homodimer
    tf = dict(zip(spec.id, spec.tf))
    sets = {}
    for r in spec.itertuples():
        for s in str(r.sets).split(","):
            sets.setdefault(s.strip(), []).append(r.id)

    flip = set()
    if a.orient_like:
        prev = pd.read_csv(a.orient_like, sep="\t", usecols=["idA", "idB"]).drop_duplicates()
        flip = set(zip(prev.idB, prev.idA)) - set(zip(prev.idA, prev.idB))

    rows = []
    for s, ids in sets.items():
        for x, y in itertools.combinations(ids, 2):
            if (x, y) in flip:          # keep an existing run's A/B order
                x, y = y, x
            rows.append({"idA": x, "idB": y, "tfA": tf[x], "tfB": tf[y],
                         "set": s, "kind": "heterotypic", "ct": a.ct})
        for x in ids:
            rows.append({"idA": x, "idB": x, "tfA": tf[x], "tfB": tf[x],
                         "set": s, "kind": "homodimer", "ct": a.ct})
    byset = pd.DataFrame(rows)
    byset.to_csv(os.path.join(a.out, "pairs_by_set.tsv"), sep="\t", index=False)
    run = byset.drop_duplicates(["idA", "idB"]).drop(columns=["set"])
    run.to_csv(os.path.join(a.out, "pairs_run.tsv"), sep="\t", index=False)
    print(f"\n{len(sets)} set(s): " + "; ".join(f"{s} ({len(v)} motifs)" for s, v in sets.items()))
    print(f"pairs_by_set: {len(byset)} rows; run manifest: {len(run)} unique pairs -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="focus-set TSV: id, tf, source, sets")
    p.add_argument("--cwms", required=True, help="catalog cwm/cwms.npz (for source=catalog)")
    p.add_argument("--jaspar_dir", default=None, help="dir of <matrix_id>.jaspar files")
    p.add_argument("--ct", required=True, help="cell type to screen in, e.g. DE")
    p.add_argument("--orient_like", default=None,
                   help="TSV with idA,idB whose A/B order to reuse (A is placed upstream, so this "
                        "fixes orientation labels and npz filenames for an existing run)")
    p.add_argument("-o", "--out", required=True)
    main(p.parse_args())
