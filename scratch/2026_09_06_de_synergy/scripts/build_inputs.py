#!/usr/bin/env python
"""Build inputs for the DE 4-TF synergy run (EOMES / MIXL1 / SOX17 / FOXH1).

Writes inputs/motifs_de4.npz  -- 7 keys usable by marginalize_pairs.py --cwms:
    ST18, ST36_sub1, ST66      signed CWMs copied verbatim from catalog v1.1 cwms.npz
                               (same trim/argmax code path as every prior synergy run)
    J_EOMES, J_MIXL1, J_SOX17, J_FOXH1
                               one-hot (W,4) IC-trimmed JASPAR consensus. A one-hot array has
                               uniform per-position |importance|, so cwm_consensus_ohe() trims
                               nothing and argmax returns the consensus unchanged.
       inputs/pairs_de4.tsv    19 unique (idA,idB) rows, ct=DE  (run manifest)
       inputs/pairs_by_set.tsv 20 rows, one per set x pair      (analysis join)
       inputs/inserts.tsv      the 7 inserts: source, consensus, width
"""
import os, re, itertools, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); IN = os.path.join(HERE, "..", "inputs")
IC_MIN = 0.2   # bits; single-motif PPM-trim convention

JASPAR = {"J_FOXH1": ("MA0479.1", "FOXH1"), "J_MIXL1": ("MA0662.1", "MIXL1"),
          "J_SOX17": ("MA0078.1", "SOX17"), "J_EOMES": ("MA0800.1", "EOMES")}
CATALOG = {"ST18": "EOMES", "ST36_sub1": "MIXL1", "ST66": "SOX17"}

def read_jaspar(path):
    rows = {}
    for line in open(path):
        m = re.match(r"^([ACGT])\s*\[([^\]]*)\]", line.strip())
        if m: rows[m.group(1)] = np.array(m.group(2).split(), float)
    return np.stack([rows[b] for b in "ACGT"], 1)          # (W,4) counts

def ic_bits(pfm):
    p = pfm / pfm.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 + np.where(p > 0, p * np.log2(p), 0).sum(1)

def trimmed_consensus_ohe(pfm, ic_min=IC_MIN):
    ic = ic_bits(pfm); keep = np.where(ic >= ic_min)[0]
    core = pfm[keep.min():keep.max() + 1]
    idx = core.argmax(1); ohe = np.zeros((len(idx), 4), np.float32); ohe[np.arange(len(idx)), idx] = 1
    return ohe, "".join("ACGT"[i] for i in idx), ic

def cwm_consensus(cwm, imp_frac=0.1):                     # mirror of marginalize_pairs.cwm_consensus_ohe
    W4 = cwm if cwm.shape[1] == 4 else cwm.T
    imp = np.abs(W4).sum(1); keep = np.where(imp >= imp_frac * imp.max())[0]
    core = W4[keep.min():keep.max() + 1]
    return "".join("ACGT"[i] for i in core.argmax(1))

def main():
    cw = np.load(os.path.join(IN, "cwms_v1.1.npz"))
    out, inserts = {}, []
    for k, tf in CATALOG.items():
        out[k] = cw[k]; cons = cwm_consensus(cw[k])
        inserts.append({"id": k, "tf": tf, "source": "catalog v1.1 CWM (argmax, imp_frac=0.1)", "consensus": cons, "width": len(cons)})
    for k, (mid, tf) in JASPAR.items():
        pfm = read_jaspar(os.path.join(IN, "jaspar", f"{mid}.jaspar"))
        ohe, cons, ic = trimmed_consensus_ohe(pfm); out[k] = ohe
        full = "".join("ACGT"[i] for i in pfm.argmax(1))
        inserts.append({"id": k, "tf": tf, "source": f"JASPAR {mid} (IC>={IC_MIN} bit trim)", "consensus": cons, "width": len(cons),
                        "untrimmed_consensus": full, "untrimmed_width": len(full), "ic_bits": ",".join(f"{x:.2f}" for x in ic)})
    np.savez_compressed(os.path.join(IN, "motifs_de4.npz"), **out)
    ins = pd.DataFrame(inserts); ins.to_csv(os.path.join(IN, "inserts.tsv"), sep="\t", index=False)
    print(ins[["id", "tf", "consensus", "width", "source"]].to_string(index=False))

    SETS = {"set1_catalog+JFOXH1": ["ST18", "ST36_sub1", "ST66", "J_FOXH1"],
            "set2_jaspar":         ["J_EOMES", "J_MIXL1", "J_SOX17", "J_FOXH1"]}
    tf = {r["id"]: r["tf"] for r in inserts}
    rows = []
    for s, ids in SETS.items():
        for a, b in itertools.combinations(ids, 2):
            rows.append({"idA": a, "idB": b, "tfA": tf[a], "tfB": tf[b], "set": s, "kind": "heterotypic", "ct": "DE"})
        for a in ids:
            rows.append({"idA": a, "idB": a, "tfA": tf[a], "tfB": tf[a], "set": s, "kind": "homodimer", "ct": "DE"})
    byset = pd.DataFrame(rows); byset.to_csv(os.path.join(IN, "pairs_by_set.tsv"), sep="\t", index=False)
    run = byset.drop_duplicates(["idA", "idB"]).drop(columns=["set"])
    run.to_csv(os.path.join(IN, "pairs_de4.tsv"), sep="\t", index=False)
    print(f"\npairs_by_set: {len(byset)} rows; run manifest: {len(run)} unique pairs")

if __name__ == "__main__":
    main()
