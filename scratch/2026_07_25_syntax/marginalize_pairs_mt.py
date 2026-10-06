#!/usr/bin/env python
"""Pairwise in-silico motif marginalization through the CRESTED MULTITASK model (one forward pass =
all 22 cell types). Same method as the single-task marginalize_pairs.py, adapted to the Keras model:
insert two CWM-consensus motifs across orientation x spacing (0-200 bp) into shared dinuc-shuffled
backgrounds, read ΔJ for ALL 22 cell types per pass. Because every arrangement scores all cell types
at once, one sweep yields the full per-CT synergy matrix (arrangement re-optimized per cell type).

Output per chunk: summary__chunk<c>.tsv with one row per (pair, cell type):
  idA,idB,ct,dJ_opt,dS,delta,opt_orient,opt_gap,opt_center_center,maxZ,wilcoxon_p,wa,wb
Env: .venv-keras (Python 3.11, KERAS_BACKEND=tensorflow). carter-gpu.
"""
import os, argparse, glob
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from marginalize_multitask import k_shuffle, one_hot, NARROWPEAKS_SCHEMA
import pyfaidx

def cwm_consensus_wl(cwm, imp_frac=0.1):
    """signed CWM (W,4 or 4,W) -> importance-trimmed argmax one-hot, channels-last (w,4)."""
    W4 = cwm if cwm.shape[1] == 4 else cwm.T
    imp = np.abs(W4).sum(1)
    keep = np.where(imp >= imp_frac * imp.max())[0] if imp.max() > 0 else np.arange(len(W4))
    core = W4[keep.min():keep.max() + 1]
    ohe = np.zeros((len(core), 4), np.float32); ohe[np.arange(len(core)), core.argmax(1)] = 1.0
    return ohe

def rc_wl(ohe):  # (w,4) reverse-complement: reverse positions + complement columns A<->T,C<->G
    return ohe[::-1][:, [3, 2, 1, 0]].copy()

def backgrounds(genome_fasta, peak_glob, n, flank=1057, seed=1234):
    rng = np.random.default_rng(seed); genome = pyfaidx.Fasta(genome_fasta)
    peaks = pd.concat([pd.read_csv(f, sep="\t", header=None, names=NARROWPEAKS_SCHEMA) for f in sorted(glob.glob(peak_glob))],
                      ignore_index=True).drop_duplicates(["chrom", "start", "end"])
    peaks["mid"] = (peaks.start + peaks.end) // 2
    sub = peaks.sample(n=min(n * 4, len(peaks)), random_state=seed)
    seqs = []
    for _, r in sub.iterrows():
        s0, e0 = int(r.mid) - flank, int(r.mid) + flank
        if s0 < 0: continue
        try: s = str(genome[r.chrom][s0:e0].seq).upper()
        except Exception: continue
        if len(s) != 2 * flank or sum(b not in "ACGT" for b in s) / len(s) > 0.05: continue
        seqs.append(k_shuffle(s, 2, rng))
        if len(seqs) >= n: break
    return np.stack([one_hot(s) for s in seqs])  # (n, L, 4)

def predict_counts(model, X, bs):
    return model.predict(X, batch_size=bs, verbose=0)  # (n, 22)

def main(a):
    import keras
    os.makedirs(a.out, exist_ok=True)
    cwms = np.load(a.cwms)
    import anndata as ad
    cts = list(ad.read_h5ad(a.adata, backed="r").obs_names)   # head order (22)
    pairs = pd.read_csv(a.pairs, sep="\t")
    if a.chunk is not None:
        pairs = pairs.iloc[np.array_split(np.arange(len(pairs)), a.nchunks)[a.chunk]]
    X = backgrounds(a.genome, a.peak_glob, a.n_seqs, seed=a.seed)      # (n,L,4)
    n, L, _ = X.shape; mid = L // 2
    print(f"[bg] {n} shared backgrounds, L={L}, {len(cts)} CT heads", flush=True)
    model = keras.models.load_model(a.model, compile=False)
    c_bg = predict_counts(model, X, a.batch_size)                      # (n,22)

    def edit(A, B, gap):
        wa, wb = A.shape[0], B.shape[0]; comp = wa + gap + wb
        sA = mid - comp // 2; sB = sA + wa + gap
        if sA < 0 or sB + wb > L: return None, None
        Xe = X.copy(); Xe[:, sA:sA + wa, :] = A; Xe[:, sB:sB + wb, :] = B
        return Xe, (sB + wb / 2) - (sA + wa / 2)
    def single(M):
        w = M.shape[0]; s = mid - w // 2; Xe = X.copy(); Xe[:, s:s + w, :] = M
        return predict_counts(model, Xe, a.batch_size) - c_bg          # (n,22)

    gaps = np.arange(0, a.maxdist + 1, a.gap_step)
    rows = []
    for r in pairs.itertuples():
        A0 = cwm_consensus_wl(cwms[r.idA]); B0 = cwm_consensus_wl(cwms[r.idB])
        same = r.idA == r.idB
        palA = np.array_equal(A0, rc_wl(A0)); palB = np.array_equal(B0, rc_wl(B0))
        orients = ([("FF", 0, 0), ("FR", 0, 1), ("RR", 1, 1)] if same or (palA and palB)
                   else [("FF", 0, 0), ("FR", 0, 1)] if (palA or palB)
                   else [("FF", 0, 0), ("FR", 0, 1), ("RF", 1, 0), ("RR", 1, 1)])
        dA = single(A0); dB = single(B0); dS = dA + dB                 # (n,22)
        # dJ grid: (n_orient, n_gap, 22) mean; keep per-seq only at running best per CT
        nO, nG, nCT = len(orients), len(gaps), len(cts)
        dJ = np.full((nO, nG, nCT), np.nan, np.float32)
        best_seq = np.zeros((nCT, n), np.float32); best_val = np.full(nCT, -np.inf)
        best_arr = [("", 0, 0.0)] * nCT
        for oi, (on, ra, rb) in enumerate(orients):
            Aa = rc_wl(A0) if ra else A0; Bb = rc_wl(B0) if rb else B0
            chunks, meta = [], []                                      # batch all gaps -> one predict
            for gi, g in enumerate(gaps):
                Xe, cc = edit(Aa, Bb, int(g))
                if Xe is None: continue
                chunks.append(Xe); meta.append((gi, int(g), float(cc)))
            if not chunks: continue
            pred = predict_counts(model, np.concatenate(chunks, 0), a.batch_size).reshape(len(meta), n, nCT)
            for row, (gi, g, cc) in enumerate(meta):
                dj = pred[row] - c_bg; m = dj.mean(0); dJ[oi, gi] = m  # (n,22)
                for c in np.where(m > best_val)[0]:
                    best_val[c] = m[c]; best_seq[c] = dj[:, c]; best_arr[c] = (on, g, cc)
        finite = dJ[np.isfinite(dJ)]
        for c, ct in enumerate(cts):
            grid_c = dJ[:, :, c]; fc = grid_c[np.isfinite(grid_c)]
            z = (best_val[c] - fc.mean()) / (fc.std() + 1e-9) if fc.size else np.nan
            indep = dS[:, c]
            try: p = float(wilcoxon(best_seq[c], indep)[1])
            except Exception: p = np.nan
            on, g, cc = best_arr[c]
            rows.append({"idA": r.idA, "idB": r.idB, "ct": ct, "dJ_opt": round(float(best_val[c]), 4),
                         "dS": round(float(dS[:, c].mean()), 4), "delta": round(float(best_val[c] - dS[:, c].mean()), 4),
                         "opt_orient": on, "opt_gap": g, "opt_center_center": round(cc, 1),
                         "maxZ": round(float(z), 2), "wilcoxon_p": p, "wa": int(A0.shape[0]), "wb": int(B0.shape[0])})
        print(f"{r.idA}__{r.idB}: best CT Δ={float(np.nanmax(best_val-dS.mean(0))):.3f}", flush=True)
    tag = f"chunk{a.chunk}" if a.chunk is not None else "all"
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"summary__{tag}.tsv"), sep="\t", index=False)
    print(f"[done] {len(rows)} pair-CT rows -> summary__{tag}.tsv", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--cwms", required=True)
    p.add_argument("--adata", required=True); p.add_argument("-g", "--genome", required=True)
    p.add_argument("--peak_glob", required=True); p.add_argument("--pairs", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--chunk", type=int, default=None); p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("-n", "--n_seqs", type=int, default=64); p.add_argument("--maxdist", type=int, default=200)
    p.add_argument("--gap_step", type=int, default=2); p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    main(p.parse_args())
