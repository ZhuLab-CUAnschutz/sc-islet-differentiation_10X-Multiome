#!/usr/bin/env python
"""In-silico PAIRWISE motif marginalization (motif synergy / TF-binding syntax) for single-task
ChromBPNet models. Generalizes .../ML4GLand/chrombpnet/scripts/marginalization.py from one insert
to a spacing x orientation sweep of two motifs, following Kundaje-lab "Dissecting regulatory syntax
in human development" (bioRxiv 2025.04.30.651381), Methods "In silico marginalizations to assess
motif synergy".

Adversarially-reviewed choices baked in:
  * consensus  = argmax of the collapsed signed CWM (cwms.npz), importance-trimmed  [fix B1]
  * background = real GC-matched inaccessible negatives (chrombpnet filtered.nonpeaks.bed) [fix M1]
  * single fold: per-arrangement dJ is a point estimate; we bootstrap over the N backgrounds to
    check the max-arrangement Z is stable (no cross-fold error bars possible)             [fix B3]
  * uniform 1-bp distance grid 0..MAXDIST (never thin distances -> keeps the Z denominator) [fix M3]
  * palindrome (RC==self) orientation dedup; distinct-><=4, same/palindrome-> fewer         [fix m3]
  * explicit fit + no-clobber assertions on the two substitutions                           [fix m4]
  * store motif widths + center-to-center distance for the optimal arrangement              [fix m1]
Thresholds (Z>4 hard, (dJ-dS)>0.15 soft) are the paper's; for the all-pairs regime they must be
recalibrated against an empirical null of lineage-mismatched pairs (done in aggregation, not here) [fix B2].

Env: eugene_tools (tangermeme 1.0.3, bpnetlite 0.8.1, seqpro, torch). GPU via SLURM carter-gpu only.
"""
import os, argparse
import numpy as np, pandas as pd, torch
import pyfaidx
from scipy.stats import wilcoxon
from bpnetlite.bpnet import BPNet
from tangermeme.ersatz import substitute
from tangermeme.predict import predict

BED3 = ["chrom", "start", "end"]

# ---------- consensus from signed CWM ----------
def cwm_consensus_ohe(cwm, imp_frac=0.1):
    """cwm: (W,4) signed contribution. Trim flanks with per-pos importance < imp_frac*max, argmax -> (4,w) OHE."""
    W4 = cwm if cwm.shape[1] == 4 else cwm.T
    imp = np.abs(W4).sum(1)
    if imp.max() <= 0:
        keep = np.arange(len(W4))
    else:
        keep = np.where(imp >= imp_frac * imp.max())[0]
    core = W4[keep.min():keep.max() + 1]
    idx = core.argmax(1)
    ohe = np.zeros((len(idx), 4), np.float32); ohe[np.arange(len(idx)), idx] = 1.0
    return ohe.T  # (4,w)

def rc(ohe):
    out = ohe[:, ::-1].copy()
    return out[[3, 2, 1, 0], :]

def is_palindrome(ohe):
    return np.array_equal(ohe, rc(ohe))

# ---------- backgrounds: real GC-matched inaccessible negatives ----------
def make_backgrounds(genome_fasta, background_bed, n_seqs, width=2114, seed=1234):
    genome = pyfaidx.Fasta(genome_fasta)
    bed = pd.read_csv(background_bed, sep="\t", header=None).iloc[:, :3]
    bed.columns = BED3
    bed["mid"] = (bed["start"] + bed["end"]) // 2
    bed["s"] = bed["mid"] - width // 2; bed["e"] = bed["mid"] + width // 2
    bed = bed[bed["s"] >= 0]
    sub = bed.sample(n=min(n_seqs, len(bed)), random_state=seed)
    seqs, MAP = [], {"A": 0, "C": 1, "G": 2, "T": 3}
    for _, r in sub.iterrows():
        s = genome[r["chrom"]][int(r["s"]):int(r["e"])].seq.upper()
        if len(s) != width:
            continue
        arr = np.zeros((width, 4), np.float32); ok = True
        for i, ch in enumerate(s):
            j = MAP.get(ch)
            if j is None: ok = False; break
            arr[i, j] = 1.0
        if ok: seqs.append(arr.T)
    X = torch.tensor(np.stack(seqs), dtype=torch.float32)  # (n,4,width)
    return X

# ---------- prediction: chrombpnet log-count head ----------
def logcounts(model, Xe, batch_size, device):
    y = predict(model, Xe, batch_size=batch_size, device=device, verbose=False)
    yc = y[1] if isinstance(y, (list, tuple)) else y
    yc = yc.reshape(yc.shape[0], -1)
    return yc.sum(-1).cpu().numpy() if yc.shape[1] > 1 else yc.cpu().numpy().ravel()

def _mot(m):
    return torch.tensor(np.ascontiguousarray(m, dtype=np.float32))[None]  # (1,4,w) for tangermeme

def center_insert(X, mot):
    mid = X.shape[-1] // 2; w = mot.shape[-1]; s = mid - w // 2
    return substitute(X, _mot(mot), start=s)

def pair_edit(X, A, B, gap):
    """place A then B centered, inner-edge gap=gap. Returns edited X, center-to-center distance."""
    L = X.shape[-1]; mid = L // 2; wa, wb = A.shape[-1], B.shape[-1]
    comp = wa + gap + wb
    sA = mid - comp // 2; sB = sA + wa + gap
    assert sA >= 0 and sB + wb <= L, f"pair span {comp} does not fit in {L}"
    assert sB >= sA + wa, "second insert would clobber first"
    Xe = substitute(X, _mot(A), start=sA)
    Xe = substitute(Xe, _mot(B), start=sB)
    cc = (sB + wb / 2) - (sA + wa / 2)
    return Xe, cc

# ---------- one pair sweep (arrangements batched into bulk GPU calls) ----------
def sweep_pair(model, X, A, B, maxdist, same, batch_size, device, gap_step=1):
    n = X.shape[0]
    c_bg = logcounts(model, X, batch_size, device)
    dA_seq = logcounts(model, center_insert(X, A), batch_size, device) - c_bg
    dB_seq = logcounts(model, center_insert(X, B), batch_size, device) - c_bg
    palA, palB = is_palindrome(A), is_palindrome(B)
    if same or (palA and palB):
        orients = [("FF", False, False), ("FR", False, True), ("RR", True, True)]
    elif palA or palB:
        orients = [("FF", False, False), ("FR", False, True)]  # RC of the palindromic one is redundant
    else:
        orients = [("FF", False, False), ("FR", False, True), ("RF", True, False), ("RR", True, True)]
    gaps = np.arange(0, maxdist + 1, gap_step)          # UNIFORM grid (Z denominator stays valid)
    dJ = np.full((len(orients), len(gaps)), np.nan)
    best = {"val": -np.inf}
    for oi, (on, ra, rb) in enumerate(orients):
        Aa = rc(A) if ra else A; Bb = rc(B) if rb else B
        chunks, meta = [], []                      # build every gap's edited stack, predict once
        for gi, k in enumerate(gaps):
            try:
                Xe, cc = pair_edit(X, Aa, Bb, int(k))
            except AssertionError:
                continue
            chunks.append(Xe); meta.append((gi, int(k), float(cc)))
        if not chunks:
            continue
        cj = logcounts(model, torch.cat(chunks, 0), batch_size, device).reshape(len(meta), n)
        for row, (gi, k, cc) in enumerate(meta):
            dj = cj[row] - c_bg; m = float(dj.mean()); dJ[oi, gi] = m
            if m > best["val"]:
                best = {"val": m, "orient": on, "gap": k, "cc": cc, "dj_seq": dj}
    return {"dJ": dJ, "orients": [o[0] for o in orients], "gaps": gaps,
            "dA": float(dA_seq.mean()), "dB": float(dB_seq.mean()),
            "dA_seq": dA_seq, "dB_seq": dB_seq, "best": best,
            "wa": int(A.shape[-1]), "wb": int(B.shape[-1]), "nbg": int(len(c_bg))}

def bootstrap_maxZ(res, nboot=200, seed=0):
    """single-fold check: resample backgrounds, recompute per-arrangement mean dJ from stored per-seq,
    return fraction of boots with max-arrangement Z>4 (stability of hard call)."""
    rng = np.random.default_rng(seed)
    # reconstruct per-seq matrix only for arrangements we stored densely is expensive; approximate
    # using best arrangement's per-seq vs the arrangement-grid means (grid means are point ests).
    dJ = res["dJ"]; finite = dJ[np.isfinite(dJ)]
    if finite.size < 5: return np.nan, np.nan
    mu, sd = finite.mean(), finite.std() + 1e-9
    zmax = (np.nanmax(dJ) - mu) / sd
    # bootstrap the best arrangement mean via its per-seq deltas -> CI on how far above grid it sits
    b = res["best"]["dj_seq"]; n = len(b)
    zs = []
    for _ in range(nboot):
        bs = b[rng.integers(0, n, n)].mean()
        zs.append((bs - mu) / sd)
    return float(np.mean(np.array(zs) > 4.0)), float(zmax)

def classify(res, syn_delta=0.15, hardZ=4.0, soft_lo=20, soft_hi=150):
    dJ = res["dJ"]; dS = res["dA"] + res["dB"]
    finite = dJ[np.isfinite(dJ)]
    z = (dJ - finite.mean()) / (finite.std() + 1e-9)
    gaps = res["gaps"]; softmask = (gaps >= soft_lo) & (gaps <= soft_hi)
    soft = bool(np.nanmax(dJ[:, softmask] - dS) > syn_delta) if softmask.any() else False
    b = res["best"]
    indep = res["dA_seq"] + res["dB_seq"]
    try: wp = float(wilcoxon(b["dj_seq"], indep)[1])
    except Exception: wp = np.nan
    boot_frac, zmax = bootstrap_maxZ(res)
    return {"dS": float(dS), "dJ_opt": float(b["val"]), "delta": float(b["val"] - dS),
            "opt_orient": b["orient"], "opt_gap": b["gap"], "opt_center_center": round(b["cc"], 1),
            "maxZ": float(zmax), "hard": bool(zmax > hardZ), "hard_boot_frac": boot_frac,
            "soft": soft, "wilcoxon_p": wp, "wa": res["wa"], "wb": res["wb"], "nbg": res["nbg"]}

def model_paths(models_root, ct):
    m = f"{models_root}/models/{ct}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5"
    bg = f"{models_root}/models/{ct}/fold_0/chrombpnet/0.5/auxiliary/filtered.nonpeaks.bed"
    return m, bg

def run_ct(ct, sub, cwms, a, model=None, X=None):
    if model is None:
        mp, bgp = (a.model, a.background) if a.model else model_paths(a.models_root, ct)
        X = make_backgrounds(a.genome, bgp, a.n_seqs, seed=a.seed)
        print(f"[bg] {ct}: {X.shape[0]} background seqs", flush=True)
        model = BPNet.from_chrombpnet(mp).to(a.device).eval()
    rows = []
    for _, r in sub.iterrows():
        idA, idB = r["idA"], r["idB"]
        A = cwm_consensus_ohe(cwms[idA]); B = cwm_consensus_ohe(cwms[idB])
        res = sweep_pair(model, X, A, B, a.maxdist, same=(idA == idB), batch_size=a.batch_size, device=a.device, gap_step=a.gap_step)
        cl = classify(res)
        tag = f"{idA}__{idB}"
        np.savez_compressed(os.path.join(a.out, f"{tag}__{ct}.npz"),
                            dJ=res["dJ"], gaps=res["gaps"], orients=np.array(res["orients"]),
                            dA=res["dA"], dB=res["dB"], best_dj_seq=res["best"]["dj_seq"],
                            indep_seq=res["dA_seq"] + res["dB_seq"], opt_orient=res["best"]["orient"], opt_gap=res["best"]["gap"])
        rows.append({"idA": idA, "idB": idB, "ct": ct, **cl})
        print(f"{tag} {ct}: dJ={cl['dJ_opt']:.3f} dS={cl['dS']:.3f} d={cl['delta']:+.3f} "
              f"{cl['opt_orient']}@gap{cl['opt_gap']}(cc{cl['opt_center_center']}) Z={cl['maxZ']:.1f} "
              f"hard={cl['hard']} soft={cl['soft']} p={cl['wilcoxon_p']:.1e}", flush=True)
    return rows, model, X

def main(a):
    os.makedirs(a.out, exist_ok=True)
    cwms = np.load(a.cwms)
    pairs = pd.read_csv(a.pairs, sep="\t")
    if a.chunk is not None:                                   # job-index chunking for the full run
        idx = np.array_split(np.arange(len(pairs)), a.nchunks)[a.chunk]
        pairs = pairs.iloc[idx]
    all_rows = []
    if a.ct:                                                  # single-CT mode (candidate/null/smoke)
        rows, _, _ = run_ct(a.ct, pairs, cwms, a)
        all_rows = rows; tag = a.ct
    else:                                                     # batch mode: lazy model per CT (from manifest)
        pairs = pairs.sort_values("ct")
        for ct, sub in pairs.groupby("ct", sort=False):
            rows, _, _ = run_ct(ct, sub, cwms, a)
            all_rows += rows
        tag = f"chunk{a.chunk}" if a.chunk is not None else "all"
    pd.DataFrame(all_rows).to_csv(os.path.join(a.out, f"summary__{tag}.tsv"), sep="\t", index=False)
    print(f"[done] wrote summary__{tag}.tsv ({len(all_rows)} pairs)", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default=None, help="explicit model .h5 (single-CT mode)")
    p.add_argument("--models_root", default=None, help="results/3_single_task_models (batch mode: derive model+bg per CT)")
    p.add_argument("--cwms", required=True, help="cwms.npz (signed CWMs keyed by short_id)")
    p.add_argument("-g", "--genome", required=True)
    p.add_argument("--background", default=None, help="GC-matched negatives bed (single-CT mode)")
    p.add_argument("--pairs", required=True, help="TSV with idA,idB[,ct]")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--ct", default=None, help="single CT; omit for batch mode (ct read from manifest)")
    p.add_argument("--chunk", type=int, default=None, help="0-based chunk index for job-array splitting")
    p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("-n", "--n_seqs", type=int, default=100)
    p.add_argument("--maxdist", type=int, default=200)
    p.add_argument("--gap_step", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
