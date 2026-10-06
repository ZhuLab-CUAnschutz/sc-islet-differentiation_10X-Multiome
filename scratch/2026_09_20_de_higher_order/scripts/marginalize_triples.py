#!/usr/bin/env python
"""In-silico 3-WAY motif marginalization (triple motif synergy / higher-order TF-binding syntax)
for single-task ChromBPNet models. Extends marginalize_pairs.py from a pairwise sweep to a
third-motif flanking sweep around a locked pair.

Design:
  Given a locked pair AB at their published pairwise optimum (orientation + inner-edge gap), slide
  a third motif C on both flanks (left of A, right of B) in both orientations (forward F and
  reverse-complement R) over 0-200 bp at 1-bp step. This produces:
      2 orientations x 2 flanks x 201 gaps = 804 arrangements per (pair, third) combination
  -- the same computational cost as one pairwise sweep.

Geometry:
  The pair AB is centered at the midpoint of the 2114-bp sequence (same convention as pairwise):
      L = 2114; mid = L // 2
      comp = wa + pair_gap + wb
      sA = mid - comp // 2
      sB = sA + wa + pair_gap
  For the third motif C:
      right flank (C right of B):  sC = sB + wb + gap_C
      left  flank (C left  of A):  sC = sA - wc - gap_C
  Arrangements that place C outside the sequence (sC < 0 or sC + wc > L) are skipped.

Classification:
  The incremental metric compares the best triple arrangement to the sum of the locked pair effect
  plus the isolated single-motif C effect: delta_incr = dJ_triple_opt - (dJ_pair + dC). This
  isolates the marginal synergy contributed by the third motif beyond what the pair already provides.

Env: eugene_tools (tangermeme 1.0.3, bpnetlite 0.8.1, seqpro, torch). GPU via SLURM carter-gpu.
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
    X = torch.tensor(np.stack(seqs), dtype=torch.float32)
    return X

# ---------- prediction: chrombpnet log-count head ----------
def logcounts(model, Xe, batch_size, device):
    y = predict(model, Xe, batch_size=batch_size, device=device, verbose=False)
    yc = y[1] if isinstance(y, (list, tuple)) else y
    yc = yc.reshape(yc.shape[0], -1)
    return yc.sum(-1).cpu().numpy() if yc.shape[1] > 1 else yc.cpu().numpy().ravel()

def _mot(m):
    return torch.tensor(np.ascontiguousarray(m, dtype=np.float32))[None]

def center_insert(X, mot):
    mid = X.shape[-1] // 2; w = mot.shape[-1]; s = mid - w // 2
    return substitute(X, _mot(mot), start=s)

def model_paths(models_root, ct):
    m = f"{models_root}/models/{ct}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5"
    bg = f"{models_root}/models/{ct}/fold_0/chrombpnet/0.5/auxiliary/filtered.nonpeaks.bed"
    return m, bg

def bootstrap_maxZ(res, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    dJ = res["dJ"]; finite = dJ[np.isfinite(dJ)]
    if finite.size < 5: return np.nan, np.nan
    mu, sd = finite.mean(), finite.std() + 1e-9
    zmax = (np.nanmax(dJ) - mu) / sd
    b = res["best"]["dj_seq"]; n = len(b)
    zs = []
    for _ in range(nboot):
        bs = b[rng.integers(0, n, n)].mean()
        zs.append((bs - mu) / sd)
    return float(np.mean(np.array(zs) > 4.0)), float(zmax)

# ---------- pair geometry ----------
def pair_positions(wa, wb, pair_gap, L=2114):
    """Compute sA, sB for a pair centered in the sequence."""
    mid = L // 2
    comp = wa + pair_gap + wb
    sA = mid - comp // 2
    sB = sA + wa + pair_gap
    assert sA >= 0 and sB + wb <= L, f"pair span {comp} does not fit in {L}"
    return sA, sB

# ---------- triple sweep: slide third motif on both flanks of locked pair ----------
def sweep_triple(model, X, A, B, C, pair_orient, pair_gap, maxdist, batch_size, device, gap_step=1):
    """Sweep third motif C across both flanks of locked pair AB.

    Returns dict with dJ grid, scalar effects, and best arrangement.
    """
    n = X.shape[0]; L = X.shape[-1]
    wa, wb, wc = A.shape[-1], B.shape[-1], C.shape[-1]

    # Background and single-motif effects
    c_bg = logcounts(model, X, batch_size, device)
    dA_seq = logcounts(model, center_insert(X, A), batch_size, device) - c_bg
    dB_seq = logcounts(model, center_insert(X, B), batch_size, device) - c_bg
    dC_seq = logcounts(model, center_insert(X, C), batch_size, device) - c_bg

    # Pair at locked arrangement
    sA, sB = pair_positions(wa, wb, pair_gap, L)
    Aa = rc(A) if pair_orient[0] == "R" else A
    Bb = rc(B) if pair_orient[1] == "R" else B
    Xe_pair = substitute(substitute(X, _mot(Aa), start=sA), _mot(Bb), start=sB)
    dJ_pair_seq = logcounts(model, Xe_pair, batch_size, device) - c_bg

    # Layouts: left_CF, left_CR, right_CF, right_CR
    gaps = np.arange(0, maxdist + 1, gap_step)
    layouts = ["left_CF", "left_CR", "right_CF", "right_CR"]
    dJ = np.full((len(layouts), len(gaps)), np.nan)
    best = {"val": -np.inf}

    for li, layout in enumerate(layouts):
        flank = layout.split("_")[0]   # "left" or "right"
        c_orient = layout.split("_")[1]  # "CF" or "CR"
        Cc = rc(C) if c_orient == "CR" else C

        chunks, meta = [], []
        for gi, g in enumerate(gaps):
            if flank == "right":
                sC = sB + wb + int(g)
            else:  # left
                sC = sA - wc - int(g)

            # Skip if C doesn't fit
            if sC < 0 or sC + wc > L:
                continue

            # Insert all three motifs
            Xe = substitute(X, _mot(Aa), start=sA)
            Xe = substitute(Xe, _mot(Bb), start=sB)
            Xe = substitute(Xe, _mot(Cc), start=sC)
            chunks.append(Xe)

            cc_AC = abs((sA + wa / 2) - (sC + wc / 2))
            cc_BC = abs((sB + wb / 2) - (sC + wc / 2))
            meta.append((gi, int(g), int(sC), float(cc_AC), float(cc_BC)))

        if not chunks:
            continue

        cj = logcounts(model, torch.cat(chunks, 0), batch_size, device).reshape(len(meta), n)
        for row, (gi, g, sC, cc_AC, cc_BC) in enumerate(meta):
            dj = cj[row] - c_bg
            m = float(dj.mean())
            dJ[li, gi] = m
            if m > best["val"]:
                best = {"val": m, "layout": layout, "gap": g, "sC": sC,
                        "cc_AC": cc_AC, "cc_BC": cc_BC, "dj_seq": dj}

    return {"dJ": dJ, "layouts": layouts, "gaps": gaps,
            "dA": float(dA_seq.mean()), "dB": float(dB_seq.mean()),
            "dC": float(dC_seq.mean()), "dJ_pair": float(dJ_pair_seq.mean()),
            "dA_seq": dA_seq, "dB_seq": dB_seq, "dC_seq": dC_seq,
            "dJ_pair_seq": dJ_pair_seq, "best": best,
            "wa": wa, "wb": wb, "wc": wc, "nbg": int(len(c_bg))}

# ---------- classification ----------
def classify_triple(res, syn_delta=0.15, hardZ=4.0):
    """Classify a triple result. Calling uses the INCREMENTAL metric."""
    dS_full = res["dA"] + res["dB"] + res["dC"]
    dS_incr = res["dJ_pair"] + res["dC"]
    b = res["best"]

    delta_full = b["val"] - dS_full
    delta_incr = b["val"] - dS_incr

    boot_frac, zmax = bootstrap_maxZ(res)

    # Wilcoxon on incremental: best triple per-seq vs (pair_seq + C_seq)
    incr_expect = res["dJ_pair_seq"] + res["dC_seq"]
    try:
        wp = float(wilcoxon(b["dj_seq"], incr_expect)[1])
    except Exception:
        wp = np.nan

    frac_pos = float((b["dj_seq"] > incr_expect).mean())

    return {"dA": res["dA"], "dB": res["dB"], "dC": res["dC"],
            "dJ_pair": res["dJ_pair"],
            "dS_full": float(dS_full), "dS_incr": float(dS_incr),
            "dJ_triple_opt": float(b["val"]),
            "delta_full": float(delta_full), "delta_incr": float(delta_incr),
            "opt_layout": b.get("layout", ""), "opt_gap": b.get("gap", -1),
            "maxZ": float(zmax), "hard": bool(zmax > hardZ),
            "hard_boot_frac": boot_frac,
            "wilcoxon_p": wp, "frac_pos": frac_pos,
            "wa": res["wa"], "wb": res["wb"], "wc": res["wc"],
            "nbg": res["nbg"]}

# ---------- per-CT sweep ----------
def run_ct(ct, sub, cwms, a, model=None, X=None):
    if model is None:
        mp, bgp = (a.model, a.background) if a.model else model_paths(a.models_root, ct)
        X = make_backgrounds(a.genome, bgp, a.n_seqs, seed=a.seed)
        print(f"[bg] {ct}: {X.shape[0]} background seqs", flush=True)
        model = BPNet.from_chrombpnet(mp).to(a.device).eval()
    rows = []
    for idx, r in sub.iterrows():
        idA, idB, idC = r["idA"], r["idB"], r["idC"]
        pair_orient = r["anchor_orient"]  # e.g. "FF", "FR", "RF", "RR"
        pair_gap = int(r["anchor_gap"])

        A = cwm_consensus_ohe(cwms[idA])
        B = cwm_consensus_ohe(cwms[idB])
        C = cwm_consensus_ohe(cwms[idC])

        # Assert widths match manifest if provided
        if "wa" in r and not pd.isna(r["wa"]):
            assert A.shape[-1] == int(r["wa"]), f"{idA}: CWM width {A.shape[-1]} != manifest wa {int(r['wa'])}"
        if "wb" in r and not pd.isna(r["wb"]):
            assert B.shape[-1] == int(r["wb"]), f"{idB}: CWM width {B.shape[-1]} != manifest wb {int(r['wb'])}"
        if "wc" in r and not pd.isna(r["wc"]):
            assert C.shape[-1] == int(r["wc"]), f"{idC}: CWM width {C.shape[-1]} != manifest wc {int(r['wc'])}"

        res = sweep_triple(model, X, A, B, C,
                           pair_orient=pair_orient, pair_gap=pair_gap,
                           maxdist=a.maxdist, batch_size=a.batch_size,
                           device=a.device, gap_step=a.gap_step)
        cl = classify_triple(res)
        tag = f"{idA}__{idB}__{idC}"
        np.savez_compressed(os.path.join(a.out, f"{tag}__{ct}.npz"),
                            dJ=res["dJ"], gaps=res["gaps"],
                            layouts=np.array(res["layouts"]),
                            dA=res["dA"], dB=res["dB"], dC=res["dC"],
                            dJ_pair=res["dJ_pair"],
                            best_dj_seq=res["best"]["dj_seq"],
                            pair_expect_seq=res["dJ_pair_seq"] + res["dC_seq"],
                            opt_layout=res["best"].get("layout", ""),
                            opt_gap=res["best"].get("gap", -1),
                            anchor_orient=pair_orient, anchor_gap=pair_gap)
        rows.append({"idA": idA, "idB": idB, "idC": idC, "ct": ct,
                      "anchor_orient": pair_orient, "anchor_gap": pair_gap, **cl})
        print(f"{tag} {ct}: dJ_triple={cl['dJ_triple_opt']:.3f} dS_incr={cl['dS_incr']:.3f} "
              f"d_incr={cl['delta_incr']:+.3f} {cl['opt_layout']}@gap{cl['opt_gap']} "
              f"Z={cl['maxZ']:.1f} hard={cl['hard']} p={cl['wilcoxon_p']:.1e} "
              f"frac+={cl['frac_pos']:.2f}", flush=True)
    return rows, model, X

# ---------- main ----------
def main(a):
    os.makedirs(a.out, exist_ok=True)
    cwms = np.load(a.cwms)
    triples = pd.read_csv(a.triples, sep="\t")
    all_rows = []
    if a.ct:                                                    # single-CT mode
        rows, _, _ = run_ct(a.ct, triples, cwms, a)
        all_rows = rows; tag = a.ct
    else:                                                       # batch mode: lazy model per CT
        triples = triples.sort_values("ct")
        for ct, sub in triples.groupby("ct", sort=False):
            rows, _, _ = run_ct(ct, sub, cwms, a)
            all_rows += rows
        tag = "all"
    pd.DataFrame(all_rows).to_csv(os.path.join(a.out, f"summary__{tag}.tsv"), sep="\t", index=False)
    print(f"[done] wrote summary__{tag}.tsv ({len(all_rows)} triples)", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="3-way motif synergy sweep: slide third motif C around locked pair AB")
    p.add_argument("--models_root", default=None,
                   help="results/3_single_task_models (batch mode: derive model+bg per CT)")
    p.add_argument("-m", "--model", default=None,
                   help="explicit model .h5 (single-CT mode)")
    p.add_argument("--background", default=None,
                   help="explicit background bed (single-CT mode)")
    p.add_argument("--cwms", required=True,
                   help="cwms.npz or motifs_de4.npz (signed CWMs keyed by short_id)")
    p.add_argument("-g", "--genome", required=True,
                   help="/path/to/hg38.fa")
    p.add_argument("--triples", required=True,
                   help="TSV with idA,idB,idC,anchor_orient,anchor_gap,wa,wb,wc[,ct]")
    p.add_argument("-o", "--out", required=True,
                   help="output directory")
    p.add_argument("--ct", default=None,
                   help="single CT; omit for batch mode (ct read from manifest)")
    p.add_argument("-n", "--n_seqs", type=int, default=100)
    p.add_argument("--maxdist", type=int, default=200)
    p.add_argument("--gap_step", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
