#!/usr/bin/env python
"""DeepLIFT confirmation for synergistic pairs (Liu et al. Nature 2026 QC step): verify the predicted
synergy at a pair's OPTIMAL arrangement is driven by the two INSERTED motifs, not by junction/spacer
bases (which would make the 'synergy' an artifact of an accidental composite site).

For each pair (idA,idB,ct,opt_orient,opt_gap): insert A,B at the optimal arrangement into N GC-matched
backgrounds, compute count-head attributions (tangermeme.deep_lift_shap), and measure the fraction of
total |attribution| in the central window that falls inside the two inserted motif footprints. High
fraction = genuine (model reads the inserted motifs); low = artifact -> flag. Reuses marginalize_pairs
helpers. Env eugene_tools, carter-gpu. Batch mode: reads ct from manifest, loads model per CT.
"""
import os, argparse
import numpy as np, pandas as pd, torch
from marginalize_pairs import cwm_consensus_ohe, make_backgrounds, rc, model_paths
from tangermeme.ersatz import substitute
from tangermeme.deep_lift_shap import deep_lift_shap
from bpnetlite.bpnet import BPNet

class CountHead(torch.nn.Module):
    """wrap chrombpnet BPNet so forward returns the scalar total log-count (for attribution)."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, X):
        y = self.m(X); yc = y[1] if isinstance(y, (list, tuple)) else y
        return yc.reshape(X.shape[0], -1).sum(-1, keepdim=True)

def _mot(m): return torch.tensor(np.ascontiguousarray(m, np.float32))[None]

def place(X, A, B, gap):
    """insert A then B centered at inner-edge gap; return edited X and (sA,eA,sB,eB) windows."""
    L = X.shape[-1]; mid = L // 2; wa, wb = A.shape[-1], B.shape[-1]
    comp = wa + gap + wb; sA = mid - comp // 2; sB = sA + wa + gap
    Xe = substitute(substitute(X, _mot(A), start=sA), _mot(B), start=sB)
    return Xe, (sA, sA + wa, sB, sB + wb)

def frac_in(model, Xe, win, half=100, nshuf=10, device="cuda"):
    mid = Xe.shape[-1] // 2
    attr = deep_lift_shap(model, Xe, n_shuffles=nshuf, device=device, verbose=False)  # (n,4,L)
    per = (attr * Xe.to(attr.device)).sum(1).abs().cpu().numpy()                       # (n,L) |contribution|
    sA, eA, sB, eB = win
    inside = per[:, sA:eA].sum(1) + per[:, sB:eB].sum(1)
    total = per[:, mid - half:mid + half].sum(1) + 1e-9
    return inside / total

def main(a):
    os.makedirs(a.out, exist_ok=True)
    cwms = np.load(a.cwms)
    pairs = pd.read_csv(a.pairs, sep="\t")
    if a.chunk is not None:
        pairs = pairs.iloc[np.array_split(np.arange(len(pairs)), a.nchunks)[a.chunk]]
    rows = []
    for ct, sub in pairs.sort_values("ct").groupby("ct", sort=False):
        mp, bgp = model_paths(a.models_root, ct)
        X = make_backgrounds(a.genome, bgp, a.n_seqs, seed=a.seed)
        model = CountHead(BPNet.from_chrombpnet(mp).to(a.device).eval())
        for r in sub.itertuples():
            A = cwm_consensus_ohe(cwms[r.idA]); B = cwm_consensus_ohe(cwms[r.idB])
            oo, og = str(r.opt_orient), int(r.opt_gap)
            Aa = rc(A) if oo[0] == "R" else A; Bb = rc(B) if oo[1] == "R" else B
            try:
                Xe, win = place(X, Aa, Bb, og)
                f = frac_in(model, Xe, win, nshuf=a.nshuf, device=a.device)
                rows.append({"idA": r.idA, "idB": r.idB, "ct": ct, "frac_in": round(float(f.mean()), 4),
                             "frac_in_sd": round(float(f.std()), 4), "n": len(f),
                             "wa": int(A.shape[-1]), "wb": int(B.shape[-1])})
            except Exception as e:
                rows.append({"idA": r.idA, "idB": r.idB, "ct": ct, "frac_in": np.nan, "err": str(e)[:60]})
            print(f"{r.idA}__{r.idB} {ct}: frac_in={rows[-1].get('frac_in')}", flush=True)
    tag = f"chunk{a.chunk}" if a.chunk is not None else "all"
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"deeplift__{tag}.tsv"), sep="\t", index=False)
    print(f"[done] {len(rows)} pairs -> deeplift__{tag}.tsv", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models_root", required=True); p.add_argument("--cwms", required=True)
    p.add_argument("-g", "--genome", required=True); p.add_argument("--pairs", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--chunk", type=int, default=None); p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("-n", "--n_seqs", type=int, default=20); p.add_argument("--nshuf", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234); p.add_argument("--device", default="cuda")
    main(p.parse_args())
