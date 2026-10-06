#!/usr/bin/env python
"""In-silico marginalization that KEEPS the per-base profile head.

The published synergy pipeline (2026_07_25_syntax/scripts/marginalize_pairs.py) discards
`y[0]` on every forward pass and stores only counts-head scalars, so no prediction track can
be reconstructed from anything on disk. This script re-runs the same marginalizations and
saves the profile.

Conditions per cell type, per background set:
  * bg                       -- unedited background
  * single__{ST}             -- one motif, CENTERED, FORWARD (the published dA/dB convention)
  * pair__{A}__{B}__A        -- motif A alone, at its in-pair position AND orientation
  * pair__{A}__{B}__B        -- motif B alone, likewise
  * pair__{A}__{B}__AB       -- both, at that (pair, cell type)'s published opt_orient/opt_gap

Two background sets are computed in the same pass:
  * own    -- this cell type's own filtered.nonpeaks.bed (reproduces the published scalars)
  * shared -- one fixed cell type's negatives used against all 22 models (cross-CT comparable)

Saved per cell type: the raw (nbg, out_len) log-softmax profiles and (nbg,) log-counts for
every condition, so all averaging / renormalization choices stay re-decidable off-GPU.

Env: eugene_tools (tangermeme, bpnetlite, torch). GPU via SLURM carter-gpu only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyfaidx
import torch
from bpnetlite.bpnet import BPNet
from tangermeme.ersatz import substitute
from tangermeme.predict import predict

BED3 = ["chrom", "start", "end"]
SEQ_LEN = 2114


# ---------- motif consensus (verbatim from marginalize_pairs.py) ----------
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
    ohe = np.zeros((len(idx), 4), np.float32)
    ohe[np.arange(len(idx)), idx] = 1.0
    return ohe.T  # (4,w)


def rc(ohe):
    out = ohe[:, ::-1].copy()
    return out[[3, 2, 1, 0], :]


# ---------- backgrounds ----------
def _ohe_str(s, width):
    MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((width, 4), np.float32)
    for i, ch in enumerate(s):
        j = MAP.get(ch)
        if j is None:
            return None
        arr[i, j] = 1.0
    return arr.T


def _dinuc_shuffle(ohe, rng):
    """k=2 (dinucleotide-preserving) shuffle of a (4,L) one-hot, via an Eulerian walk."""
    seq = "ACGT"
    s = "".join(seq[i] for i in ohe.argmax(0))
    # bucket successors by the preceding base, then rebuild an Eulerian path
    edges = {}
    for a, b in zip(s[:-1], s[1:]):
        edges.setdefault(a, []).append(b)
    last = s[-1]
    for a in edges:
        tail = None
        if a == last and edges[a]:
            # hold one edge back so the walk can terminate at `last`
            tail = edges[a].pop()
        rng.shuffle(edges[a])
        if tail is not None:
            edges[a].append(tail)
    cur, out = s[0], [s[0]]
    for _ in range(len(s) - 1):
        nxt = edges[cur].pop(0)
        out.append(nxt)
        cur = nxt
    return _ohe_str("".join(out), len(s))


def make_backgrounds(genome_fasta, bed_path, n_seqs, width=SEQ_LEN, seed=1234, shuffle=False):
    """Sequences centered on `bed_path` intervals. `shuffle` -> dinucleotide-shuffle each one.

    The non-shuffled / filtered.nonpeaks.bed path is byte-identical to the published pipeline.
    """
    genome = pyfaidx.Fasta(genome_fasta)
    bed = pd.read_csv(bed_path, sep="\t", header=None).iloc[:, :3]
    bed.columns = BED3
    bed["mid"] = (bed["start"] + bed["end"]) // 2
    bed["s"] = bed["mid"] - width // 2
    bed["e"] = bed["mid"] + width // 2
    bed = bed[bed["s"] >= 0]
    sub = bed.sample(n=min(n_seqs, len(bed)), random_state=seed)
    rng = np.random.default_rng(seed)
    seqs = []
    for _, r in sub.iterrows():
        s = genome[r["chrom"]][int(r["s"]):int(r["e"])].seq.upper()
        if len(s) != width:
            continue
        arr = _ohe_str(s, width)
        if arr is None:
            continue
        seqs.append(_dinuc_shuffle(arr, rng) if shuffle else arr)
    return torch.tensor(np.stack(seqs), dtype=torch.float32)  # (n,4,width)


# ---------- prediction: KEEP the profile head ----------
def profiles_and_counts(model, Xe, batch_size, device):
    """-> (logq (n,out_len) float32 log-softmax profile, c (n,) log-counts)."""
    y = predict(model, Xe, batch_size=batch_size, device=device, verbose=False)
    assert isinstance(y, (list, tuple)) and len(y) == 2, f"unexpected predict output: {type(y)}"
    prof, cnt = y[0], y[1]
    assert prof.shape[1] == 1, f"profile head has {prof.shape[1]} tracks, expected 1"
    assert cnt.reshape(cnt.shape[0], -1).shape[1] == 1, "counts head is not scalar; sum() would be wrong for logs"
    logq = torch.log_softmax(prof.reshape(prof.shape[0], -1).double(), dim=-1)
    c = cnt.reshape(-1).double()
    return logq.cpu().numpy().astype(np.float32), c.cpu().numpy().astype(np.float64)


def _mot(m):
    return torch.tensor(np.ascontiguousarray(m, dtype=np.float32))[None]  # (1,4,w)


def center_insert(X, mot):
    mid = X.shape[-1] // 2
    w = mot.shape[-1]
    return substitute(X, _mot(mot), start=mid - w // 2)


def pair_starts(wa, wb, gap, length=SEQ_LEN):
    """Published geometry: A then B straddling the center with inner-edge `gap`."""
    mid = length // 2
    comp = wa + gap + wb
    sA = mid - comp // 2
    sB = sA + wa + gap
    assert sA >= 0 and sB + wb <= length, f"pair span {comp} does not fit in {length}"
    assert sB >= sA + wa, "second insert would clobber first"
    return sA, sB, (sB + wb / 2) - (sA + wa / 2)


def model_paths(models_root, ct):
    base = f"{models_root}/models/{ct}/fold_0/chrombpnet/0.5"
    return (f"{base}/models/chrombpnet_nobias.h5",
            f"{base}/auxiliary/filtered.nonpeaks.bed",
            f"{base}/auxiliary/filtered.peaks.bed")


# ---------- per cell type ----------
def triple_positions(wa, wb, wc, pair_gap, opt_layout, opt_gap_C, length=SEQ_LEN):
    """Compute sA, sB, sC for a triple given the pair geometry and third-motif layout."""
    sA, sB, _ = pair_starts(wa, wb, pair_gap, length)
    flank, c_orient = opt_layout.split("_")[0], opt_layout.split("_")[1]
    if flank == "right":
        sC = sB + wb + opt_gap_C
    else:
        sC = sA - wc - opt_gap_C
    assert 0 <= sC and sC + wc <= length, \
        f"C at sC={sC} wc={wc} falls outside [0, {length})"
    return sA, sB, sC, c_orient


def run_ct(ct, model, X, cwms, singles, pairs_ct, a, tag, triples_ct=None):
    """Every condition for one cell type against one background set. Returns dict of arrays."""
    out = {}
    X0 = X.clone()

    logq, c = profiles_and_counts(model, X, a.batch_size, a.device)
    out["bg"] = (logq, c)
    c_bg = c

    for st in singles:
        m = cwm_consensus_ohe(cwms[st])
        out[f"single__{st}"] = profiles_and_counts(model, center_insert(X, m), a.batch_size, a.device)

    for _, r in pairs_ct.iterrows():
        idA, idB = r["idA"], r["idB"]
        A = cwm_consensus_ohe(cwms[idA])
        B = cwm_consensus_ohe(cwms[idB])
        assert A.shape[-1] == int(r["wa"]), f"{idA} width {A.shape[-1]} != published {r['wa']}"
        assert B.shape[-1] == int(r["wb"]), f"{idB} width {B.shape[-1]} != published {r['wb']}"
        orient, gap = str(r["opt_orient"]), int(r["opt_gap"])
        Aa = rc(A) if orient[0] == "R" else A
        Bb = rc(B) if orient[1] == "R" else B
        sA, sB, cc = pair_starts(A.shape[-1], B.shape[-1], gap)
        assert abs(cc - float(r["opt_center_center"])) < 0.51, \
            f"center-center {cc} != published {r['opt_center_center']} for {idA}/{idB}/{ct}"

        XA = substitute(X, _mot(Aa), start=sA)
        XB = substitute(X, _mot(Bb), start=sB)
        XAB = substitute(substitute(X, _mot(Aa), start=sA), _mot(Bb), start=sB)
        key = f"pair__{idA}__{idB}"
        out[f"{key}__A"] = profiles_and_counts(model, XA, a.batch_size, a.device)
        out[f"{key}__B"] = profiles_and_counts(model, XB, a.batch_size, a.device)
        out[f"{key}__AB"] = profiles_and_counts(model, XAB, a.batch_size, a.device)
        out[f"{key}__meta"] = dict(idA=idA, idB=idB, orient=orient, gap=gap,
                                   sA=int(sA), sB=int(sB), wa=int(A.shape[-1]), wb=int(B.shape[-1]),
                                   center_center=float(cc))

    if triples_ct is not None:
        for _, r in triples_ct.iterrows():
            idA, idB, idC = r["idA"], r["idB"], r["idC"]
            A = cwm_consensus_ohe(cwms[idA])
            B = cwm_consensus_ohe(cwms[idB])
            C = cwm_consensus_ohe(cwms[idC])
            pair_orient = str(r["anchor_orient"])
            pair_gap = int(r["anchor_gap"])
            opt_layout = str(r["opt_layout"])
            opt_gap_C = int(r["opt_gap"])

            Aa = rc(A) if pair_orient[0] == "R" else A
            Bb = rc(B) if pair_orient[1] == "R" else B
            sA, sB, sC, c_orient = triple_positions(
                A.shape[-1], B.shape[-1], C.shape[-1],
                pair_gap, opt_layout, opt_gap_C)
            Cc = rc(C) if c_orient == "CR" else C

            XA = substitute(X, _mot(Aa), start=sA)
            XB = substitute(X, _mot(Bb), start=sB)
            XC = substitute(X, _mot(Cc), start=sC)
            XABC = substitute(substitute(substitute(X, _mot(Aa), start=sA),
                              _mot(Bb), start=sB), _mot(Cc), start=sC)
            key = f"triple__{idA}__{idB}__{idC}"
            out[f"{key}__A"] = profiles_and_counts(model, XA, a.batch_size, a.device)
            out[f"{key}__B"] = profiles_and_counts(model, XB, a.batch_size, a.device)
            out[f"{key}__C"] = profiles_and_counts(model, XC, a.batch_size, a.device)
            out[f"{key}__ABC"] = profiles_and_counts(model, XABC, a.batch_size, a.device)
            out[f"{key}__meta"] = dict(
                idA=idA, idB=idB, idC=idC,
                pair_orient=pair_orient, pair_gap=pair_gap,
                opt_layout=opt_layout, opt_gap_C=opt_gap_C,
                sA=int(sA), sB=int(sB), sC=int(sC),
                wa=int(A.shape[-1]), wb=int(B.shape[-1]), wc=int(C.shape[-1]))
            dJ_abc = float((profiles_and_counts(model, XABC, a.batch_size, a.device)[1] - c_bg).mean())
            print(f"  [{tag}] triple {idA}x{idB}x{idC}: dJ_ABC={dJ_abc:+.3f}", flush=True)

    assert torch.equal(X, X0), "tangermeme.substitute mutated the background tensor in place"

    dJ_obs = {k: float((v[1] - c_bg).mean()) for k, v in out.items()
              if k.startswith("pair__") and k.endswith("__AB")}
    print(f"  [{tag}] bg logcounts mean={c_bg.mean():.3f}  " +
          "  ".join(f"{k.split('__')[1]}x{k.split('__')[2]}:dJ={v:+.3f}" for k, v in dJ_obs.items()),
          flush=True)
    return out


def pack(out):
    """dict of (logq, c) / meta -> flat npz-able dict."""
    flat = {}
    for k, v in out.items():
        if k.endswith("__meta"):
            flat[k] = json.dumps(v)
        else:
            flat[f"{k}::logq"] = v[0]
            flat[f"{k}::c"] = v[1]
    return flat


def main(a):
    os.makedirs(a.out, exist_ok=True)
    cwms = np.load(a.cwms)
    singles_tsv = pd.read_csv(a.singles, sep="\t")
    pairs = pd.read_csv(a.pairs, sep="\t")
    triples = pd.read_csv(a.triples, sep="\t") if a.triples else None

    singles = list(dict.fromkeys(
        singles_tsv["short_id"].tolist() + pairs["idA"].tolist() + pairs["idB"].tolist()))
    if triples is not None:
        singles = list(dict.fromkeys(
            singles + triples["idA"].tolist() + triples["idB"].tolist() + triples["idC"].tolist()))
    missing = [s for s in singles if s not in cwms.files]
    assert not missing, f"missing from cwms.npz: {missing}"
    print(f"[motifs] {len(singles)} centered-forward inserts: {singles}", flush=True)
    for s in singles:
        print(f"    {s}: consensus width {cwm_consensus_ohe(cwms[s]).shape[-1]}", flush=True)

    cts = a.cts.split(",") if a.cts else sorted(pairs["ct"].unique())
    n_triple_conds = 4 * len(triples) if triples is not None else 0
    print(f"[plan] {len(cts)} cell types x "
          f"{1 + len(singles) + 3 * len(pairs[['idA','idB']].drop_duplicates()) + n_triple_conds} conditions "
          f"x {len(a.bg_sets.split(','))} background sets, n={a.n_seqs}", flush=True)

    # shared background set: one fixed cell type's negatives, used against every model
    X_shared = None
    if "shared" in a.bg_sets:
        _, nonpk, pk = model_paths(a.models_root, a.shared_bg_ct)
        src = pk if a.background_mode == "dinuc_shuffled_peak" else nonpk
        X_shared = make_backgrounds(a.genome, src, a.n_seqs, seed=a.seed,
                                    shuffle=(a.background_mode == "dinuc_shuffled_peak"))
        print(f"[bg] shared ({a.shared_bg_ct}, {a.background_mode}): {X_shared.shape[0]} seqs", flush=True)

    for ct in cts:
        mp, nonpk, pk = model_paths(a.models_root, ct)
        pairs_ct = pairs[pairs["ct"] == ct]
        triples_ct = triples[triples["ct"] == ct] if triples is not None and "ct" in triples.columns else triples
        model = BPNet.from_chrombpnet(mp).to(a.device).eval()
        trim = (SEQ_LEN - model.trimming * 2) if hasattr(model, "trimming") else None
        print(f"[{ct}] model loaded (trimming={getattr(model, 'trimming', '?')})", flush=True)

        payload, meta = {}, {"ct": ct, "n_seqs": a.n_seqs, "seed": a.seed,
                             "background_mode": a.background_mode, "shared_bg_ct": a.shared_bg_ct}
        for bgset in a.bg_sets.split(","):
            if bgset == "own":
                src = pk if a.background_mode == "dinuc_shuffled_peak" else nonpk
                X = make_backgrounds(a.genome, src, a.n_seqs, seed=a.seed,
                                     shuffle=(a.background_mode == "dinuc_shuffled_peak"))
                print(f"[bg] own ({ct}): {X.shape[0]} seqs", flush=True)
            else:
                X = X_shared
            res = run_ct(ct, model, X, cwms, singles, pairs_ct, a, tag=bgset,
                         triples_ct=triples_ct)
            meta[f"nbg_{bgset}"] = int(X.shape[0])
            for k, v in pack(res).items():
                payload[f"{bgset}::{k}"] = v

        first = next(v for k, v in payload.items() if k.endswith("::logq"))
        meta["out_len"] = int(first.shape[-1])
        assert meta["out_len"] == 1000, f"profile output length {meta['out_len']}, expected 1000"
        payload["meta"] = json.dumps(meta)
        np.savez_compressed(os.path.join(a.out, f"tracks__{ct}.npz"), **payload)
        print(f"[{ct}] wrote tracks__{ct}.npz ({meta['out_len']} bp out, "
              f"{sum(1 for k in payload if k.endswith('::logq'))} conditions)", flush=True)

        del model
        torch.cuda.empty_cache()

    print("[done]", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models_root", required=True, help="results/3_single_task_models")
    p.add_argument("--cwms", required=True, help="cwms.npz (signed CWMs keyed by short_id)")
    p.add_argument("-g", "--genome", required=True)
    p.add_argument("--singles", required=True, help="TSV with short_id (illustration TFs)")
    p.add_argument("--pairs", required=True, help="TSV with idA,idB,ct,wa,wb,opt_orient,opt_gap,...")
    p.add_argument("--triples", default=None,
                   help="TSV with idA,idB,idC,anchor_orient,anchor_gap,opt_layout,opt_gap,wa,wb,wc[,ct]")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--cts", default=None, help="comma-separated cell types; default = all in --pairs")
    p.add_argument("--bg_sets", default="own,shared", help="comma-separated subset of own,shared")
    p.add_argument("--shared_bg_ct", default="DE")
    p.add_argument("--background_mode", default="nonpeak", choices=["nonpeak", "dinuc_shuffled_peak"])
    p.add_argument("-n", "--n_seqs", type=int, default=32, help="32 = the published first-pass nbg")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
