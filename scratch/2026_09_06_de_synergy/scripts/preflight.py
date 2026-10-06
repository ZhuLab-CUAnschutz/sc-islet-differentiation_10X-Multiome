#!/usr/bin/env python
"""Cluster-side preflight (CPU, seconds): push motifs_de4.npz through the REAL cwm_consensus_ohe and
assert the catalog inserts are the strings every prior synergy run used; check model/bg/genome paths."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from marginalize_pairs import cwm_consensus_ohe, model_paths
WD = os.path.join(HERE, ".."); MROOT = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models"
EXPECT = {"ST18": "TTCACACC", "ST36_sub1": "GGTGCTAATCAG", "ST66": "AGAATGC"}
m = np.load(os.path.join(WD, "inputs", "motifs_de4.npz")); ins = pd.read_csv(os.path.join(WD, "inputs", "inserts.tsv"), sep="\t").set_index("id")
ok = True
for k in m.files:
    ohe = cwm_consensus_ohe(m[k]); s = "".join("ACGT"[i] for i in ohe.argmax(0))
    exp = EXPECT.get(k, ins.loc[k, "consensus"]); flag = "OK" if s == exp else "MISMATCH"; ok &= s == exp
    print(f"{k:10s} w={ohe.shape[1]:2d} {s:14s} expected {exp:14s} {flag}")
pairs = pd.read_csv(os.path.join(WD, "inputs", "pairs_de4.tsv"), sep="\t")
assert set(pairs.idA) | set(pairs.idB) <= set(m.files), "manifest id not in npz"
print(f"manifest: {len(pairs)} pairs, ct={sorted(pairs.ct.unique())}")
mp, bg = model_paths(MROOT, "DE"); gen = "/carter/users/aklie/data/ref/genomes/hg38/hg38.fa"
for p in (mp, bg, gen):
    print(("OK " if os.path.exists(p) else "MISSING ") + p); ok &= os.path.exists(p)
sys.exit(0 if ok else 1)
