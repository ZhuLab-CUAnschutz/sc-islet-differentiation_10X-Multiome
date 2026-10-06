"""
fimo.py — MEME-suite FIMO helpers (build a MEME from named TF PWMs, run FIMO,
load + feature-annotate hits). Used by the FIMO scan and the contribution
trajectory analysis. MEME-suite FIMO v5.5.0 at config.MEME_FIMO_BIN.
"""
import os
import subprocess
import numpy as np
import pandas as pd
from tangermeme.annotate import read_meme

import config


def write_named_tf_meme(path):
    """Write a MEME file with the 3 named TF PWMs (FIMO-safe names) from HOCOMOCO."""
    mm = read_meme(config.HOCOMOCO_MEME)
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for label, hid in config.NAMED_TF.items():
            pwm = np.asarray(mm[hid], dtype=float)
            f.write(f"MOTIF {config.FIMO_TF_NAMES[label]}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[1]} nsites= 20 E= 0\n")
            for j in range(pwm.shape[1]):
                col = pwm[:, j] / pwm[:, j].sum()
                f.write(" " + " ".join(f"{x:.6f}" for x in col) + "\n")
            f.write("\n")
    return path


def run_fimo(fasta, outdir, thresh=1e-3):
    """Run FIMO of the 3 named TFs against `fasta`; return the hits DataFrame
    with a `feature` column (which construct piece each hit falls in)."""
    os.makedirs(outdir, exist_ok=True)
    meme = write_named_tf_meme(os.path.join(outdir, "named3.meme"))
    run = os.path.join(outdir, "named3_run")
    subprocess.run([config.MEME_FIMO_BIN, "--oc", run, "--thresh", str(thresh),
                    "--max-strand", meme, fasta], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = pd.read_csv(f"{run}/fimo.tsv", sep="\t", comment="#").dropna(subset=["start"])
    t["start"] = t["start"].astype(int)
    t["stop"] = t["stop"].astype(int)
    t["feature"] = [config.feature_of_span(s, e) for s, e in zip(t["start"], t["stop"])]
    return t.sort_values("p-value").reset_index(drop=True)


def load_hits(csv):
    return pd.read_csv(csv)
