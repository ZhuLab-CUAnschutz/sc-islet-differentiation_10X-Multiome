#!/usr/bin/env python
"""Render one CWM logo per catalog motif (signed contribution, logomaker) and write them as a
{short_id: "data:image/png;base64,..."} JSON for build_report.py to embed.
"""
import io, json, base64, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker


def main(a):
    cwms = np.load(a.cwms)
    out = {}
    for k in cwms.files:
        W = cwms[k]
        W = W if W.shape[1] == 4 else W.T
        df = pd.DataFrame(W, columns=list("ACGT"))
        fig, ax = plt.subplots(figsize=(max(2.0, 0.22 * len(df)), 1.0))
        logomaker.Logo(df, ax=ax)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        out[k] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    json.dump(out, open(a.out, "w"))
    print(f"[logos] {len(out)} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cwms", required=True, help="catalog cwm/cwms.npz")
    p.add_argument("-o", "--out", required=True, help="e.g. results/6_synergy/inputs/logos_b64.json")
    p.add_argument("--dpi", type=int, default=200)
    main(p.parse_args())
