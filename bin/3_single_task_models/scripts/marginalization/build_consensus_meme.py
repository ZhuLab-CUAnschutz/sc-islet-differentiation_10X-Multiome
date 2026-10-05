#!/usr/bin/env python
"""Write a clean PPM MEME of ALL 76 cluster consensus motifs (the new single-task
compendium itself) for marginalization. This is the `consensus` motif set.
Reuses the validated block()/parser from build_inputs_from_mc.py
(export_compendium_meme inverse_ic=True + clip/normalize).

Query written to .../marginalization/mc_catalog/query.meme;
consumed by marginalize_consensus.sh.
"""
from __future__ import annotations
import re, tempfile, sys
from pathlib import Path
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import MotifCompendium
import MotifCompendium.utils.analysis as ua

sys.path.insert(0, str(Path(__file__).parent))
from build_inputs_from_mc import parse_ppm_meme, block, HEADER, MC, MARG  # reuse validated helpers

INP = MARG / "mc_catalog"      # query.meme + consensus_meta.tsv land here


def main():
    mc = MotifCompendium.load(str(MC / "08_final_averages.mc"))
    md = mc.metadata.reset_index(drop=True)
    tmp = Path(tempfile.mkdtemp())
    ua.export_compendium_meme(mc, str(tmp / "c.meme"), "name", inverse_ic=True)
    ppm = parse_ppm_meme(tmp / "c.meme")

    out = [HEADER]; rows = []
    for _, r in md.iterrows():
        cf = int(re.search(r"#(\d+)$", r["name"]).group(1))
        nm = f"cl{cf}_consensus"
        out.append(block(nm, ppm[r["name"]]))
        tf0 = str(r["annotation_final_mc_name0"]).split(",")[0] if r.get("annotation_final_mc_name0") else "?"
        rows.append((nm, cf, r["posneg"], tf0, int(r["num_final_constituents"]), int(r["total_seqlets"])))

    INP.mkdir(parents=True, exist_ok=True)
    qpath = INP / "query.meme"
    qpath.write_text("".join(out))
    import csv
    with open(INP / "consensus_meta.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["motif", "cf", "posneg", "tf", "num_final_constituents", "total_seqlets"])
        w.writerows(rows)

    chk = parse_ppm_meme(qpath)
    bad = [n for n, m in chk.items() if (m < 0).any() or not np.allclose(m.sum(1), 1.0, atol=1e-3)]
    print(f"consensus motifs written: {len(chk)} (pos {sum(1 for r in rows if r[2]=='pos')}, neg {sum(1 for r in rows if r[2]=='neg')})")
    print(f"VALIDATION bad rows: {len(bad)} (should be 0)")
    print(f"Wrote {qpath}")


if __name__ == "__main__":
    main()
