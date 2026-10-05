#!/usr/bin/env python
"""Build the constituent-analysis inputs (query.meme + manifest.json) for the
NEW single-task WITH-NEGATIVES compendium, by reading the MotifCompendium .mc objects.

PWMs come from `export_compendium_meme(..., inverse_ic=True)` — the pipeline's own
TomTom-compatible PPM export (rows sum to 1). (Do NOT use get_standard_motif_stack(): it
returns an IC/contribution representation with negative values, which MEME/TomTom reject.)

- 08_final_averages.mc -> one consensus PPM per final cluster (76).
- 07_final_clusters.mc -> constituents, grouped by `cluster_final` id (727).
Writes consensus + constituent PPMs for every MULTI-constituent cluster (>=2) into one MEME,
plus manifest.json. Names are filesystem-safe and self-consistent across MEME/marg/tomtom/render.

This is the `constituent` motif set (751 motifs). Query written to
.../marginalization/constituent/query.meme; consumed by marginalize_global.sh.
"""
from __future__ import annotations
import json, re, tempfile
from pathlib import Path
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import MotifCompendium
import MotifCompendium.utils.analysis as ua

RESULTS = Path("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models")
MC = RESULTS / "motifs" / "motif_compendium_cluster"
MARG = RESULTS / "marginalization"      # top-level, split by motif set
INP = MARG / "constituent"      # query.meme + manifest.json land here
HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
          "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
MIN_IC = 0.2


def parse_ppm_meme(path):
    """Return {motif_name: np.ndarray(w,4)} from a MEME file."""
    out, name, rows, on = {}, None, [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("MOTIF "):
            if name is not None:
                out[name] = np.asarray(rows, float)
            name = s[6:].strip(); rows = []; on = False
        elif s.startswith("letter-probability"):
            on = True
        elif on:
            v = s.split()
            if len(v) == 4:
                try:
                    rows.append([float(x) for x in v])
                except ValueError:
                    on = False
            else:
                on = False
    if name is not None:
        out[name] = np.asarray(rows, float)
    return out


def ic_trim(prob):
    p = np.clip(prob, 1e-9, 1.0)
    ic = 2.0 + (p * np.log2(p)).sum(1)
    keep = np.where(ic >= MIN_IC)[0]
    return prob if len(keep) == 0 else prob[keep[0]: keep[-1] + 1]


def block(name, prob):
    prob = np.clip(np.asarray(prob, float), 0.0, None)   # kill any fp negatives
    prob = ic_trim(prob)
    rs = prob.sum(1, keepdims=True)
    zero = (rs[:, 0] == 0)
    if zero.any():                                       # all-zero rows -> uniform
        prob[zero] = 0.25; rs[zero] = 1.0
    prob = prob / rs
    out = [f"MOTIF {name}", f"letter-probability matrix: alength= 4 w= {prob.shape[0]} nsites= 200"]
    out += ["\t".join(f"{x:.6f}" for x in r) for r in prob]
    out.append("URL none")
    return "\n".join(out) + "\n\n"


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def main():
    cons_mc = MotifCompendium.load(str(MC / "08_final_averages.mc"))   # 76 consensus
    con_mc = MotifCompendium.load(str(MC / "07_final_clusters.mc"))    # 727 constituents
    cons_md = cons_mc.metadata.reset_index(drop=True)
    con_md = con_mc.metadata.reset_index(drop=True)

    # Export TomTom-compatible PPMs (inverse_ic=True), then parse by mc "name".
    tmp = Path(tempfile.mkdtemp())
    ua.export_compendium_meme(cons_mc, str(tmp / "cons.meme"), "name", inverse_ic=True)
    ua.export_compendium_meme(con_mc, str(tmp / "con.meme"), "name", inverse_ic=True)
    cons_ppm = parse_ppm_meme(tmp / "cons.meme")
    con_ppm = parse_ppm_meme(tmp / "con.meme")

    cons_md["cf"] = cons_md["name"].str.extract(r"#(\d+)$").astype(int)
    order = cons_md.sort_values("total_seqlets", ascending=False)
    order = order.sort_values("posneg", key=lambda s: s.map({"pos": 0, "neg": 1}), kind="stable")
    sid_map = {row["cf"]: f"ST{n+1:02d}" for n, (_, row) in enumerate(order.iterrows())}

    combined = [HEADER]; manifest = {}; n_motifs = 0; selected = 0
    for _, crow in cons_md.iterrows():
        cf = int(crow["cf"])
        members = con_md[con_md["cluster_final"] == cf]
        if len(members) < 2:
            continue
        selected += 1
        cl = f"cl{cf}"
        cons_name = f"{cl}_consensus"
        combined.append(block(cons_name, cons_ppm[crow["name"]])); n_motifs += 1
        constituents = []
        for j, (_, mrow) in enumerate(members.iterrows(), 1):
            ct = str(mrow["celltype"]).replace(".counts", "")
            nm = f"{cl}_{safe(ct)}_{j}_{mrow['posneg']}"
            combined.append(block(nm, con_ppm[mrow["name"]])); n_motifs += 1
            constituents.append({"name": nm, "celltype": ct})
        tf0 = str(crow["annotation_final_mc_name0"]).split(",")[0] if crow.get("annotation_final_mc_name0") else "?"
        manifest[cl] = {
            "short_id": sid_map[cf], "tf_name": tf0, "quality": "", "category": "",
            "posneg": crow["posneg"], "n_patterns": int(crow["num_final_constituents"]),
            "consensus_name": cons_name, "constituents": constituents,
        }

    INP.mkdir(parents=True, exist_ok=True)
    qpath = INP / "query.meme"
    qpath.write_text("".join(combined))
    (INP / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # validate: no negatives, every row sums ~1
    chk = parse_ppm_meme(qpath)
    bad = [n for n, m in chk.items() if (m < 0).any() or not np.allclose(m.sum(1), 1.0, atol=1e-3)]
    print(f"multi-constituent clusters: {selected}/{len(cons_md)}  "
          f"(pos {sum(1 for v in manifest.values() if v['posneg']=='pos')}, "
          f"neg {sum(1 for v in manifest.values() if v['posneg']=='neg')})")
    print(f"motifs written: {n_motifs}  | MOTIF lines: {sum(1 for l in open(qpath) if l.startswith('MOTIF '))}")
    print(f"VALIDATION — motifs with negative/non-normalized rows: {len(bad)} (should be 0)")
    print(f"Wrote {qpath} and {INP / 'manifest.json'}")


if __name__ == "__main__":
    main()
