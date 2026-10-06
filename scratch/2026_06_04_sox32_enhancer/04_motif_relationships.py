#!/usr/bin/env python
"""
04_motif_relationships.py — how EOMES / T / FOXH1 relate to the Vierstra motif
archetypes. Two outputs:

  (a) figure: each TF's HOCOMOCO H12 logo next to its NEAREST Vierstra consensus
      archetype (TOMTOM), + a markdown relationship summary.
  (b) browsable HTML: for each TF, the archetype CLUSTER(s) it belongs to and the
      full DBD FAMILY, with every member-motif logo (like db_viewer, focused).

Run under eugene_tools.
"""
import os, sys, html as _html
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import plot_locus as pl
from tangermeme.annotate import read_meme, tomtom

sys.path.insert(0, "/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/scratch/2026_05_25_motif_review/lib")
from potential_tf import load_vierstra_db

FIG, OUT = "figures", "outputs"
hoco = read_meme(config.HOCOMOCO_MEME)
vcons = read_meme(config.VIERSTRA_CONSENSUS_MEME)
vmot = read_meme(config.VIERSTRA_MOTIFS_MEME)
db = load_vierstra_db(config.VIERSTRA_XLSX)
clusters, motifs = db["clusters"], db["motifs"]

# nearest Vierstra consensus archetype per TF (TOMTOM)
cons_names = list(vcons); cons_pwms = [np.asarray(vcons[k], float) for k in cons_names]
tf_pwms = [np.asarray(hoco[h], float) for h in config.NAMED_TF.values()]
p, *_ = tomtom(tf_pwms, cons_pwms)
nearest = {tf: (cons_names[int(np.argmin(p[i]))], float(p[i].min()))
           for i, tf in enumerate(config.NAMED_TF)}

# ---- (a) figure + markdown ----
n = len(config.NAMED_TF)
fig, axes = plt.subplots(n, 2, figsize=(13, 2.3 * n))
md = ["# EOMES / T / FOXH1 — Vierstra archetype relationships\n"]
for i, (tf, hid) in enumerate(config.NAMED_TF.items()):
    pl.plot_logo(pl.ic_scale(hoco[hid]), ax=axes[i, 0]); axes[i, 0].set_yticks([])
    axes[i, 0].set_title(f"{tf}  ({hid})", fontsize=10)
    vname, vp = nearest[tf]
    pl.plot_logo(pl.ic_scale(vcons[vname]), ax=axes[i, 1]); axes[i, 1].set_yticks([])
    axes[i, 1].set_title(f"nearest Vierstra: {vname.split()[0]}  (p={vp:.1e})", fontsize=9)
    hits = motifs[motifs["Motif"].astype(str).str.contains(config.NAMED_TF_VIERSTRA_QUERY[tf],
                                                            case=False, na=False, regex=True)]
    md.append(f"\n## {tf} ({hid})\n- Nearest Vierstra consensus: **{vname.split()[0]}** (p={vp:.1e})")
    for cid in sorted(hits["Cluster_ID"].unique()):
        cr = clusters[clusters.Cluster_ID == cid].iloc[0]
        mem = motifs[motifs.Cluster_ID == cid]["Motif"].tolist()
        md.append(f"- xlsx archetype **{cr['Name']}** (cluster {cid}, DBD={cr['DBD']}, "
                  f"{len(mem)} members): {', '.join(mem[:8])}{'...' if len(mem) > 8 else ''}")
fig.suptitle("Collaborator TFs (left: HOCOMOCO H12) vs nearest Vierstra consensus archetype (right)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/sox32_TF_vierstra_relationships.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/sox32_TF_vierstra_relationships.png", dpi=150, bbox_inches="tight"); plt.close(fig)
md.append("\n## Summary\n- EOMES and T (TBXT) are both **T-box (TBX)** DBD (shared AGGTGTGA core; "
          "Vierstra resolves them as separate consensus archetypes TBX/EOMES vs TBX/T).\n"
          "- FOXH1 is **forkhead (FOX)** (TGTGGATT core), a distinct archetype.\n"
          "- So T-box contribution footprints can be EOMES or T; FOXH1 is separable.")
open(f"{OUT}/sox32_TF_vierstra_relationships.md", "w").write("\n".join(md) + "\n")
print("[wrote] TF-vierstra figure + md")

# ---- (b) browsable cluster/family HTML ----
logo_cache = {}
def member_logo(name):
    if name not in logo_cache:
        logo_cache[name] = pl.pwm_logo_b64(vmot[name]) if name in vmot else None
    return logo_cache[name]

def _member_cell(m):
    logo = member_logo(m)
    img = f'<img src="{logo}">' if logo else "?"
    tf = _html.escape(m.split("_")[0])
    return (f'<div class="member"><div class="member-tf">{tf}</div>{img}'
            f'<div class="member-name">{_html.escape(m)}</div></div>')


def cluster_block(cid, hl=False):
    cr = clusters[clusters.Cluster_ID == cid].iloc[0]
    mem = motifs[motifs.Cluster_ID == cid]["Motif"].tolist()
    cells = "".join(_member_cell(m) for m in mem)
    return (f'<details class="cluster {"hl" if hl else ""}" {"open" if hl else ""}>'
            f'<summary><b>{_html.escape(str(cr["Name"]))}</b> (cluster {cid}, '
            f'DBD={_html.escape(str(cr["DBD"]))}, {len(mem)} members)</summary>'
            f'<div class="members">{cells}</div></details>')

sections = []
for tf, hid in config.NAMED_TF.items():
    own = sorted(motifs[motifs["Motif"].astype(str).str.contains(
        config.NAMED_TF_VIERSTRA_QUERY[tf], case=False, na=False, regex=True)]["Cluster_ID"].unique())
    dbd = clusters[clusters.Cluster_ID.isin(own)]["DBD"].iloc[0]
    fam = sorted(clusters[clusters.DBD == dbd]["Cluster_ID"].tolist())
    sections.append(
        f'<section><h2>{_html.escape(tf)}</h2>'
        f'<div class="query"><div class="rep-label">query: {_html.escape(hid)}</div>'
        f'<img src="{pl.pwm_logo_b64(hoco[hid], w=2.4, h=0.7)}">'
        f'<div class="rep-label">nearest Vierstra: <b>{_html.escape(nearest[tf][0].split()[0])}</b></div></div>'
        f'<h3>Archetype cluster(s) containing {_html.escape(tf)} (DBD {_html.escape(dbd)})</h3>'
        + "".join(cluster_block(c, hl=True) for c in own)
        + f'<h3>Full {_html.escape(dbd)} family ({len(fam)} clusters)</h3>'
        + "".join(cluster_block(c, hl=(c in own)) for c in fam) + "</section>")

css = ("body{font-family:-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:16px;background:#fafafa}"
       "h2{margin-top:28px;border-bottom:2px solid #888;padding-bottom:4px}h3{font-size:14px;color:#444}"
       ".query{display:flex;gap:18px;align-items:center;background:#fff8e1;padding:8px 12px;border-radius:6px}"
       ".rep-label{font-size:12px;color:#555}.cluster{background:#fff;margin:6px 0;padding:6px 12px;border:1px solid #e1e1e1;border-radius:6px}"
       ".cluster.hl{border:2px solid #d95f02;background:#fff7f0}.cluster summary{cursor:pointer;font-size:13px}"
       ".members{display:flex;flex-wrap:wrap;margin-top:6px}.member{margin:4px 6px;padding:4px;background:#fafafa;"
       "border:1px solid #eee;border-radius:4px;text-align:center;width:150px}.member img{width:150px}"
       ".member-name{font-family:monospace;font-size:9px;color:#666;word-break:break-all}"
       ".member-tf{font-family:monospace;font-size:10px;color:#07a;font-weight:600}")
open(f"{FIG}/sox32_vierstra_cluster_browser.html", "w").write(
    f"<!doctype html><html><head><meta charset='utf-8'><title>sox32 TFs — Vierstra clusters</title>"
    f"<style>{css}</style></head><body><h1>EOMES / T / FOXH1 — Vierstra archetype clusters &amp; families</h1>"
    f"{''.join(sections)}</body></html>")
print(f"[wrote] {FIG}/sox32_vierstra_cluster_browser.html ({len(logo_cache)} member logos)")
print("DONE")
