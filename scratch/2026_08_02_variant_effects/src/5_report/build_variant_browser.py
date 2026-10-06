"""Stage 5b: interactive variant browser.

Builds a folder deliverable (`deliverable/browser/`) for tracking individual
variants: `variant_browser.html` with a sortable/filterable table of the selected
variants; clicking a row shows that variant's info, a per-cell-type effect bar
(logfc across all 22 cell types, stage-colored), and its all-cell-type figure
(predicted accessibility + attributions for every cell type) from `allct/`.

The all-cell-type figures are large, so they are referenced from the `allct/`
subfolder (keep the folder together) rather than embedded; the small effect bars
are inlined. Run after Stage 3.8 (selection) and the --all-celltypes render.

Usage:
    python src/5_report/build_variant_browser.py --config config/config.yaml
"""

import argparse
import base64
import html
import io as _io
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib

log = setup_logging(__file__)


def effect_bar_datauri(logfc_row, order, stage_of, stage_colors) -> str:
    """Inline PNG: logfc across the 22 cell types (display order), stage-colored."""
    vals = [float(logfc_row[c]) for c in order]
    colors = [stage_colors.get(stage_of[c], "0.6") for c in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.barh(range(len(order)), vals, color=colors, edgecolor="0.3", linewidth=0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlabel("logfc (alt vs ref)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pdf_to_png(pdf: Path, out_png: Path, dpi: int = 90) -> bool:
    if not pdf.exists():
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf),
                        str(Path(td) / "p")], check=True)
        pngs = list(Path(td).glob("p*.png"))
        if not pngs:
            return False
        shutil.copy2(pngs[0], out_png)
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    order = metadata.cell_types(cfg)
    stage_of = md["dev_stage"].to_dict()
    _, stage_colors = metadata.category(cfg, "dev_stage")

    sel = pd.read_csv(io.sandbox_path(cfg, "results/variants/browser_variants.tsv"), sep="\t")
    browser = io.sandbox_path(cfg, "deliverable/browser")
    if browser.exists():
        shutil.rmtree(browser)
    (browser / "allct").mkdir(parents=True, exist_ok=True)

    # per-trait logfc matrices for the effect bars
    lf = {t["key"]: pd.read_csv(io.sandbox_path(cfg, f"results/{t['key']}/celltype_vep_logfc.csv"),
                                index_col="variant_id") for t in cfg["traits"]}

    rows_html, detail_html = [], []
    for _, r in sel.iterrows():
        vid, trait = r["variant_id"], r["trait"]
        safe = vid.replace(":", "_")
        src_pdf = io.sandbox_path(cfg, f"results/variants/allct/{safe}.allct.pdf")
        has_fig = pdf_to_png(src_pdf, browser / "allct" / f"{safe}.png")
        if has_fig:  # keep the vector PDF for click-to-zoom (full resolution)
            shutil.copy2(src_pdf, browser / "allct" / f"{safe}.pdf")
        bar = effect_bar_datauri(lf[trait].loc[vid], order, stage_of, stage_colors)
        did = f"v{int(r['rank'])}"
        cells = [int(r["rank"]), html.escape(vid), trait, r["top_cell_type"],
                 f"{r['logfc']:.2f}", r["direction"], str(r["scope"]).split(":")[0],
                 r["primary_motif"], f"{r['PIP']:.3g}", "yes" if r["disrupts_hit"] else "no"]
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
        rows_html.append(f'<tr onclick="show(\'{did}\')" data-did="{did}">{tds}</tr>')

        fig_html = (f'<a href="allct/{safe}.pdf" target="_blank" title="click to open full-resolution '
                    f'(zoomable) PDF"><img class="allct" src="allct/{safe}.png" alt="all cell types"></a>'
                    f'<div class="muted">click the figure to open the full-resolution, zoomable PDF</div>'
                    if has_fig else '<p class="missing">all-cell-type figure not rendered yet</p>')
        detail_html.append(
            f'<div class="detail" id="{did}">'
            f'<h3>#{int(r["rank"])} {html.escape(vid)} <span class="muted">({r["direction"]} of '
            f'{html.escape(str(r["primary_motif"]))}; {html.escape(str(r["scope"]))}; {trait})</span></h3>'
            f'<div class="meta">top cell type <b>{r["top_cell_type"]}</b> · logfc {r["logfc"]:.2f} · '
            f'peak {r["peak_pred_counts"]:.0f} · PIP {r["PIP"]:.3g} · '
            f'disrupts hit: {html.escape(str(r["hit_motifs"]) or "none")}</div>'
            f'<div class="row"><figure><img class="bar" src="{bar}">'
            f'<figcaption>logfc across cell types</figcaption></figure></div>'
            f'<div class="row">{fig_html}</div></div>')

    thr = cfg["significance"]["pval_threshold"]
    header = "".join(f"<th onclick=\"sortT({i})\">{h}</th>" for i, h in enumerate(
        ["rank", "variant", "trait", "top cell type", "logfc", "dir", "scope", "motif (ST##:TF)", "PIP", "hit"]))
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Variant browser - 22 developmental cell types</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a}}
 header{{padding:16px 24px;border-bottom:2px solid #eee}} h1{{margin:0;font-size:20px}}
 .muted{{color:#888;font-weight:400}} .wrap{{display:flex;gap:16px;padding:16px 24px;align-items:flex-start}}
 .left{{flex:0 0 560px;position:sticky;top:12px;max-height:92vh;overflow:auto}}
 .right{{flex:1;min-width:0}}
 input{{width:100%;padding:6px 8px;font-size:14px;margin-bottom:8px}}
 table{{border-collapse:collapse;font-size:12px;width:100%}}
 th,td{{border:1px solid #e2e2e2;padding:3px 6px;text-align:left}} th{{background:#f4f4f4;cursor:pointer;position:sticky;top:0}}
 tbody tr{{cursor:pointer}} tbody tr:hover{{background:#fbf3d0}} tr.sel{{background:#ffe9a8}}
 .detail{{display:none}} .detail.active{{display:block}}
 .meta{{color:#444;margin:4px 0 10px}} .row{{margin:8px 0}}
 img.allct{{max-width:100%;border:1px solid #eee}} img.bar{{max-width:480px;border:1px solid #eee}}
 figcaption{{font-size:12px;color:#888}} .missing{{color:#c0392b}}
</style></head><body>
<header><h1>Variant browser</h1>
<div class="muted">{len(sel)} curated variants (non-CTCF; lineage-TF disruptions/additions prioritized).
Click a row to see the variant's effect across cell types and its predicted accessibility + attributions for
every cell type. Significance p &le; {thr}.</div></header>
<div class="wrap">
 <div class="left">
  <input id="filt" onkeyup="filt()" placeholder="filter (variant, TF, cell type, trait, scope)...">
  <table id="t"><thead><tr>{header}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>
 </div>
 <div class="right">{''.join(detail_html)}
  <div id="hint" class="muted">Select a variant on the left.</div>
 </div>
</div>
<script>
 function show(id){{document.querySelectorAll('.detail').forEach(function(d){{d.classList.toggle('active',d.id===id);}});
   document.querySelectorAll('#t tbody tr').forEach(function(r){{r.classList.toggle('sel',r.dataset.did===id);}});
   var h=document.getElementById('hint'); if(h) h.style.display='none';}}
 function filt(){{var q=document.getElementById('filt').value.toLowerCase();
   document.querySelectorAll('#t tbody tr').forEach(function(r){{r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';}});}}
 function sortT(i){{var tb=document.querySelector('#t tbody'),rows=[].slice.call(tb.rows);
   var asc=tb.dataset.asc!==('1_'+i); tb.dataset.asc=asc?('1_'+i):('0_'+i);
   rows.sort(function(a,b){{var x=a.cells[i].innerText,y=b.cells[i].innerText,nx=parseFloat(x),ny=parseFloat(y);
     if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx; return asc?x.localeCompare(y):y.localeCompare(x);}});
   rows.forEach(function(r){{tb.appendChild(r);}});}}
</script></body></html>"""
    (browser / "variant_browser.html").write_text(doc)
    n_fig = len(list((browser / "allct").glob("*.png")))
    log.info("variant browser: %d variants, %d all-cell-type figures -> %s",
             len(sel), n_fig, browser / "variant_browser.html")
    io.write_provenance(cfg, "5_report_build_variant_browser",
                        {"n_variants": int(len(sel)), "n_figures": n_fig})


if __name__ == "__main__":
    main()
