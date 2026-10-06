"""Stage 5c: variant browser as an index page + one dedicated page per variant.

Builds ``deliverable/browser/``:
  - ``index.html`` - sortable/filterable table; each row links to that variant's page.
  - ``pages/{vid}.html`` - a dedicated, readable page: the cell-type-colored effect
    barplot and one large panel per cell type (predicted accessibility + reference and
    alternate attributions, annotated with the finemo called hits), from Stage 3.9.

Panels are ordered by developmental stage (the metadata display order), matching the
barplot. Run after Stage 3.9 (per-variant page figures) and Stage 3.8 (selection).

Usage:
    python src/5_report/build_variant_pages.py --config config/config.yaml
"""

import argparse
import base64
import html
import io as _io
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib

log = setup_logging(__file__)

STYLE = """
 body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fafafa}
 header{padding:16px 24px;border-bottom:2px solid #eee;background:#fff}
 h1{margin:0;font-size:20px} .muted{color:#888;font-weight:400}
 a{color:#1552b0;text-decoration:none} a:hover{text-decoration:underline}
 .content{padding:16px 24px}
 input{width:100%;max-width:560px;padding:7px 9px;font-size:14px;margin-bottom:10px}
 table{border-collapse:collapse;font-size:13px;width:100%;background:#fff}
 th,td{border:1px solid #e2e2e2;padding:4px 8px;text-align:left}
 th{background:#f4f4f4;cursor:pointer;position:sticky;top:0}
 tbody tr{cursor:pointer} tbody tr:hover{background:#fbf3d0}
 .wrap{display:flex;gap:22px;align-items:flex-start}
 .left{flex:0 0 430px;position:sticky;top:12px}
 .left img{width:100%;border:1px solid #e6e6e6;background:#fff;border-radius:4px}
 .right{flex:1;min-width:0}
 figure{margin:0 0 18px 0;background:#fff;border:1px solid #e6e6e6;border-radius:5px;padding:6px}
 figure img{width:100%;display:block}
 .meta{color:#444;margin:6px 0 14px} .back{font-size:13px}
 .legend{font-size:12px;color:#666;margin-top:8px}
"""


def _table(sel: pd.DataFrame, thr: float) -> str:
    cols = ["rank", "variant", "trait", "top cell type", "logfc", "dir", "scope",
            "motif (ST##:TF)", "PIP", "hit"]
    header = "".join(f'<th onclick="sortT({i})">{h}</th>' for i, h in enumerate(cols))
    rows = []
    for _, r in sel.iterrows():
        safe = str(r["variant_id"]).replace(":", "_")
        cells = [int(r["rank"]), html.escape(str(r["variant_id"])), r["trait"],
                 r["top_cell_type"], f"{r['logfc']:.2f}", r["direction"],
                 str(r["scope"]).split(":")[0], r["primary_motif"],
                 f"{r['PIP']:.3g}", "yes" if r["disrupts_hit"] else "no"]
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
        rows.append(f'<tr onclick="location.href=\'pages/{safe}.html\'">{tds}</tr>')
    return (f'<input id="filt" onkeyup="filt()" placeholder="filter (variant, motif, '
            f'cell type, trait, scope)...">'
            f'<table id="t"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


INDEX_JS = """
 function filt(){var q=document.getElementById('filt').value.toLowerCase();
   document.querySelectorAll('#t tbody tr').forEach(function(r){
     r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';});}
 function sortT(i){var tb=document.querySelector('#t tbody'),rows=[].slice.call(tb.rows);
   var asc=tb.dataset.asc!==('1_'+i); tb.dataset.asc=asc?('1_'+i):('0_'+i);
   rows.sort(function(a,b){var x=a.cells[i].innerText,y=b.cells[i].innerText,nx=parseFloat(x),ny=parseFloat(y);
     if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx; return asc?x.localeCompare(y):y.localeCompare(x);});
   rows.forEach(function(r){tb.appendChild(r);});}
"""


def write_index(sel: pd.DataFrame, thr: float, out: Path) -> None:
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Variant browser - 22 developmental cell types</title><style>{STYLE}</style></head><body>
<header><h1>Variant browser</h1>
<div class="muted">{len(sel)} curated variants (non-CTCF; lineage-TF disruptions/additions
prioritized). Click a row to open that variant's dedicated page: effect across all 22
cell types and, per cell type, predicted accessibility + attributions with the called
hits annotated. Significance p &le; {thr}. Motif = the hit called in the strongest cell
type(s).</div></header>
<div class="content">{_table(sel, thr)}</div>
<script>{INDEX_JS}</script></body></html>"""
    out.write_text(doc)


def write_variant_page(cfg, r, order, cmap, src_dir: Path, page_dir: Path, assets: Path) -> bool:
    """Copy this variant's panel PNGs and write its dedicated page. False if no figures."""
    safe = str(r["variant_id"]).replace(":", "_")
    if not (src_dir / "bar.png").exists():
        return False
    dst = assets / safe
    dst.mkdir(parents=True, exist_ok=True)
    for png in src_dir.glob("*.png"):
        shutil.copy2(png, dst / png.name)

    panels = []
    for ct in order:  # developmental-stage order, matching the barplot
        # the page lives in pages/, so asset paths are relative to it: {safe}/{ct}.png
        if (dst / f"{ct}.png").exists():
            panels.append(f'<figure><a href="{safe}/{ct}.png" target="_blank">'
                          f'<img src="{safe}/{ct}.png" alt="{html.escape(ct)}"></a></figure>')
    disrupts = html.escape(str(r.get("hit_motifs", "")) or "none")
    meta = (f'top cell type <b>{html.escape(str(r["top_cell_type"]))}</b> &middot; '
            f'logfc {r["logfc"]:.2f} &middot; {r["direction"]} of '
            f'<b>{html.escape(str(r["primary_motif"]))}</b> &middot; {html.escape(str(r["scope"]))} '
            f'&middot; PIP {r["PIP"]:.3g} &middot; disrupts hit: {disrupts}')
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(str(r["variant_id"]))} - variant page</title><style>{STYLE}</style></head><body>
<header><div class="back"><a href="../index.html">&larr; back to index</a></div>
<h1>{html.escape(str(r["variant_id"]))} <span class="muted">[{r["trait"]}]</span></h1>
<div class="meta">{meta}</div></header>
<div class="content"><div class="wrap">
 <div class="left"><a href="{safe}/bar.png" target="_blank">
   <img src="{safe}/bar.png" alt="effect across cell types"></a>
   <div class="legend">logfc (alt vs ref) across all 22 cell types, colored by cell type.
   Predictions span the full 1 kb window; attributions zoom to &plusmn;50 bp. Reference is
   dashed/lighter, alternate solid/darker. Attribution annotations (only those overlapping
   the variant, for that cell type) show two tracks: <b style="color:#111">finemo called
   hits</b> (below) and <b style="color:#8e44ad">seqlet matches</b> (above), each labeled
   ST##:TF. Predictions and attributions share one scale across cell types. Click any
   figure to open it full size.</div></div>
 <div class="right">{''.join(panels) or '<p>no panels rendered yet</p>'}</div>
</div></div></body></html>"""
    (page_dir / f"{safe}.html").write_text(doc)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    order = metadata.cell_types(cfg)
    cmap = metadata.color_map(cfg)
    thr = cfg["significance"]["pval_threshold"]

    sel = pd.read_csv(io.sandbox_path(cfg, "results/variants/browser_variants.tsv"), sep="\t")
    browser = io.sandbox_path(cfg, "deliverable/browser")
    if browser.exists():
        shutil.rmtree(browser)
    pages = browser / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    src_root = io.sandbox_path(cfg, "results/variants/pages")

    n_pages = 0
    for _, r in sel.iterrows():
        safe = str(r["variant_id"]).replace(":", "_")
        if write_variant_page(cfg, r, order, cmap, src_root / safe, pages, pages):
            n_pages += 1
    write_index(sel, thr, browser / "index.html")
    log.info("variant browser: index + %d/%d dedicated pages -> %s",
             n_pages, len(sel), browser / "index.html")
    io.write_provenance(cfg, "5_report_build_variant_pages",
                        {"n_variants": int(len(sel)), "n_pages": n_pages})


if __name__ == "__main__":
    main()
