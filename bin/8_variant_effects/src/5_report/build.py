"""Stage 5: assemble the deliverable folder and write REPORT.md.

Copies the report outputs into `deliverable/` under the reference filenames (a
drop-in for the Drive folder) and writes a data-forward REPORT.md: what was run,
where outputs live, and the tables/figures produced. No interpretive takeaways.

Usage:
    python src/5_report/build.py --config config/config.yaml
"""

import argparse
import base64
import html
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from common import io
from common.logging_setup import setup_logging

log = setup_logging(__file__)

# Figures embedded per trait in report.html (the rest stay in the deliverable folder).
EMBED_PER_TRAIT = [
    "celltype_vep_logfc.PCA_celltypes_by_celltype.pdf",
    "celltype_vep_logfc.PCA_celltypes_by_stage.pdf",
    "celltype_vep_logfc.PCA_celltypes_by_lineage.pdf",
    "celltype_vep_logfc.PCA_celltypes_by_endocrine_category.pdf",
    "celltype_vep_logfc.sig_variant_summary.pdf",
    "celltype_vep_logfc.clustermap_by_celltype.pdf",
    "celltype_vep_logfc.clustermap_by_dev_stage.pdf",
    "celltype_vep_logfc.clustermap_by_lineage.pdf",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    log.warning("missing, skipped: %s", src)
    return False


def assemble(cfg) -> dict:
    """Copy outputs into deliverable/ and return a manifest of what was found."""
    deliver = io.sandbox_path(cfg, "deliverable")
    deliver.mkdir(parents=True, exist_ok=True)
    # rebuild fresh so stale/renamed files don't accumulate, but preserve the
    # separately-built variant browser subfolder (built by build_variant_pages)
    for child in deliver.iterdir():
        if child.name == "browser":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    manifest: dict[str, list[str]] = {"qc": [], "per_trait": [], "variants": [],
                                      "contrast": [], "hits": []}

    for name in ["model_performance_overview.pdf", "sc-islet-differentiation_motif_logos.pdf"]:
        if copy_if_exists(io.sandbox_path(cfg, f"results/qc/{name}"), deliver / name):
            manifest["qc"].append(name)

    for trait in cfg["traits"]:
        key = trait["key"]
        vep = io.sandbox_path(cfg, f"results/{key}/vep_report_out")
        if vep.exists():
            for f in sorted(vep.glob("*")):
                copy_if_exists(f, deliver / key / f.name)
            manifest["per_trait"].append(key)
        gs = io.sandbox_path(cfg, f"results/{key}/group_specific")
        if gs.exists():
            for f in sorted(gs.glob("*.tsv")):
                copy_if_exists(f, deliver / key / "group_specific" / f.name)

    variants = io.sandbox_path(cfg, "results/variants")
    if variants.exists():
        for f in sorted(variants.glob("*.pdf")):
            copy_if_exists(f, deliver / "variants" / f.name)
            manifest["variants"].append(f.name)
    contrast = io.sandbox_path(cfg, "results/variants/contrast")
    if contrast.exists():
        for f in sorted(contrast.glob("*.pdf")):
            copy_if_exists(f, deliver / "variants" / "contrast" / f.name)
            manifest["contrast"].append(f.name)

    for name in ["hits_disruption.tsv", "variant_by_celltype.tsv", "variant_by_motif.tsv"]:
        if copy_if_exists(io.sandbox_path(cfg, f"results/hits/{name}"), deliver / "hits" / name):
            manifest["hits"].append(name)
    copy_if_exists(io.sandbox_path(cfg, "figures/hits_disruption_per_celltype.pdf"),
                   deliver / "hits" / "hits_disruption_per_celltype.pdf")
    return manifest


def write_report(cfg, manifest) -> Path:
    """Write REPORT.md in the house data-forward style."""
    traits = ", ".join(t["key"] for t in cfg["traits"])
    thr = cfg["significance"]["pval_threshold"]
    lines = [
        "# Variant-effect prediction on the 22 developmental cell types",
        "",
        "Predicted enhancer-activity change for T1D/T2D and glycemic credible-set variants across "
        "the 22 single-task ChromBPNet cell types, with cell-type-specific summaries and a "
        "variant-by-called-hits disruption analysis. Reproduces the 2025_11_05 report on the new cell types.",
        "",
        "## What was run",
        "",
        f"- Traits: {traits}. Scored once on the deduplicated variant union, split per trait for analysis.",
        f"- Significance: {cfg['significance']['pval_col']} <= {thr} (shuffled null from variant-scorer).",
        f"- Catalog for hit calling: {cfg['catalog_version']}.",
        "- Provenance for each stage is in `provenance/`; logs in `logs/`.",
        "",
        "## Outputs",
        "",
        "QC (all cell types): " + (", ".join(manifest["qc"]) or "none yet") + ".",
        "",
        "Per trait with a VEP report: " + (", ".join(manifest["per_trait"]) or "none yet") + ". Each "
        "`{trait}/vep_report_out/` holds the PCA of cell types (by stage and endocrine status), "
        "ridgeplots, boxplots, the significant-variant summary, and specificity tables.",
        "",
        f"Per-variant prediction and attribution PDFs: {len(manifest['variants'])} in `variants/`.",
        "",
        "Hits disruption: " + (", ".join(manifest["hits"]) or "none yet")
        + ". `hits_disruption.tsv` is the long variant x cell type x motif table; the cross-tabs give "
        "variant x cell type and variant x TF.",
        "",
        "## Reproduce",
        "",
        "```",
        "# from the sandbox dir; scoring on carter-gpu, analysis in env eugene_tools",
        "sbatch --array=1-22%8 --export=ALL,LIST=inputs/combined_credset.minimal.snps.chrombpnet.bed,"
        "OUTROOT=results/chrombpnet src/1_score/1_score_chrombpnet.sh",
        "python src/2_aggregate/1_aggregate_vep.py --config config/config.yaml",
        "python src/3_report/1_vep_report.py --config config/config.yaml",
        "sbatch --export=ALL src/3_report/2_plot_variant.sh",
        "python src/3_report/3_motif_and_model_qc.py --config config/config.yaml",
        "python src/4_hits/0_significant_variants.py --config config/config.yaml",
        "sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed src/4_hits/1_variant_shap.sh",
        "sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed src/4_hits/2_call_variant_hits.sh",
        "python src/4_hits/3_summarize_hits.py --config config/config.yaml",
        "python src/5_report/build.py --config config/config.yaml",
        "```",
        "",
    ]
    out = io.sandbox_path(cfg, "REPORT.md")
    out.write_text("\n".join(lines))
    return out


def pdf_to_png_datauris(pdf: Path, dpi: int = 110) -> list[str]:
    """Rasterize a PDF (one PNG per page) and return base64 data URIs."""
    if not pdf.exists():
        return []
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(Path(td) / "p")],
                       check=True)
        return [f"data:image/png;base64,{base64.b64encode(png.read_bytes()).decode()}"
                for png in sorted(Path(td).glob("p*.png"))]


def _img_block(pdf: Path, dpi: int = 110) -> str:
    uris = pdf_to_png_datauris(pdf, dpi)
    if not uris:
        return f'<p class="missing">missing: {html.escape(pdf.name)}</p>'
    caption = html.escape(pdf.name)
    imgs = "".join(f'<img src="{u}" alt="{caption}">' for u in uris)
    return f'<figure>{imgs}<figcaption>{caption}</figcaption></figure>'


def _table_block(df: pd.DataFrame, caption: str, max_rows: int = 20) -> str:
    sub = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in sub.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in sub.itertuples(index=False))
    return (f'<div class="tablewrap"><h4>{html.escape(caption)} '
            f'<span class="muted">(top {len(sub)} of {len(df)})</span></h4>'
            f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def write_html(cfg, manifest) -> Path:
    """Self-contained report.html: embedded figures, sortable tables, trait selector."""
    qc = io.sandbox_path(cfg, "results/qc")
    qc_html = "".join(_img_block(qc / n, dpi=90) for n in
                      ["model_performance_overview.pdf", "sc-islet-differentiation_motif_logos.pdf"])

    sections, options = [], []
    for trait in cfg["traits"]:
        key = trait["key"]
        vep = io.sandbox_path(cfg, f"results/{key}/vep_report_out")
        if not vep.exists():
            continue
        options.append(f'<option value="{key}">{key} ({html.escape(trait["study"])})</option>')
        figs = "".join(_img_block(vep / n) for n in EMBED_PER_TRAIT)
        tables = ""
        spec = vep / "celltype_vep_logfc.celltype_specific_variants.tsv"
        var = vep / "celltype_vep_logfc.top_variance_variants.tsv"
        if spec.exists():
            tables += _table_block(pd.read_csv(spec, sep="\t"), "Cell-type-specific variants")
        if var.exists():
            tables += _table_block(pd.read_csv(var, sep="\t"), "Top-variance variants")
        gs = io.sandbox_path(cfg, f"results/{key}/group_specific/grouping.specific_variants.tsv")
        if gs.exists():
            tables += _table_block(pd.read_csv(gs, sep="\t"),
                                   "Group-specific variants (by grouping)", max_rows=25)
        sections.append(
            f'<section class="trait" id="trait-{key}"><h3>{key} '
            f'<span class="muted">{html.escape(trait["study"])}</span></h3>{figs}{tables}</section>')

    contrast_dir = io.sandbox_path(cfg, "results/variants/contrast")
    cfigs = sorted(contrast_dir.glob("*.pdf")) if contrast_dir.exists() else []
    contrast_html = "".join(_img_block(f, dpi=78) for f in cfigs)

    hits_fig = _img_block(io.sandbox_path(cfg, "figures/hits_disruption_per_celltype.pdf"))
    hits_tsv = io.sandbox_path(cfg, "results/hits/hits_disruption.tsv")
    hits_tbl = (_table_block(pd.read_csv(hits_tsv, sep="\t"), "Variant-hit disruptions")
                if hits_tsv.exists() else '<p class="missing">no hits table yet</p>')

    thr = cfg["significance"]["pval_threshold"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Variant-effect prediction: 22 developmental cell types</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fff}}
 header{{padding:20px 28px;border-bottom:2px solid #eee}}
 h1{{margin:0 0 4px;font-size:22px}} .muted{{color:#888;font-weight:400}}
 .controls{{position:sticky;top:0;background:#fafafa;padding:12px 28px;border-bottom:1px solid #eee;z-index:5}}
 main{{padding:20px 28px;max-width:1100px}}
 section{{margin:0 0 40px}} figure{{margin:12px 0}}
 img{{max-width:100%;border:1px solid #eee;border-radius:4px;display:block;margin:6px 0}}
 figcaption{{font-size:12px;color:#888}}
 .tablewrap{{overflow-x:auto;margin:14px 0}} table{{border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #e2e2e2;padding:3px 8px;text-align:left}} th{{background:#f4f4f4;cursor:pointer}}
 .trait{{display:none}} .trait.active{{display:block}} .missing{{color:#c0392b;font-size:13px}}
 select{{font-size:15px;padding:4px 8px}}
</style></head><body>
<header><h1>Variant-effect prediction on the 22 developmental cell types</h1>
<div class="muted">Predicted enhancer-activity change for T1D/T2D and glycemic credible-set variants across the
22 single-task ChromBPNet cell types. Significance: {cfg['significance']['pval_col']} &le; {thr}.
Catalog {cfg['catalog_version']}. Built {date.today().isoformat()}.</div></header>
<div class="controls"><label>Trait: <select id="traitsel" onchange="showTrait(this.value)">{''.join(options)}</select></label></div>
<main>
<section><h2>Model QC (all cell types)</h2>{qc_html}</section>
<h2>Per-trait variant-effect report</h2>{''.join(sections)}
<section><h2>Variants disrupting called motif hits</h2>{hits_fig}{hits_tbl}</section>
<section><h2>Cell-type contrast examples ({len(cfigs)} variants, ranked by effect size)</h2>
<div class="muted">Ordered best-first: largest |logfc| in an accessible cell type (real peak), clear profile
change, and gained/lost motif in the attribution logos. Each panel shows predicted accessibility
(reference vs alternate, count-scaled) and attribution logos in the cell types where the variant acts
("ON") vs where it is inert ("off"); dotted line marks the variant. Title notes whether it disrupts a
called motif hit.</div>
{contrast_html or '<p class="missing">no contrast panels yet</p>'}</section>
</main>
<script>
 function showTrait(k){{document.querySelectorAll('.trait').forEach(function(s){{s.classList.toggle('active',s.id==='trait-'+k);}});}}
 var first=document.querySelector('#traitsel option'); if(first) showTrait(first.value);
 document.querySelectorAll('table.sortable th').forEach(function(th,i){{th.onclick=function(){{
   var tb=th.closest('table').tBodies[0], rows=[].slice.call(tb.rows);
   var asc=th.dataset.asc!=='1'; th.dataset.asc=asc?'1':'0';
   rows.sort(function(a,b){{var x=a.cells[i].innerText,y=b.cells[i].innerText;
     var nx=parseFloat(x),ny=parseFloat(y); if(!isNaN(nx)&&!isNaN(ny)){{return asc?nx-ny:ny-nx;}}
     return asc?x.localeCompare(y):y.localeCompare(x);}});
   rows.forEach(function(r){{tb.appendChild(r);}});}};}});
</script></body></html>"""
    out = io.sandbox_path(cfg, "report.html")
    out.write_text(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    manifest = assemble(cfg)
    report = write_report(cfg, manifest)
    html_report = write_html(cfg, manifest)
    io.write_provenance(cfg, "5_report_build", {"manifest": manifest})
    log.info("assembled deliverable, wrote %s and %s", report, html_report)


if __name__ == "__main__":
    main()
