"""Per-CT motif-hit genomic-context annotator (Phase 7.2 of MC cluster pipeline).

For one CT's hits.tsv (from finemo), compute per hit:
  - distance to nearest TSS (signed bp; matches Liu 2026 Methods)
  - genomic region category (promoter > 5UTR > 3UTR > exon > intron > distal,
    priority-resolved against a GENCODE v43 features bed)
  - distance to nearest narrowPeak summit (proxy for nucleosome-edge-vs-summit
    position pending NucleoATAC; matches paper's peak-summit complement)

Aggregate per motif_name into:
  motif_genomic_context.tsv  — one row per motif:
    motif_name, n_hits, median_tss_distance_abs, median_tss_distance_signed,
    median_summit_distance_abs, pct_{promoter,5UTR,3UTR,exon,intron,distal}

Also write the long-format per-hit annotation for downstream box plots:
  hit_genomic_context.tsv (gzipped) — one row per hit.
"""
import argparse, subprocess, tempfile, gzip
from pathlib import Path
import pandas as pd
import numpy as np

BT = "/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/bedtools"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hits_tsv", required=True)
    p.add_argument("--peakfile", required=True, help="narrowPeak for this CT")
    p.add_argument("--tss_bed", required=True)
    p.add_argument("--features_bed", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # --- 1) load hits ---
    print("[1/6] loading hits", flush=True)
    cols_to_keep = ["chr", "start", "end", "motif_name", "strand", "peak_id", "hit_importance"]
    h = pd.read_csv(args.hits_tsv, sep="\t", usecols=cols_to_keep)
    h["center"] = ((h["start"].astype(int) + h["end"].astype(int)) // 2).astype(int)
    h["hit_idx"] = np.arange(len(h))
    print(f"  hits={len(h)}, motifs={h['motif_name'].nunique()}", flush=True)

    # --- 2) peak summits ---
    print("[2/6] peak summits", flush=True)
    peaks = pd.read_csv(args.peakfile, sep="\t", header=None,
                        usecols=[0, 1, 2, 9], names=["chr", "p_start", "p_end", "p_off"])
    peaks["summit"] = (peaks["p_start"].astype(int) + peaks["p_off"].astype(int)).astype(int)
    summit_bed_df = peaks[["chr", "summit"]].copy()
    summit_bed_df["end"] = summit_bed_df["summit"] + 1
    summit_bed_df = summit_bed_df[["chr", "summit", "end"]].rename(columns={"summit": "start"})

    # --- 3) write center bed (4-col: chr, center, center+1, hit_idx) ---
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cen_bed = td / "centers.bed"
        cen_df = h[["chr", "center", "hit_idx"]].copy()
        cen_df["end"] = cen_df["center"] + 1
        cen_df = cen_df[["chr", "center", "end", "hit_idx"]].rename(columns={"center": "start"})
        cen_df.sort_values(["chr", "start"]).to_csv(cen_bed, sep="\t", index=False, header=False)
        print(f"  wrote centers.bed", flush=True)

        # --- 4) TSS distance (bedtools closest) ---
        print("[3/6] TSS distance via bedtools closest", flush=True)
        tss_sorted = td / "tss.sorted.bed"
        subprocess.run(f"sort -k1,1 -k2,2n {args.tss_bed} > {tss_sorted}", shell=True, check=True)
        tss_out = td / "centers_tss.bed"
        # closest: -D ref => signed distance (downstream of ref strand convention).
        # We use absolute later in agg; signed kept for users wanting upstream/downstream.
        subprocess.run(f"{BT} closest -a {cen_bed} -b {tss_sorted} -D b -t first > {tss_out}",
                       shell=True, check=True)
        # cols: chr, start, end, hit_idx, tss_chr, tss_start, tss_end, gene_id, gene_name, strand, dist
        td_df = pd.read_csv(tss_out, sep="\t", header=None,
                            names=["chr", "start", "end", "hit_idx",
                                   "tss_chr", "tss_start", "tss_end", "tss_gene_id",
                                   "tss_gene_name", "tss_strand", "tss_distance"])

        # --- 5) region label (bedtools intersect -loj) ---
        print("[4/6] region label via bedtools intersect -loj", flush=True)
        feat_sorted = td / "features.sorted.bed"
        subprocess.run(f"sort -k1,1 -k2,2n {args.features_bed} > {feat_sorted}", shell=True, check=True)
        feat_out = td / "centers_feat.bed"
        subprocess.run(f"{BT} intersect -a {cen_bed} -b {feat_sorted} -loj > {feat_out}",
                       shell=True, check=True)
        # cols: chr, start, end, hit_idx, f_chr, f_start, f_end, feature_label
        fd_df = pd.read_csv(feat_out, sep="\t", header=None,
                            names=["chr", "start", "end", "hit_idx",
                                   "f_chr", "f_start", "f_end", "region"])
        fd_df["region"] = fd_df["region"].replace(".", "distal")
        # collapse: each hit_idx -> one region (loj can dup if features overlap, but ours don't)
        fd_one = fd_df.drop_duplicates("hit_idx", keep="first")[["hit_idx", "region"]]

        # --- 6) summit distance ---
        print("[5/6] summit distance via bedtools closest", flush=True)
        summit_bed = td / "summits.bed"
        summit_bed_df.sort_values(["chr", "start"]).to_csv(summit_bed, sep="\t", index=False, header=False)
        summit_out = td / "centers_summit.bed"
        subprocess.run(f"{BT} closest -a {cen_bed} -b {summit_bed} -d -t first > {summit_out}",
                       shell=True, check=True)
        sd_df = pd.read_csv(summit_out, sep="\t", header=None,
                            names=["chr", "start", "end", "hit_idx",
                                   "s_chr", "s_start", "s_end", "summit_distance"])

    # --- 7) join all annotations onto hits ---
    print("[6/6] joining + per-motif aggregation", flush=True)
    annot = (h[["hit_idx", "motif_name", "hit_importance"]]
             .merge(td_df[["hit_idx", "tss_distance", "tss_gene_name"]], on="hit_idx", how="left")
             .merge(fd_one, on="hit_idx", how="left")
             .merge(sd_df[["hit_idx", "summit_distance"]], on="hit_idx", how="left"))
    annot["region"] = annot["region"].fillna("distal")

    # write per-hit (gzipped)
    long_path = out / "hit_genomic_context.tsv.gz"
    annot.to_csv(long_path, sep="\t", index=False, compression="gzip")
    print(f"  wrote {long_path}", flush=True)

    # per-motif summary
    REGIONS = ["promoter", "5UTR", "3UTR", "exon", "intron", "distal"]
    grp = annot.groupby("motif_name")

    summary = pd.DataFrame({
        "n_hits": grp.size(),
        "median_tss_distance_abs": grp["tss_distance"].apply(lambda s: float(np.median(np.abs(s)))),
        "median_tss_distance_signed": grp["tss_distance"].median(),
        "median_summit_distance_abs": grp["summit_distance"].apply(lambda s: float(np.median(np.abs(s)))),
    })
    region_counts = grp["region"].value_counts().unstack(fill_value=0)
    for r in REGIONS:
        if r not in region_counts.columns:
            region_counts[r] = 0
    region_pct = region_counts[REGIONS].div(region_counts[REGIONS].sum(axis=1), axis=0) * 100
    region_pct.columns = [f"pct_{c}" for c in region_pct.columns]
    summary = summary.join(region_pct, how="left").reset_index()

    summary_path = out / "motif_genomic_context.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"  wrote {summary_path} ({len(summary)} motifs)", flush=True)
    print(summary[["motif_name", "n_hits", "median_tss_distance_abs",
                   "median_summit_distance_abs"]].head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
