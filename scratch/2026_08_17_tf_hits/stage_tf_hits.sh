#!/bin/bash
#SBATCH --job-name=stage_tf_hits
#SBATCH --partition=carter-compute
#SBATCH --account=carter-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_08_17_tf_hits/stage_tf_hits.%j.out

# Stage the v1.1 Fi-NeMo TF hit calls as TF-labelled BED files for Globus transfer.
# Source : results/3_single_task_models/motifs/v1.1/hits/<ct>/hits.tsv  (22 cell types)
# Output : scratch/2026_08_17_tf_hits/  -> Globus -> Han_analysis:/single_cell_processed/2026_08_17/

set -euo pipefail

REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
V=$REPO/results/3_single_task_models/motifs/v1.1
OUT=$REPO/scratch/2026_08_17_tf_hits

mkdir -p "$OUT/hits"
cd "$OUT"

# ---------------------------------------------------------------- 1. motif map
# hits.tsv column 6 (motif_name) == metadata.tsv cluster_name. Build the
# motif_name -> readable ST##_TF label map, and keep it as a shipped table.
awk -F'\t' -v OFS='\t' '
NR==1 { for (i=1;i<=NF;i++) h[$i]=i
        print "motif_name","short_id","label","tf","category","width"; next }
{
  sid=$(h["short_id"]); cn=$(h["cluster_name"])
  cid=$(h["curator_id"]); tf=$(h["curator_tf"]); cat=$(h["curator_category"])
  lab = (cid=="" ? sid "_unannotated" : cid)
  gsub(/[ \/()]/,"_",lab); gsub(/_+$/,"",lab); gsub(/__+/,"_",lab)
  if (tf=="") tf="NA"; if (cat=="") cat="NA"
  print cn, sid, lab, tf, cat, $(h["width"])
}' "$V/metadata.tsv" > motif_catalog.tsv

echo "[map] $(($(wc -l < motif_catalog.tsv)-1)) motifs"

cp "$V/catalog.meme" catalog.meme

# ------------------------------------------------------------------- 2. beds
# hits.tsv cols: 1 chr, 2 start, 3 end, 6 motif_name, 11 hit_importance, 13 strand
# out: BED6+1 = chr, start, end, label, hit_importance, strand, motif_name
: > hit_counts_long.tsv
for d in "$V"/hits/*/; do
  ct=$(basename "$d")
  [ -f "$d/hits.tsv" ] || { echo "[skip] $ct (no hits.tsv)"; continue; }

  awk -F'\t' -v OFS='\t' '
    NR==FNR { if (FNR>1) lab[$1]=$3; next }
    FNR==1  { next }
    { l = ($6 in lab) ? lab[$6] : $6
      print $1, $2, $3, l, $11, $13, $6 }
  ' motif_catalog.tsv "$d/hits.tsv" \
  | sort -k1,1 -k2,2n -S 2G --parallel=4 \
  | gzip -c > "hits/${ct}.hits.bed.gz"

  # per-motif counts for this cell type
  zcat "hits/${ct}.hits.bed.gz" | cut -f4 | sort -S 1G | uniq -c \
  | awk -v ct="$ct" -v OFS='\t' '{print ct, $2, $1}' >> hit_counts_long.tsv

  n=$(zcat "hits/${ct}.hits.bed.gz" | wc -l)
  echo "[bed] $ct  $n hits"
done

# ------------------------------------------------- 3. motif x cell type matrix
awk -F'\t' -v OFS='\t' '
{ n[$2 SUBSEP $1]=$3; motif[$2]; ct[$1] }
END {
  printf "motif"
  nc=0; for (c in ct) { nc++; cts[nc]=c }
  # stable column order
  for (i=1;i<nc;i++) for (j=i+1;j<=nc;j++) if (cts[j]<cts[i]) { t=cts[i]; cts[i]=cts[j]; cts[j]=t }
  for (i=1;i<=nc;i++) printf "\t%s", cts[i]
  printf "\n"
  nm=0; for (m in motif) { nm++; ms[nm]=m }
  for (i=1;i<nm;i++) for (j=i+1;j<=nm;j++) if (ms[j]<ms[i]) { t=ms[i]; ms[i]=ms[j]; ms[j]=t }
  for (i=1;i<=nm;i++) {
    printf "%s", ms[i]
    for (k=1;k<=nc;k++) { v=n[ms[i] SUBSEP cts[k]]; printf "\t%d", (v==""?0:v) }
    printf "\n"
  }
}' hit_counts_long.tsv > hit_counts_matrix.tsv

echo "[done] $(du -sh "$OUT" | cut -f1) staged at $OUT"
ls -la "$OUT" "$OUT/hits"
