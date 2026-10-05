#!/bin/bash
#SBATCH --job-name=syn_downstream
#SBATCH --account=carter-compute
#SBATCH --partition=carter-compute
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 2:00:00
#SBATCH --output=slurm_logs/syn_downstream.%j.out
# CPU steps after the sweep: frac_pos -> calls + clustering -> signatures -> logos -> report bundle.
# Safe to run on a partial sweep (uses whatever chunks have finished).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"   # submit from bin/6_synergy/
export SYNERGY_DIR=$SYN
S=$BIN/scripts
$PY $S/extract_frac_pos.py --results $SYN/sweep -o $SYN/inputs/frac_pos.tsv
$PY $S/downstream.py --results $SYN/sweep --meta $CATALOG/metadata.tsv --fracpos $SYN/inputs/frac_pos.tsv --out $SYN
$PY $S/explore_signatures.py
$PY $S/plot_split.py
[ -f $SYN/inputs/logos_b64.json ] || $PY $S/make_logos.py --cwms $CWMS -o $SYN/inputs/logos_b64.json
$PY $S/build_report.py
