#!/bin/bash
set -e

LOCAL=/Users/adamklie/Desktop/ucsd/islet_organoid_differentiation/sandbox/2026_09_06_de_synergy
REMOTE=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_09_06_de_synergy

ssh -o BatchMode=yes -o ServerAliveInterval=20 aklie@narrows-login.sdsc.edu \
  "bash -lc 'while squeue -h -j 1047592 | grep -q .; do sleep 30; done; sacct -j 1047592 -X -n -o state'" \
  > "$LOCAL/logs/mt_final_states.txt"

if grep -qvE '^\s*COMPLETED\s*$|^\s*$' "$LOCAL/logs/mt_final_states.txt"; then
  echo "Multitask job did not complete cleanly"
  cat "$LOCAL/logs/mt_final_states.txt"
  exit 1
fi

rsync -az -e "ssh -o BatchMode=yes" \
  "aklie@narrows-login.sdsc.edu:$REMOTE/results/mt_de4/" "$LOCAL/results/mt_de4/"
rsync -az -e "ssh -o BatchMode=yes" \
  "aklie@narrows-login.sdsc.edu:$REMOTE/results/mt_null_full/" "$LOCAL/results/mt_null_full/"
rsync -az -e "ssh -o BatchMode=yes" \
  "aklie@narrows-login.sdsc.edu:/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/scratch/2026_09_06_de_synergy/scripts/logs/de4_mt.1047592_*.out" "$LOCAL/logs/"

cd "$LOCAL"
python3 scripts/analyze_mt_de4.py > logs/analyze_mt_de4.txt
echo "MT_ANALYSIS_COMPLETE"
