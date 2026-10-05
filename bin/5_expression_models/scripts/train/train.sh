#!/bin/bash
# Train one seq->expression arm on carter-gpu.
# Submit examples:
#   gpu.sh -s train.sh -j decima_probe  -m 64G -t 08:00:00 -c 4 -g a30:1  -- --arm decima --regime probe
#   gpu.sh -s train.sh -j borzoi_probe  -m 64G -t 08:00:00 -c 4 -g a30:1  -- --arm borzoi --regime probe
#   gpu.sh -s train.sh -j decima_lora   -m 80G -t 24:00:00 -c 8 -g a30:1  -- --arm decima --regime lora
# (Args after `--` pass through to train.py. Adjust gpu.sh flag syntax to your wrapper.)

date
echo -e "Job ID: ${SLURM_JOB_ID:-local}\n"

source /carter/users/aklie/opt/miniconda3/etc/profile.d/conda.sh && conda activate decima   # gRelu-clone + decima
python -c "import torch; print('cuda:', torch.cuda.is_available())"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python train.py "$@"

echo -e "\nDone."; date
