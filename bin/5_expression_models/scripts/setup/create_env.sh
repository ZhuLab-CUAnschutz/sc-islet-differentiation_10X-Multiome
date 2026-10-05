#!/bin/bash
# One-time build of the gReLU/Decima (PyTorch) env for the seq->expression models.
# Separate from the CREsted `.venv-keras` (TensorFlow) env used by the ATAC oracles.
#
# Run on a compute node (NOT the 4GB head node):
#   srun --partition=carter-compute --mem=32G --cpus-per-task=4 --pty bash
#   bash create_env.sh
set -euo pipefail

ENV_DIR="/cellar/users/aklie/opt/decima/.venv-torch"   # adjust to taste
PY=python3.11

echo "Creating venv at ${ENV_DIR}"
$PY -m venv "$ENV_DIR"
source "${ENV_DIR}/bin/activate"
pip install --upgrade pip wheel

# --- Core DL stack (CUDA 12.x wheels; match the A30 driver on carter-gpu) ---
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
pip install "lightning>=2.2"

# --- gReLU + Decima ---
# gReLU (Genentech) provides BorzoiModel, ConvHead, interpret utils.
pip install grelu
# Decima: install from GitHub (pins the model/loss/preprocess code we call).
pip install "git+https://github.com/Genentech/decima.git"

# --- Interpretation / motif deps ---
pip install captum tangermeme modiscolite

# --- LoRA (regime=lora escalation) ---
pip install peft

# --- Data / plotting (align with existing project stack) ---
pip install "anndata>=0.10" "scanpy>=1.10" pyBigWig pysam pyyaml \
            pandas numpy scipy scikit-learn matplotlib seaborn

echo
echo "Done. Activate with:  source ${ENV_DIR}/bin/activate"
echo "Sanity check:"
python - <<'PY'
import torch, grelu
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("grelu", getattr(grelu, "__version__", "?"))
try:
    import decima
    print("decima", getattr(decima, "__version__", "?"))
except Exception as e:
    print("decima import failed:", e)
PY
