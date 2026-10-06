#!/bin/bash
# One-time env build for bin/7_sequence_design: a venv layered on the base torch env, adding
# CREsted 1.7.1 + Keras 3.14. The Keras PyTorch backend is used because the original TensorFlow
# env is not mounted on the GPU nodes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source paths.sh

VENV="$BIN/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    "$BASE_PY" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install "numpy==1.26.4" "keras==3.14.0" "crested==1.7.1"
"$VENV/bin/python" - <<'PYEOF'
from importlib.metadata import version
for package in ("crested", "keras", "torch", "numpy", "pandas"):
    print(package, version(package))
PYEOF
