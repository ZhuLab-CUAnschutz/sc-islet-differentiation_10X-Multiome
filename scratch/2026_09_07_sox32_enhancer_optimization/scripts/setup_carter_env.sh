#!/bin/bash
set -euo pipefail

RUN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
VENV="$RUN_DIR/.venv"

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
