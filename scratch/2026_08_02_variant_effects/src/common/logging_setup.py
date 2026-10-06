"""Structured logging for pipeline stages.

Each stage calls ``setup_logging(__file__)`` at module top. Logs go to stdout and
to ``logs/<path-under-src>.log`` (the tree under ``logs/`` mirrors ``src/``), so a
run stays auditable after the fact. The file handler is skipped under pytest.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logging(script_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger for a stage script.

    Logs to stdout always, and to ``<sandbox>/logs/<path-under-src>.log`` when
    the script lives under a ``src/`` tree. ``script_file`` is the caller's
    ``__file__``.
    """
    script = Path(script_file).resolve()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    under_pytest = "PYTEST_CURRENT_TEST" in os.environ
    if not under_pytest and "src" in script.parts:
        src_idx = len(script.parts) - 1 - script.parts[::-1].index("src")
        sandbox_root = Path(*script.parts[:src_idx])
        rel = script.relative_to(sandbox_root / "src").with_suffix(".log")
        log_path = sandbox_root / "logs" / rel
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(script.stem)
