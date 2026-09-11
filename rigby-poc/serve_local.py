"""Serve this checkout's API on 8011.

The venv lives beside the other checkout and has no editable install pointing
here, so the path is set explicitly rather than relied on.
"""
import os
import sys

# ONE BLAS THREAD, set before anything imports numpy. OpenBLAS starts a worker
# thread per logical core the moment numpy loads -- sixteen here -- and each
# reserves its own buffer. On a machine near its commit limit that was enough to
# stop the API booting at all: "OpenBLAS error: Memory allocation still failed
# after 10 retries, giving up". Nothing here does work that sixteen BLAS threads
# would speed up, and the run workers inherit this, because gripper_runs starts
# them with a copy of this process's environment. setdefault, so a value set
# deliberately from outside still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, os.path.join(here, "src"))

import uvicorn  # noqa: E402

uvicorn.run("rigby_poc.app:app", host="127.0.0.1", port=8011, log_level="info")
