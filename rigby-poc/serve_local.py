"""Serve this checkout's API on 8011.

The venv lives beside the other checkout and has no editable install pointing
here, so the path is set explicitly rather than relied on.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, os.path.join(here, "src"))

import uvicorn  # noqa: E402

uvicorn.run("rigby_poc.app:app", host="127.0.0.1", port=8011, log_level="info")
