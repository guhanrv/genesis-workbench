"""Make ``pca_v1/lib`` importable for the off-cluster test suite.

The notebooks ship the lib to executors via ``addPyFile`` + a runtime ``sys.path.append``;
off-cluster we add the sibling ``lib/`` dir to ``sys.path`` so ``import fraposa`` etc. work
under a plain ``pytest`` from anywhere. (Each test module also inserts the path defensively,
so they run standalone via ``python test_*.py`` too.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

# Newer MLflow (3.x) refuses the filesystem tracking store unless opted in.
# Off-cluster unit tests log to a temp file store; they do not need a DB backend.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
