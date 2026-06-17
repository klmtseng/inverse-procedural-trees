"""Centralised filesystem paths for the inverse-procedural-trees repo."""
import os

# resolves to the repo root (one above src/)
ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
))

PHOTOS   = os.path.join(ROOT, "photos")
LEAF_LIB = os.path.join(ROOT, "leaf_libs")
RESULTS  = os.path.join(ROOT, "results")

os.makedirs(LEAF_LIB, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
