"""
LGPC-BC (Local-Global Photometric Consistency - Background Consistency) v1.1
-----------------------------------------------------------------------------

A CPU-only, classical (non-learned) forensic consistency check for
gel/blot images, implementing the frozen methodology v1.1.

This package deliberately implements ONLY what the frozen methodology
specifies. It does not add alternative detection channels, learned
models, or extra heuristics.

See src/detector/detector.py for the top-level pipeline entry point.
"""

__version__ = "1.1.0"
