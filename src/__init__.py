"""Modular pipeline for studying how text-embedding scale affects GNN node classification.

Sub-modules:
    logging_utils  - dual console/file logging.
    config         - dataclasses, encoder specs and dataset URLs.
    data           - download, parse and build the Amazon "Toys & Games" graph.
    encoders       - offline (frozen) feature extraction: BoW, TF-IDF, sBERT, Qwen3.
    models         - GCN / GraphSAGE PyG architectures.
    training       - single-run training, hyper-parameter search and seeded evaluation.
    summary        - Markdown result table.
"""

__all__ = [
    "logging_utils",
    "config",
    "data",
    "encoders",
    "models",
    "training",
    "summary",
]
