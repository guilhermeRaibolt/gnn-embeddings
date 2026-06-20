"""Render the final results as a Markdown table in the log."""

from __future__ import annotations

import logging
from typing import Sequence


def log_markdown_summary(results: Sequence[dict], gnn_type: str, logger: logging.Logger) -> None:
    """Append a clean Markdown summary table to the log file / console."""
    logger.info("")
    logger.info("## Final Summary (GNN backbone: %s)", gnn_type)
    logger.info("")
    logger.info("| Encoder | Scale | Feature Dim | Best Params | Mean Test Acc | 95% CI |")
    logger.info("| --- | --- | ---: | --- | ---: | ---: |")
    for row in sorted(results, key=lambda r: r["mean_test_accuracy"]):
        cfg = row["best_config"]
        params = f"lr={cfg['learning_rate']}, hidden={cfg['hidden_channels']}, dropout={cfg['dropout']}"
        logger.info(
            "| %s | %s | %d | %s | %.4f | ± %.4f |",
            row["name"], row["scale"], row["embedding_dim"],
            params, row["mean_test_accuracy"], row["ci95"],
        )
    logger.info("")
