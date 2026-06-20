"""Figure generation for the embedding-scale study.

All plots are rendered head-less (Agg backend) and saved to ``figures/`` so they
work on a GPU compute node with no display. Three figure types:

    1. Accuracy vs. encoder (bar chart with 95% CI error bars)  -> the headline.
    2. t-SNE projection of node features, coloured by class      -> per encoder.
    3. Training curves (loss/accuracy over epochs)               -> per encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402


def _savefig(fig, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# --------------------------------------------------------------------------- #
# 1. Accuracy vs. encoder / scale                                             #
# --------------------------------------------------------------------------- #
def plot_accuracy_vs_scale(
    results: Sequence[dict], out_path: Path, gnn_type: str, logger: logging.Logger
) -> None:
    """Bar chart of mean test accuracy per encoder with 95% CI error bars."""
    if not results:
        return
    names = [r["name"] for r in results]
    means = [r["mean_test_accuracy"] for r in results]
    cis = [r["ci95"] for r in results]
    # Colour lexical/dense baselines vs. Qwen3 scales differently.
    colours = ["#4c72b0" if r["kind"] == "qwen3" else "#999999" for r in results]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=cis, capsize=6, color=colours, edgecolor="black", linewidth=0.6)
    for xi, m, c in zip(x, means, cis):
        ax.text(xi, m + (c or 0) + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, min(1.0, max(means) + max(cis) + 0.08))
    ax.set_title(f"Embedding scale vs. node-classification accuracy ({gnn_type.upper()})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4c72b0", ec="black"),
        plt.Rectangle((0, 0), 1, 1, color="#999999", ec="black"),
    ]
    ax.legend(handles, ["Qwen3 (LLM)", "Baseline"], loc="lower right")
    _savefig(fig, out_path, logger)


# --------------------------------------------------------------------------- #
# 2. t-SNE of node features                                                   #
# --------------------------------------------------------------------------- #
def plot_embedding_tsne(
    features: Tensor,
    y: Tensor,
    label_names: Sequence[str],
    out_path: Path,
    title: str,
    logger: logging.Logger,
    max_points: int = 2000,
    max_legend_classes: int = 15,
    seed: int = 42,
) -> None:
    """2-D t-SNE scatter of node features, coloured by class label."""
    from sklearn.manifold import TSNE

    feats = features.detach().cpu().numpy()
    labels = y.detach().cpu().numpy()

    # Subsample for tractable t-SNE on large graphs.
    rng = np.random.default_rng(seed)
    if feats.shape[0] > max_points:
        idx = rng.choice(feats.shape[0], size=max_points, replace=False)
        feats, labels = feats[idx], labels[idx]

    perplexity = float(min(30, max(5, (feats.shape[0] - 1) // 3)))
    logger.info("Computing t-SNE for '%s' on %d points (perplexity=%.0f) ...",
                title, feats.shape[0], perplexity)
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=seed,
    ).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 7))
    present = sorted(set(labels.tolist()))
    show_legend = len(present) <= max_legend_classes
    cmap = plt.get_cmap("tab20", len(present))
    for ci, cls in enumerate(present):
        mask = labels == cls
        name = label_names[cls] if cls < len(label_names) else str(cls)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   color=cmap(ci), label=name if show_legend else None)
    ax.set_title(f"t-SNE of node features — {title}")
    ax.set_xticks([]); ax.set_yticks([])
    if show_legend:
        ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.6)
    _savefig(fig, out_path, logger)


# --------------------------------------------------------------------------- #
# 3. Training curves                                                          #
# --------------------------------------------------------------------------- #
def plot_training_curves(
    history: Sequence[Dict], out_path: Path, title: str, logger: logging.Logger
) -> None:
    """Loss and accuracy vs. epoch for a single (representative) run."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss"); ax_loss.grid(alpha=0.3); ax_loss.legend()

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.grid(alpha=0.3); ax_acc.legend()

    fig.suptitle(f"Training curves — {title}")
    _savefig(fig, out_path, logger)
