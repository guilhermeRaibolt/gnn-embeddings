"""Training, hyper-parameter search and multi-seed evaluation utilities."""

from __future__ import annotations

import copy
import itertools
import logging
import math
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.data import Data

from .config import SearchConfig
from .models import build_model


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data assembly                                                                #
# --------------------------------------------------------------------------- #
def build_data_object(graph_payload: dict, node_features: Tensor) -> Data:
    if node_features.size(0) != graph_payload["num_nodes"]:
        raise ValueError("Node feature row count does not match the graph node count.")
    return Data(
        x=node_features.float(),
        edge_index=graph_payload["edge_index"].long(),
        y=graph_payload["y"].long(),
        train_mask=graph_payload["train_mask"].bool(),
        val_mask=graph_payload["val_mask"].bool(),
        test_mask=graph_payload["test_mask"].bool(),
    )


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def _evaluate(model: nn.Module, data: Data, mask: Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[mask]
        labels = data.y[mask]
        loss = F.cross_entropy(logits, labels).item()
        acc = _accuracy(logits, labels)
    return acc, loss


# --------------------------------------------------------------------------- #
# Single training run                                                          #
# --------------------------------------------------------------------------- #
def train_single_run(
    data: Data,
    gnn_type: str,
    config: SearchConfig,
    seed: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: str,
    logger: logging.Logger,
    log_every: int,
) -> Dict:
    """Train one model with early stopping on validation accuracy."""
    seed_everything(seed)
    local = copy.deepcopy(data).to(device)
    model = build_model(
        gnn_type=gnn_type,
        in_channels=local.num_node_features,
        hidden_channels=config.hidden_channels,
        out_channels=int(local.y.max().item()) + 1,
        dropout=config.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)

    best_state: Optional[dict] = None
    best_epoch, best_val_acc, best_val_loss, wait = -1, -1.0, float("inf"), 0
    history: List[Dict] = []  # per-epoch curve for visualisation

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local.x, local.edge_index)
        loss = F.cross_entropy(logits[local.train_mask], local.y[local.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = _accuracy(logits[local.train_mask].detach(), local.y[local.train_mask])
        val_acc, val_loss = _evaluate(model, local, local.val_mask)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        improved = (val_acc > best_val_acc) or (
            math.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
        )
        if improved:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == max_epochs:
            logger.info(
                "epoch=%03d | seed=%d | lr=%.4g hidden=%d dropout=%.2f | "
                "train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch, seed, config.learning_rate, config.hidden_channels, config.dropout,
                loss.item(), train_acc, val_loss, val_acc,
            )

        if wait >= patience:
            logger.info("Early stopping at epoch %d (best epoch=%d).", epoch, best_epoch)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    train_acc, train_loss = _evaluate(model, local, local.train_mask)
    val_acc, val_loss = _evaluate(model, local, local.val_mask)
    test_acc, test_loss = _evaluate(model, local, local.test_mask)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_acc": train_acc, "train_loss": train_loss,
        "val_acc": val_acc, "val_loss": val_loss,
        "test_acc": test_acc, "test_loss": test_loss,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Hyper-parameter search                                                       #
# --------------------------------------------------------------------------- #
def generate_search_space(
    learning_rates: Sequence[float],
    hidden_channels: Sequence[int],
    dropouts: Sequence[float],
    strategy: str = "grid",
    max_trials: int = 0,
    seed: int = 42,
) -> List[SearchConfig]:
    configs = [
        SearchConfig(lr, hidden, drop)
        for lr, hidden, drop in itertools.product(learning_rates, hidden_channels, dropouts)
    ]
    if strategy == "random" and 0 < max_trials < len(configs):
        configs = random.Random(seed).sample(configs, max_trials)
    return configs


def tune_hyperparameters(
    data: Data,
    gnn_type: str,
    search_space: Sequence[SearchConfig],
    tuning_seed: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: str,
    logger: logging.Logger,
    log_every: int,
) -> Tuple[SearchConfig, List[Dict]]:
    """Grid/random search; selects the config with the best validation accuracy."""
    results: List[Dict] = []
    best_config: Optional[SearchConfig] = None
    best_val_acc, best_val_loss = -1.0, float("inf")

    logger.info("Hyper-parameter search over %d configurations.", len(search_space))
    for i, config in enumerate(search_space, start=1):
        logger.info(
            "Trial %d/%d | lr=%.4g hidden=%d dropout=%.2f",
            i, len(search_space), config.learning_rate, config.hidden_channels, config.dropout,
        )
        metrics = train_single_run(
            data, gnn_type, config, tuning_seed, max_epochs, patience,
            weight_decay, device, logger, log_every,
        )
        trial_row = {**asdict(config), **metrics}
        trial_row.pop("history", None)  # keep tuning JSON compact
        results.append(trial_row)
        improved = (metrics["val_acc"] > best_val_acc) or (
            math.isclose(metrics["val_acc"], best_val_acc) and metrics["val_loss"] < best_val_loss
        )
        if improved:
            best_config, best_val_acc, best_val_loss = config, metrics["val_acc"], metrics["val_loss"]

    if best_config is None:
        raise RuntimeError("Hyper-parameter search selected no configuration.")
    logger.info(
        "Best config | lr=%.4g hidden=%d dropout=%.2f | val_acc=%.4f",
        best_config.learning_rate, best_config.hidden_channels, best_config.dropout, best_val_acc,
    )
    return best_config, results


# --------------------------------------------------------------------------- #
# Multi-seed evaluation + confidence interval                                  #
# --------------------------------------------------------------------------- #
def mean_confidence_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Mean and half-width of the (Student-t) confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    sem = float(arr.std(ddof=1)) / math.sqrt(arr.size)
    try:
        from scipy import stats

        half = float(stats.t.ppf((1 + confidence) / 2, arr.size - 1)) * sem
    except Exception:  # fall back to normal approximation if scipy is absent
        half = 1.96 * sem
    return mean, half


def evaluate_with_seeds(
    data: Data,
    gnn_type: str,
    config: SearchConfig,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: str,
    logger: logging.Logger,
    log_every: int,
) -> Tuple[float, float, List[Dict]]:
    """Run the tuned config with several seeds; return mean test acc + 95% CI."""
    per_seed: List[Dict] = []
    for seed in seeds:
        logger.info("Final evaluation with seed=%d", seed)
        metrics = train_single_run(
            data, gnn_type, config, seed, max_epochs, patience,
            weight_decay, device, logger, log_every,
        )
        per_seed.append(metrics)
        logger.info(
            "  seed=%d done | best_epoch=%d | val_acc=%.4f | test_acc=%.4f",
            seed, metrics["best_epoch"], metrics["val_acc"], metrics["test_acc"],
        )
    test_accs = [m["test_acc"] for m in per_seed]
    mean_acc, ci95 = mean_confidence_interval(test_accs)
    return mean_acc, ci95, per_seed
