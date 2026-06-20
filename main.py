#!/usr/bin/env python3
"""End-to-end study: how text-embedding scale affects GNN node classification.

Task: node classification on the Amazon "Toys and Games" product graph, where
edges come from the ``bought_together`` signal and labels are the 2nd-level
subcategory of the path rooted at "Toys & Games".

Encoders compared (each frozen, extracted offline once, then cached):
    BoW, TF-IDF, sBERT (all-MiniLM-L6-v2), Qwen3-0.6B, Qwen3-4B, Qwen3-8B.

For every encoder we independently grid/random-search the GNN hyper-parameters,
then re-train the best config across 5 seeds and report mean test accuracy with a
95% confidence interval. A Markdown summary table is written at the end of the log.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch

from src.config import (
    DATASET_URLS,
    DEFAULT_ENCODERS,
    GRAPH_CACHE_FILENAME,
    RAW_METADATA_FILENAME,
    TEXT_INPUT_VERSION,
    get_encoder,
)
from src.data import build_graph_payload, download_dataset
from src.encoders import extract_features
from src.logging_utils import setup_logging
from src.summary import log_markdown_summary
from src.visualization import (
    plot_accuracy_vs_scale,
    plot_embedding_tsne,
    plot_training_curves,
)
from src.training import (
    build_data_object,
    evaluate_with_seeds,
    generate_search_space,
    seed_everything,
    tune_hyperparameters,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = Path(__file__).resolve().parent
    p.add_argument("--data-dir", type=Path, default=root / "data")
    p.add_argument("--logs-dir", type=Path, default=root / "logs")
    p.add_argument("--results-dir", type=Path, default=root / "results")
    p.add_argument("--figures-dir", type=Path, default=root / "figures")
    p.add_argument("--no-plots", action="store_true", help="Disable figure generation.")
    p.add_argument("--tsne-max-points", type=int, default=2000, help="t-SNE subsample size.")
    p.add_argument("--gnn", choices=["sage", "gcn"], default="sage", help="GNN backbone.")
    p.add_argument(
        "--encoders",
        type=str,
        default=",".join(s.safe_name for s in DEFAULT_ENCODERS),
        help="Comma-separated encoder safe-names to run, in order.",
    )
    # Training
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--log-every", type=int, default=20)
    # Hyper-parameter search
    p.add_argument("--learning-rates", type=str, default="0.01,0.005,0.001")
    p.add_argument("--hidden-channels", type=str, default="128,256")
    p.add_argument("--dropouts", type=str, default="0.2,0.5")
    p.add_argument("--search-strategy", choices=["grid", "random"], default="grid")
    p.add_argument("--max-trials", type=int, default=0, help="Cap for random search (0 = no cap).")
    # Seeds / splitting
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tuning-seed", type=int, default=42)
    p.add_argument("--eval-seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--min-class-count", type=int, default=10)
    # Feature extraction
    p.add_argument("--sbert-batch-size", type=int, default=256)
    p.add_argument("--qwen-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    # Caching / device
    p.add_argument("--force-rebuild-graph", action="store_true")
    p.add_argument("--force-recompute-embeddings", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    embeddings_dir = data_dir / "embeddings"

    logger, log_path = setup_logging(args.logs_dir)
    logger.info("Log file: %s", log_path)
    logger.info("Arguments: %s", vars(args))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    logger.info("Device: %s", args.device)

    seed_everything(args.split_seed)

    # ----- Stage 1: dataset + graph ---------------------------------------- #
    logger.info("=" * 90)
    logger.info("Stage 1/4 | Dataset download and graph construction")
    metadata_path = raw_dir / RAW_METADATA_FILENAME
    download_dataset(DATASET_URLS, metadata_path, logger)
    graph = build_graph_payload(
        metadata_path=metadata_path,
        cache_path=processed_dir / GRAPH_CACHE_FILENAME,
        logger=logger,
        split_seed=args.split_seed,
        min_class_count=args.min_class_count,
        force_rebuild=args.force_rebuild_graph,
    )
    logger.info(
        "Graph ready | nodes=%d edges=%d classes=%d | train/val/test=%d/%d/%d",
        graph["num_nodes"], graph["num_edges"], graph["num_classes"],
        int(graph["train_mask"].sum()), int(graph["val_mask"].sum()), int(graph["test_mask"].sum()),
    )

    # ----- Shared search space + seeds ------------------------------------- #
    search_space = generate_search_space(
        learning_rates=_floats(args.learning_rates),
        hidden_channels=_ints(args.hidden_channels),
        dropouts=_floats(args.dropouts),
        strategy=args.search_strategy,
        max_trials=args.max_trials,
        seed=args.tuning_seed,
    )
    eval_seeds = _ints(args.eval_seeds)
    encoder_specs = [get_encoder(name.strip()) for name in args.encoders.split(",") if name.strip()]

    # ----- Stages 2-4: per-encoder extract -> tune -> seed evaluation ------ #
    final_results: List[dict] = []
    for spec in encoder_specs:
        logger.info("=" * 90)
        logger.info("Encoder '%s' (kind=%s, scale=%s)", spec.name, spec.kind, spec.scale)

        # Stage 2: frozen feature extraction (cached).
        features = extract_features(
            spec,
            graph["texts"],
            cache_path=embeddings_dir / TEXT_INPUT_VERSION / f"{spec.safe_name}_embeds.pt",
            device=args.device,
            logger=logger,
            sbert_batch_size=args.sbert_batch_size,
            qwen_batch_size=args.qwen_batch_size,
            max_length=args.max_length,
            force_recompute=args.force_recompute_embeddings,
        )
        data = build_data_object(graph, features)

        # Stage 3: independent hyper-parameter search.
        best_config, tuning_results = tune_hyperparameters(
            data, args.gnn, search_space, args.tuning_seed, args.max_epochs,
            args.patience, args.weight_decay, args.device, logger, args.log_every,
        )

        # Stage 4: final tuned config across multiple seeds.
        mean_acc, ci95, per_seed = evaluate_with_seeds(
            data, args.gnn, best_config, eval_seeds, args.max_epochs,
            args.patience, args.weight_decay, args.device, logger, args.log_every,
        )

        summary_row = {
            "name": spec.name,
            "safe_name": spec.safe_name,
            "kind": spec.kind,
            "scale": spec.scale,
            "hf_id": spec.hf_id,
            "embedding_dim": int(features.size(1)),
            "best_config": asdict(best_config),
            "mean_test_accuracy": mean_acc,
            "ci95": ci95,
            "seed_test_accuracies": [m["test_acc"] for m in per_seed],
            "per_seed_results": per_seed,
            "tuning_results": tuning_results,
        }
        final_results.append(summary_row)

        # Per-encoder figures: feature t-SNE + training curve (best/first seed).
        if not args.no_plots:
            fig_prefix = f"{spec.safe_name}_{args.gnn}"
            try:
                plot_embedding_tsne(
                    features, graph["y"], graph["label_names"],
                    args.figures_dir / f"{fig_prefix}_tsne.png", spec.name, logger,
                    max_points=args.tsne_max_points,
                )
                plot_training_curves(
                    per_seed[0].get("history", []),
                    args.figures_dir / f"{fig_prefix}_training_curve.png",
                    f"{spec.name} ({args.gnn.upper()})", logger,
                )
            except Exception as plot_exc:  # never let a plot failure abort the run
                logger.warning("Figure generation for '%s' failed: %s", spec.name, plot_exc)

        # Drop bulky per-epoch history before serialising the JSON.
        compact = dict(summary_row)
        compact["per_seed_results"] = [
            {k: v for k, v in m.items() if k != "history"} for m in per_seed
        ]
        write_json(compact, args.results_dir / f"{spec.safe_name}_{args.gnn}_final.json")
        logger.info("Completed '%s' | mean_test_acc=%.4f | 95%% CI=± %.4f", spec.name, mean_acc, ci95)

    # ----- Combined summary ------------------------------------------------ #
    combined = [
        {**r, "per_seed_results": [{k: v for k, v in m.items() if k != "history"}
                                   for m in r["per_seed_results"]]}
        for r in final_results
    ]
    write_json({"gnn": args.gnn, "results": combined}, args.results_dir / f"summary_{args.gnn}.json")
    if not args.no_plots and final_results:
        try:
            plot_accuracy_vs_scale(
                final_results, args.figures_dir / f"accuracy_vs_scale_{args.gnn}.png",
                args.gnn, logger,
            )
        except Exception as plot_exc:
            logger.warning("Accuracy chart generation failed: %s", plot_exc)
    log_markdown_summary(final_results, args.gnn, logger)
    logger.info("All encoders complete. Full log at %s", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # top-level safety net so failures are logged
        logger = logging.getLogger("gnn_experiment")
        if logger.handlers:
            logger.exception("Pipeline failed: %s", exc)
        else:
            print(f"Pipeline failed before logger setup: {exc}", file=sys.stderr)
        raise
