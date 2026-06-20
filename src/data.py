"""Dataset download, metadata parsing and "bought_together" graph construction.

Produces a cached payload (dict of tensors + texts) that is encoder-agnostic, so
the graph is built exactly once and reused across every embedding scale.
"""

from __future__ import annotations

import ast
import gzip
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import torch
from torch import Tensor

from .config import TOYS_ROOT_ALIASES


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #
def download_dataset(
    urls: Sequence[str],
    destination: Path,
    logger: logging.Logger,
    retries: int = 3,
) -> Path:
    """Download the metadata archive, trying each mirror with simple back-off."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Using cached dataset archive: %s", destination)
        return destination

    last_error: Optional[Exception] = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                logger.info("Downloading dataset from %s (attempt %d/%d)", url, attempt, retries)
                urlretrieve(url, destination)
                logger.info("Saved dataset archive to %s", destination)
                return destination
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                logger.warning("Download failed from %s (attempt %d): %s", url, attempt, exc)
                if destination.exists():
                    destination.unlink()
                time.sleep(min(5 * attempt, 15))
    raise RuntimeError(
        f"Could not download {destination.name} from any mirror. "
        "Some McAuley category files require manual access; place the file at "
        f"{destination} and re-run."
    ) from last_error


# --------------------------------------------------------------------------- #
# Metadata parsing                                                             #
# --------------------------------------------------------------------------- #
def _parse_line(line: str) -> dict:
    """Older Amazon dumps use Python-dict syntax; newer ones are valid JSON."""
    try:
        return ast.literal_eval(line)
    except (SyntaxError, ValueError):
        return json.loads(line)


def iter_metadata(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if raw_line:
                yield _parse_line(raw_line)


def _normalise_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _norm_category(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def get_category_paths(record: dict) -> List[List[str]]:
    """Return category paths as a list of lists, handling both schema variants."""
    categories = record.get("categories")
    if isinstance(categories, list) and categories and isinstance(categories[0], (list, tuple)):
        return [list(path) for path in categories]
    flat = record.get("category")  # 2018+ flat list variant
    if isinstance(flat, (list, tuple)) and flat:
        return [list(flat)]
    return []


def select_toys_label(record: dict) -> Optional[str]:
    """Label = 2nd-level subcategory of the path rooted at "Toys & Games".

    Returns ``None`` (drop the product) when no category path roots at Toys &
    Games, so products only cross-listed under other top categories are excluded.
    """
    for path in get_category_paths(record):
        if path and _norm_category(path[0]) in TOYS_ROOT_ALIASES and len(path) >= 2:
            label = str(path[1]).strip()
            if label:
                return label
    return None


def build_product_text(record: dict) -> str:
    """Concatenate non-label text fields used by every encoder."""
    title = _normalise_text(record.get("title"))
    brand = _normalise_text(record.get("brand"))
    description = _normalise_text(record.get("description"))

    segments = []
    if title:
        segments.append(f"Title: {title}")
    if brand:
        segments.append(f"Brand: {brand}")
    if description:
        segments.append(f"Description: {description}")
    return "\n".join(segments).strip()


def get_bought_together(record: dict) -> List[str]:
    """Extract the bought_together neighbour ASINs across schema variants."""
    related = record.get("related") or {}
    neighbours = related.get("bought_together")
    if not neighbours:
        neighbours = record.get("bought_together")  # flat 2018+ variant
    if isinstance(neighbours, (list, tuple)):
        return [str(a).strip() for a in neighbours if str(a).strip()]
    return []


# --------------------------------------------------------------------------- #
# Splitting                                                                    #
# --------------------------------------------------------------------------- #
def stratified_split(
    labels: Sequence[int],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-class 60/20/20 split so every class appears in all three masks."""
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:  # guarantee a non-empty test slice
            n_train = max(1, n - 2)
            n_val = max(1, (n - n_train) // 2)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


# --------------------------------------------------------------------------- #
# Graph build + cache                                                          #
# --------------------------------------------------------------------------- #
def build_graph_payload(
    metadata_path: Path,
    cache_path: Path,
    logger: logging.Logger,
    split_seed: int,
    min_class_count: int,
    force_rebuild: bool = False,
) -> dict:
    """Build (or load) the Toys & Games product graph payload."""
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached graph payload from %s", cache_path)
        return torch.load(cache_path, weights_only=False)

    logger.info("Parsing metadata and constructing the bought_together graph ...")
    records: Dict[str, dict] = {}
    label_counter: Counter = Counter()

    for record in iter_metadata(metadata_path):
        asin = str(record.get("asin", "")).strip()
        if not asin:
            continue
        label = select_toys_label(record)
        text = build_product_text(record)
        if not label or not text:
            continue
        records[asin] = {
            "label": label,
            "text": text,
            "bought_together": get_bought_together(record),
        }
        label_counter[label] += 1

    if not records:
        raise RuntimeError("No Toys & Games products were parsed from the metadata.")

    # Drop rare classes that cannot be meaningfully split / evaluated.
    kept_labels = {lbl for lbl, c in label_counter.items() if c >= min_class_count}
    filtered = {a: r for a, r in records.items() if r["label"] in kept_labels}
    logger.info(
        "Retained %d/%d products across %d classes (dropped labels with < %d nodes).",
        len(filtered), len(records), len(kept_labels), min_class_count,
    )

    asins = sorted(filtered)
    asin_to_idx = {asin: i for i, asin in enumerate(asins)}
    label_names = sorted({filtered[a]["label"] for a in asins})
    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}

    labels = [label_to_idx[filtered[a]["label"]] for a in asins]
    texts = [filtered[a]["text"] for a in asins]

    # Undirected edges from bought_together (only between in-scope products).
    edge_set = set()
    for asin in asins:
        src = asin_to_idx[asin]
        for neighbour in filtered[asin]["bought_together"]:
            dst = asin_to_idx.get(neighbour)
            if dst is None or dst == src:
                continue
            edge_set.add(tuple(sorted((src, dst))))

    if not edge_set:
        raise RuntimeError("No edges were constructed from bought_together relations.")

    directed: List[Tuple[int, int]] = []
    for src, dst in sorted(edge_set):
        directed.append((src, dst))
        directed.append((dst, src))
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()

    train_mask, val_mask, test_mask = stratified_split(labels, seed=split_seed)

    payload = {
        "asins": asins,
        "texts": texts,
        "y": torch.tensor(labels, dtype=torch.long),
        "label_names": label_names,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_nodes": len(asins),
        "num_edges": int(edge_index.size(1)),
        "num_classes": len(label_names),
        "split_seed": split_seed,
        "min_class_count": min_class_count,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    logger.info(
        "Graph cached at %s | nodes=%d | directed_edges=%d | classes=%d",
        cache_path, payload["num_nodes"], payload["num_edges"], payload["num_classes"],
    )
    return payload
