"""Static configuration: encoder catalogue, dataset locations and search dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Dataset (Amazon product metadata, McAuley lab "Toys and Games" category).
# Source index: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
# The per-category metadata files use the 2014 schema:
#   {asin, title, description, brand, categories: [[...]], related: {bought_together: [...]}}
# Several mirrors are tried in order; the first reachable one wins.
# ---------------------------------------------------------------------------
DATASET_URLS: List[str] = [
    "https://jmcauley.ucsd.edu/pml_data/meta_Toys_and_Games.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
]

RAW_METADATA_FILENAME = "meta_Toys_and_Games.json.gz"
TEXT_INPUT_VERSION = "no_category_flat_v1"
GRAPH_CACHE_FILENAME = f"toys_and_games_graph_{TEXT_INPUT_VERSION}.pt"

# The top-level ("root") category that defines an in-scope node. Records whose
# first category path does NOT root here (e.g. products cross-listed under
# "Musical Instruments") are dropped, and the label is the 2nd-level subcategory.
TOYS_ROOT_ALIASES = {"toys & games"}


@dataclass(frozen=True)
class EncoderSpec:
    """One frozen text encoder to benchmark.

    kind:
        "bow"    - bag-of-words counts (sklearn CountVectorizer)
        "tfidf"  - TF-IDF weighted counts (sklearn TfidfVectorizer)
        "sbert"  - SentenceTransformer dense embeddings
        "qwen3"  - Qwen3 embedding model via Hugging Face transformers
    """

    name: str                       # human-readable label for the report table
    safe_name: str                  # filesystem-safe id used in cache filenames
    kind: str
    hf_id: Optional[str] = None     # Hugging Face repo id (sbert / qwen3)
    max_features: Optional[int] = None  # vocabulary cap (bow / tfidf)
    scale: str = "-"                # parameter scale, only meaningful for qwen3


# Full catalogue ordered exactly as the pipeline runs: lexical baselines, sBERT,
# then the three Qwen3 scales in ascending size (0.6B -> 4B -> 8B).
DEFAULT_ENCODERS: List[EncoderSpec] = [
    EncoderSpec("BoW", "bow", "bow", max_features=4096),
    EncoderSpec("TF-IDF", "tfidf", "tfidf", max_features=4096),
    EncoderSpec(
        "sBERT-all-MiniLM-L6-v2",
        "sbert_minilm",
        "sbert",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
    EncoderSpec("Qwen3-0.6B", "qwen3_0.6b", "qwen3", hf_id="Qwen/Qwen3-Embedding-0.6B", scale="0.6B"),
    EncoderSpec("Qwen3-4B", "qwen3_4b", "qwen3", hf_id="Qwen/Qwen3-Embedding-4B", scale="4B"),
    EncoderSpec("Qwen3-8B", "qwen3_8b", "qwen3", hf_id="Qwen/Qwen3-Embedding-8B", scale="8B"),
]


def get_encoder(safe_name: str) -> EncoderSpec:
    for spec in DEFAULT_ENCODERS:
        if spec.safe_name == safe_name:
            return spec
    raise KeyError(
        f"Unknown encoder '{safe_name}'. Available: "
        + ", ".join(s.safe_name for s in DEFAULT_ENCODERS)
    )


@dataclass(frozen=True)
class SearchConfig:
    """A single point in the GNN hyper-parameter search space."""

    learning_rate: float
    hidden_channels: int
    dropout: float
