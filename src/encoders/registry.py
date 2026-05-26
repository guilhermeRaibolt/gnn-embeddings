"""Encoder registry: build any encoder from a YAML config dict.

Every encoder type supported by the project is registered here so that
experiment scripts only need to call ``build_encoder(cfg, cache_dir)``
rather than containing explicit import branches.

Supported ``type`` values (as written in experiment YAML files)
---------------------------------------------------------------
Text encoders
~~~~~~~~~~~~~
``"BagOfWords"``   → :class:`~src.encoders.bow_tfidf.BagOfWordsEncoder`
``"TFIDF"``        → :class:`~src.encoders.bow_tfidf.TFIDFEncoder`
``"SentenceBERT"`` → :class:`~src.encoders.sbert.SentenceBERTEncoder`
``"Qwen3"``        → :class:`~src.encoders.qwen3.Qwen3Encoder`
``"CLIPText"``     → :class:`~src.encoders.clip_text.CLIPTextEncoder`

Vision encoders
~~~~~~~~~~~~~~~
This project loads **precomputed 4096-dim Caffe visual features** from the
McAuley/UCSD distribution (see :mod:`src.data.image_features`) rather than
encoding images on the fly.  The on-the-fly vision encoders that used to live
here (``ResNet``, ``CLIPVision``, ``DINOv2``) have been moved to ``legacy/``;
see ``legacy/README.md`` for details and reintegration instructions.

YAML example::

    encoders:
      - name: tfidf
        type: TFIDF
        max_features: 10000
      - name: sbert
        type: SentenceBERT
        model_name: sentence-transformers/all-MiniLM-L6-v2
        batch_size: 64
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_encoder(
    config: dict[str, Any],
    cache_dir: str | Path = "data/embeddings",
    device: str = "auto",
) -> Any:
    """Instantiate an encoder from a YAML config dict.

    The ``type`` field selects the encoder class; all remaining keys
    are forwarded as keyword arguments to the constructor (``cache_dir``
    and ``device`` are always injected from the function parameters and
    cannot be overridden by the config dict).

    Args:
        config: Dict with at least a ``"type"`` key, plus any
            encoder-specific constructor arguments.
        cache_dir: Directory for cached embedding tensors.
        device: Inference device (``"auto"``, ``"cpu"``, ``"cuda"``).

    Returns:
        An initialised encoder instance.

    Raises:
        KeyError: If ``config["type"]`` is not registered.
        ImportError: If the required library for the encoder is missing.
    """
    enc_type: str = config["type"]
    kwargs = {k: v for k, v in config.items() if k not in ("type", "name")}

    if enc_type == "BagOfWords":
        from src.encoders.bow_tfidf import BagOfWordsEncoder
        return BagOfWordsEncoder(
            cache_dir=cache_dir,
            device=device,
            max_features=kwargs.get("max_features", 10_000),
            min_df=kwargs.get("min_df", 2),
        )

    if enc_type == "TFIDF":
        from src.encoders.bow_tfidf import TFIDFEncoder
        return TFIDFEncoder(
            cache_dir=cache_dir,
            device=device,
            max_features=kwargs.get("max_features", 10_000),
            min_df=kwargs.get("min_df", 2),
        )

    if enc_type == "SentenceBERT":
        from src.encoders.sbert import SentenceBERTEncoder
        return SentenceBERTEncoder(
            cache_dir=cache_dir,
            device=device,
            model_name=kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=kwargs.get("batch_size", 64),
            normalize_embeddings=kwargs.get("normalize_embeddings", True),
        )

    if enc_type == "Qwen3":
        from src.encoders.qwen3 import Qwen3Encoder
        return Qwen3Encoder(
            cache_dir=cache_dir,
            device=device,
            model_size=kwargs.get("model_size", "0.6B"),
            batch_size=kwargs.get("batch_size", 16),
            pooling=kwargs.get("pooling", "last_token"),
            normalize_embeddings=kwargs.get("normalize_embeddings", True),
        )

    if enc_type == "CLIPText":
        from src.encoders.clip_text import CLIPTextEncoder
        return CLIPTextEncoder(
            cache_dir=cache_dir,
            device=device,
            model_name=kwargs.get("model_name", "openai/clip-vit-base-patch32"),
            batch_size=kwargs.get("batch_size", 64),
            normalize_embeddings=kwargs.get("normalize_embeddings", True),
        )

    # ── Vision encoders ──────────────────────────────────────────────────────
    # Removed: ResNet, CLIPVision, DINOv2 — see legacy/README.md.  The H2
    # experiment now uses precomputed Caffe features loaded by
    # ``src/data/image_features.py`` instead of an on-the-fly vision encoder.

    raise KeyError(
        f"Unknown encoder type '{enc_type}'. "
        f"Registered types: BagOfWords, TFIDF, SentenceBERT, Qwen3, CLIPText."
    )
