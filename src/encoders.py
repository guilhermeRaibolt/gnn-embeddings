"""Offline (frozen) feature extraction for every encoder scale.

Each extractor returns an ``(N, D)`` float tensor of node features and caches it to
``data/embeddings/<safe_name>_embeds.pt`` so it is only ever computed once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import EncoderSpec

QWEN_INSTRUCTION = (
    "Represent the Amazon product for node classification by product subcategory."
)


# --------------------------------------------------------------------------- #
# Cache helpers                                                                #
# --------------------------------------------------------------------------- #
def _load_cached(cache_path: Path, expected_rows: int, logger: logging.Logger) -> Tensor | None:
    if not cache_path.exists():
        return None
    logger.info("Loading cached embeddings from %s", cache_path)
    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    emb = saved["embeddings"] if isinstance(saved, dict) else saved
    if emb.size(0) != expected_rows:
        raise ValueError(
            f"Cached embeddings at {cache_path} have {emb.size(0)} rows "
            f"but the graph has {expected_rows} nodes; delete the cache and re-run."
        )
    return emb.float()


def _save(cache_path: Path, embeddings: Tensor, model_id: str, logger: logging.Logger) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "num_texts": int(embeddings.size(0)),
            "dimension": int(embeddings.size(1)),
            "embeddings": embeddings,
        },
        cache_path,
    )
    logger.info("Saved embeddings to %s with shape %s", cache_path, tuple(embeddings.shape))


# --------------------------------------------------------------------------- #
# Lexical baselines: BoW and TF-IDF                                            #
# --------------------------------------------------------------------------- #
def _extract_sparse(
    texts: Sequence[str], kind: str, max_features: int, logger: logging.Logger
) -> Tensor:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    logger.info("Fitting %s vectoriser (max_features=%d) ...", kind.upper(), max_features)
    if kind == "tfidf":
        vectoriser = TfidfVectorizer(max_features=max_features, stop_words="english")
    elif kind == "bow":
        vectoriser = CountVectorizer(max_features=max_features, stop_words="english")
    else:  # pragma: no cover - guarded by dispatch
        raise ValueError(f"Unsupported sparse kind: {kind}")

    matrix = vectoriser.fit_transform(texts)  # scipy CSR
    dense = torch.from_numpy(matrix.toarray()).float()
    # L2-normalise rows so feature magnitude is comparable across encoders.
    return F.normalize(dense, p=2, dim=1)


# --------------------------------------------------------------------------- #
# Dense semantic baseline: sentence-transformers                              #
# --------------------------------------------------------------------------- #
def _extract_sbert(
    texts: Sequence[str], model_id: str, batch_size: int, device: str, logger: logging.Logger
) -> Tensor:
    from sentence_transformers import SentenceTransformer

    logger.info("Encoding %d texts with sBERT %s (batch=%d) ...", len(texts), model_id, batch_size)
    model = SentenceTransformer(model_id, device=device)
    with torch.no_grad():
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    embeddings = embeddings.detach().cpu().float()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return embeddings


# --------------------------------------------------------------------------- #
# Qwen3 embedding models (transformers, last-token pooling)                    #
# --------------------------------------------------------------------------- #
def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Official Qwen3-Embedding pooling: take the final non-pad token."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    batch = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), seq_len]


def _torch_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _extract_qwen3(
    texts: Sequence[str],
    model_id: str,
    batch_size: int,
    max_length: int,
    device: str,
    logger: logging.Logger,
) -> Tensor:
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading Qwen3 embedding model %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id, torch_dtype=_torch_dtype(device)).to(device)
    model.eval()

    prepared = [f"Instruct: {QWEN_INSTRUCTION}\nQuery: {t}" for t in texts]
    chunks: List[Tensor] = []
    cursor = 0
    current_bs = batch_size

    while cursor < len(prepared):
        take = min(current_bs, len(prepared) - cursor)
        batch = prepared[cursor : cursor + take]
        try:
            tokens = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**tokens)
                emb = _last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.detach().cpu().float())
            cursor += take
            if cursor == len(prepared) or cursor % max(batch_size * 20, 1) == 0:
                logger.info("  %s progress: %d/%d", model_id, cursor, len(prepared))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if take == 1:
                raise RuntimeError(
                    f"{model_id} OOMs at batch size 1. Use a larger GPU or lower --max-length."
                ) from exc
            current_bs = max(1, take // 2)
            logger.warning("CUDA OOM for %s; reducing batch size to %d.", model_id, current_bs)

    stacked = torch.cat(chunks, dim=0)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return stacked


# --------------------------------------------------------------------------- #
# Public dispatch                                                              #
# --------------------------------------------------------------------------- #
def extract_features(
    spec: EncoderSpec,
    texts: Sequence[str],
    cache_path: Path,
    *,
    device: str,
    logger: logging.Logger,
    sbert_batch_size: int = 256,
    qwen_batch_size: int = 16,
    max_length: int = 1024,
    force_recompute: bool = False,
) -> Tensor:
    """Compute or load cached node features for ``spec``."""
    if not force_recompute:
        cached = _load_cached(cache_path, len(texts), logger)
        if cached is not None:
            return cached

    logger.info("Extracting features with encoder '%s' (kind=%s)", spec.name, spec.kind)
    if spec.kind in ("bow", "tfidf"):
        embeddings = _extract_sparse(texts, spec.kind, spec.max_features or 4096, logger)
    elif spec.kind == "sbert":
        embeddings = _extract_sbert(texts, spec.hf_id, sbert_batch_size, device, logger)
    elif spec.kind == "qwen3":
        embeddings = _extract_qwen3(texts, spec.hf_id, qwen_batch_size, max_length, device, logger)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported encoder kind: {spec.kind}")

    _save(cache_path, embeddings, spec.hf_id or spec.kind, logger)
    return embeddings
