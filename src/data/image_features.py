"""McAuley/UCSD precomputed image-feature loader.

The McAuley group distributes 4096-dim Caffe (CaffeNet / AlexNet-style)
reference visual features alongside the Amazon product metadata at
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html.  Using these
precomputed features lets the H2 experiment skip the URL → JPEG → vision-
encoder pipeline entirely (which is bandwidth-heavy, GPU-heavy, and a source
of the "missing image" confounder reported in the original H2 design).

Binary file format (McAuley convention)
---------------------------------------
Each ``image_features_<Category>.b`` file is a concatenation of fixed-size
records::

    record_i = <ASIN: 10 ASCII bytes><features: 4096 × float32 little-endian>

Total record size = 10 + 4096 × 4 = **16 394 bytes**.  No header, no
trailer.  EOF is the only end-of-file indicator.

Public API
----------
- :func:`load_mcauley_image_features` — read one or more binary files into a
  Python ``dict[str, np.ndarray]`` keyed by ASIN.
- :func:`align_features_to_nodes` — given a list of ASINs in node order and
  the dict above, return a ``(N, D)`` ``torch.Tensor`` plus a boolean
  availability mask.  Missing rows are zero-filled.

Missing-feature policy
----------------------
Some products in the metadata files have no entry in the image-features file
(for example, deleted listings).  Following the design of the legacy
on-the-fly pipeline, missing rows are filled with **zero vectors**.  The H2
runner still performs the "with-images vs without-images" confounder analysis
using the boolean availability mask returned here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)

# McAuley binary layout constants.
ASIN_BYTES: int = 10
FEATURE_DIM: int = 4096
FEATURE_BYTES: int = FEATURE_DIM * 4  # float32
RECORD_BYTES: int = ASIN_BYTES + FEATURE_BYTES  # 16 394


def _resolve_features_path(raw_dir: Path, category: str) -> Path:
    """Return the on-disk path to ``image_features_<Category>.b``.

    Args:
        raw_dir: Directory holding the raw McAuley files (typically
            ``<root>/raw``).
        category: Top-level category name (matches the ``meta_<Category>.json``
            convention used by :class:`~src.data.amazon_dataset.AmazonCopurchase`).

    Returns:
        Absolute path to the binary file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = raw_dir / f"image_features_{category}.b"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing McAuley image-features file: {path}\n"
            f"Download 'image_features_{category}.b' from "
            f"https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html "
            f"and place it in {raw_dir}/."
        )
    return path


def load_mcauley_image_features(
    paths: Path | Iterable[Path],
) -> dict[str, np.ndarray]:
    """Read one or more McAuley binary feature files into a dict.

    The function streams the file in fixed-size chunks so memory usage is
    bounded by a few records at a time.  Float vectors are stored as
    ``np.float32`` arrays of shape ``(4096,)``.

    Args:
        paths: A single path or an iterable of paths to
            ``image_features_<Category>.b`` files.

    Returns:
        Dict mapping ``asin`` (10-char ASCII string) to a 4096-dim
        ``np.float32`` array.  When the same ASIN appears in multiple files
        (very rare across distinct categories), the **last** occurrence wins.

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError: If a record is truncated (file length not a multiple of
            16 394 bytes).
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    features: dict[str, np.ndarray] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        n_bytes = path.stat().st_size
        if n_bytes % RECORD_BYTES != 0:
            raise ValueError(
                f"{path} is {n_bytes} bytes — not a multiple of "
                f"{RECORD_BYTES}.  File is likely truncated or corrupt."
            )
        n_records_expected = n_bytes // RECORD_BYTES
        n_records_read = 0
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(RECORD_BYTES)
                if not chunk:
                    break
                if len(chunk) != RECORD_BYTES:
                    raise ValueError(
                        f"{path}: truncated record at byte "
                        f"{n_records_read * RECORD_BYTES}"
                    )
                asin = chunk[:ASIN_BYTES].decode("ascii", errors="replace").strip()
                vec = np.frombuffer(chunk[ASIN_BYTES:], dtype=np.float32).copy()
                features[asin] = vec
                n_records_read += 1
        logger.info(
            "Loaded %d / %d image-feature records from %s",
            n_records_read, n_records_expected, path.name,
        )
    return features


def align_features_to_nodes(
    asins: list[str],
    features: dict[str, np.ndarray],
    *,
    dim: int = FEATURE_DIM,
) -> tuple[torch.Tensor, np.ndarray]:
    """Order features to match a list of node ASINs; zero-fill missing.

    Args:
        asins: Node ASINs in node-id order (length ``N``).
        features: ASIN → feature vector dict, as returned by
            :func:`load_mcauley_image_features`.
        dim: Expected feature dimensionality.  Defaults to 4096.

    Returns:
        ``(image_emb, available)`` where:

        - ``image_emb`` is a ``(N, dim)`` ``torch.float32`` tensor.  Rows for
          ASINs not found in ``features`` are zero vectors.
        - ``available`` is a ``(N,)`` ``np.bool_`` array — ``True`` where a
          feature vector was found.

    Raises:
        ValueError: If any feature vector in ``features`` has the wrong size.
    """
    n = len(asins)
    out = np.zeros((n, dim), dtype=np.float32)
    available = np.zeros(n, dtype=bool)

    for i, asin in enumerate(asins):
        vec = features.get(asin)
        if vec is None:
            continue
        if vec.shape != (dim,):
            raise ValueError(
                f"Feature vector for asin={asin!r} has shape {vec.shape}, "
                f"expected ({dim},)."
            )
        out[i] = vec
        available[i] = True

    n_avail = int(available.sum())
    logger.info(
        "Aligned %d / %d ASINs to McAuley features (%.1f%% coverage).",
        n_avail, n, 100.0 * n_avail / max(1, n),
    )
    return torch.from_numpy(out), available


def load_and_align(
    raw_dir: Path | str,
    categories: Iterable[str],
    asins: list[str],
) -> tuple[torch.Tensor, np.ndarray]:
    """Convenience wrapper: resolve paths, read files, align to nodes.

    Args:
        raw_dir: Directory holding the McAuley raw files.
        categories: Top-level category names whose feature files to load.
        asins: Node ASINs in node-id order.

    Returns:
        Same as :func:`align_features_to_nodes`.

    Raises:
        FileNotFoundError: If any category's feature file is missing.
    """
    raw_dir = Path(raw_dir)
    paths = [_resolve_features_path(raw_dir, c) for c in categories]
    features = load_mcauley_image_features(paths)
    return align_features_to_nodes(asins, features)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import struct
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    tmp = Path(tempfile.mkdtemp(prefix="image_features_smoke_"))
    bin_path = tmp / "image_features_Test.b"

    # Synthesize 3 records: asin "AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"
    # with features = [k] * 4096 for k in {1.0, 2.0, 3.0}.
    sample_asins = ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"]
    with bin_path.open("wb") as fh:
        for k, asin in enumerate(sample_asins, start=1):
            fh.write(asin.encode("ascii"))
            fh.write(struct.pack("<" + "f" * FEATURE_DIM, *([float(k)] * FEATURE_DIM)))

    feats = load_mcauley_image_features(bin_path)
    assert set(feats.keys()) == set(sample_asins)
    assert feats["AAAAAAAAAA"].shape == (4096,)
    assert np.allclose(feats["BBBBBBBBBB"], 2.0)

    # Ask for the 3 known ASINs + 1 missing.
    node_asins = ["AAAAAAAAAA", "DDDDDDDDDD", "CCCCCCCCCC"]
    emb, avail = align_features_to_nodes(node_asins, feats)
    assert emb.shape == (3, 4096)
    assert emb.dtype == torch.float32
    assert avail.tolist() == [True, False, True]
    assert torch.allclose(emb[0], torch.ones(4096))
    assert torch.allclose(emb[1], torch.zeros(4096))         # missing → zeros
    assert torch.allclose(emb[2], torch.full((4096,), 3.0))

    print("image_features.py smoke test OK")
    shutil.rmtree(tmp, ignore_errors=True)
