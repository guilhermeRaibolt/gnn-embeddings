# GNN Embeddings

Comparison of text-classification baselines on Amazon product metadata. The goal is to check if more robust encoders (embeddings) result in more accurate models when dealing with Graph Neural Networks (GNNs).

>Plots comparing encoders across MLP and Logistic Regression for each label depth live in [results.ipynb](results.ipynb).

## Scope

Each product is described by `title + description`; the label is its category at a configurable depth in the Amazon taxonomy. Amazon stores categories as ordered paths (e.g. `["Musical Instruments", "Guitars", "Acoustic Guitars"]`), and `depth` simply picks the index in that path: depth 1 keeps a coarse label, higher depths drill into more specific sub-categories. Items whose path is shorter than the requested depth fall back to the deepest leaf of their first branch (see `extract_category` in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py)).

The depth is selected per-run by setting `target_subcategory_depth` in the model entry points ([src/models/mlp.py](src/models/mlp.py), [src/models/logistic_regression.py](src/models/logistic_regression.py)); each encoder loader is then pointed at a depth-specific cache directory (e.g. `data/tfidf_depth2/`). The default `DEFAULT_SUBCATEGORY_DEPTH` lives in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py) and is only used as a fallback. Three depths are evaluated:

| Depth | # classes (Musical Instruments subset) |
| ----- | -------------------------------------- |
| 1     | ~48                                    |
| 2     | ~239                                   |
| 3     | ~545                                   |

Class counts differ slightly between runs because rare classes (`<2` samples) are dropped at split time — see `min_class_count` in [src/datasets/splits.py](src/datasets/splits.py).

## Pipeline

The raw `meta_*.json.gz` files are parsed once into `(text, category)` pairs at the chosen depth. Each encoder turns the texts into a feature matrix `X` and saves `X`, `y`, and the fitted encoder under `data/<encoder>_depth<d>/`. A stratified train/test split (seed `42`) is generated alongside and reused, so every encoder at the same depth is evaluated on the exact same rows. Classifiers then load the cached matrices and split indices — no re-encoding happens at training time.

## Encoders compared

| Encoder | Module | Default model | Output dim | Notes |
| ------- | ------ | ------------- | ---------- | ----- |
| BOW     | [src/encoders/bow.py](src/encoders/bow.py)       | sklearn `CountVectorizer`        | sparse | baseline |
| TF-IDF  | [src/encoders/tf_idf.py](src/encoders/tf_idf.py) | sklearn `TfidfVectorizer`        | sparse | baseline |
| SBERT   | [src/encoders/sbert.py](src/encoders/sbert.py)   | `all-MiniLM-L6-v2` (~22M params) | 384    | `DEFAULT_BERT_MODEL`, `batch_size=64` |
| Qwen    | [src/encoders/qwen.py](src/encoders/qwen.py)     | `Qwen/Qwen3-Embedding-0.6B`      | 1024   | `DEFAULT_QWEN_MODEL`, `batch_size=32`, `max_seq_length=512`, fp16 on CUDA |

The Qwen variant was deliberately downsized from the original `Qwen3-Embedding-4B` to `Qwen3-Embedding-0.6B`: at ~85k inputs the 4B model needed ~6 h per encoding pass on a single GPU (and exceeded the SLURM time limit), while 0.6B finishes in minutes with only a small quality drop. Swap `DEFAULT_QWEN_MODEL` at the top of [src/encoders/qwen.py](src/encoders/qwen.py) to try a different size; bump `batch_size` accordingly to the available VRAM.

## Classifiers

| Classifier            | Module                                                                 | Key knobs |
| --------------------- | ---------------------------------------------------------------------- | --------- |
| Logistic Regression   | [src/models/logistic_regression.py](src/models/logistic_regression.py) | `max_iter=1000`, `random_state=42`. Sklearn solver, CPU only. |
| MLP (2 hidden layers) | [src/models/mlp.py](src/models/mlp.py)                                 | `epochs`, `batch_size`, `lr`, `weight_decay`, `val_fraction`, `patience`, `min_delta`, `seed`. |

Both expose `train_*(X, y, encoder_name, train_idx, test_idx, ...)` and print Accuracy / Macro-F1 / Weighted-F1 at the end.

### MLP early stopping

The MLP loop is parameterised so long runs don't overfit:

- `epochs=200` — upper bound; in practice the loop almost never reaches it.
- `val_fraction=0.1` — 10 % of the training indices are held out (deterministic via `seed`) as a validation set; the test split is untouched.
- `patience=10` — stop after this many epochs without an improvement.
- `min_delta=1e-4` — minimum absolute drop in val loss to count as an improvement.

The best-val-loss `state_dict` is kept in memory and restored before the final test evaluation, so reported scores correspond to the best checkpoint, not the last epoch. Each epoch logs both `train_loss` and `val_loss`; an asterisk marks new best epochs in the log files under [logs/](logs).

## Repo layout

```
data/<encoder>_depth<d>/   cached features + split indices for label depth <d>
data/<encoder>/            legacy cache (no depth suffix, kept for backwards compatibility)
logs/<model>/depth<d>_<n>classes/<encoder>.log
src/datasets/              dataset loader (Amazon meta_*.json.gz) + stratified split helpers
src/encoders/              one module per encoder, identical fit/transform/save/load API
src/models/                classifiers (logistic_regression, mlp); gnn.py reserved for the graph model
results.ipynb              parses logs/ and renders the per-depth comparison plots
```

## Quick start

```bash
pip install -r requirements.txt
python3 -m src.models.logistic_regression
python3 -m src.models.mlp
```

To change the label depth, edit `target_subcategory_depth` at the top of the `__main__` block in [src/models/mlp.py](src/models/mlp.py) or [src/models/logistic_regression.py](src/models/logistic_regression.py) and re-run. Each encoder loader will look for `data/<encoder>_depth<d>/`; if the directory is missing it is built from scratch at the requested depth, so multiple depths can coexist without manually clearing caches.

## Results

Numerical scores are appended at the bottom of every log file in `logs/`. Per-encoder loss curves and final-score bar charts (per depth) are generated by [results.ipynb](results.ipynb).
