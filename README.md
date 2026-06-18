# GNN Embeddings

Comparison of text-classification baselines on Amazon product metadata. The goal is to check if more robust encoders (embeddings) result in more accurate models when dealing with Graph Neural Networks (GNNs).

> Plots comparing encoders across MLP, Logistic Regression and GraphSAGE for each label depth live in [results.ipynb](results.ipynb).

## Scope

The active dataset is the **Toys and Games** Amazon metadata dump (`meta_Toys_and_Games.json.gz`, downloaded on first run from the McAuley Amazon Product Data archive into `data/toys_games.json.gz`). Each product is described by `title + description`; the label is its category at a configurable depth in the Amazon taxonomy.

Amazon stores categories as ordered paths (e.g. `["Toys & Games", "Puzzles", "Jigsaw Puzzles"]`), and `depth` simply picks the index in that path: depth 1 keeps a coarse label, higher depths drill into more specific sub-categories. The selector in `extract_category` only follows branches that start with `Toys & Games`; items whose path is shorter than the requested depth fall back to the deepest leaf of their first branch (see [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py)).

The depth is selected per-run by setting `target_subcategory_depth` in the model entry points ([src/models/mlp.py](src/models/mlp.py), [src/models/logistic_regression.py](src/models/logistic_regression.py), [src/models/sage.py](src/models/sage.py)); each encoder loader is then pointed at a depth-specific cache directory (e.g. `data/tfidf_depth2/`). The default `DEFAULT_SUBCATEGORY_DEPTH` lives in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py) and is only used as a fallback.

Class counts differ slightly between runs because rare classes (`<2` samples) are dropped at split time — see `min_class_count` in [src/datasets/splits.py](src/datasets/splits.py).

## Pipeline

The raw `meta_*.json.gz` file is parsed once into `(asin, text, category, related)` records at the chosen depth. Each encoder turns the texts into a feature matrix `X` and saves `X`, `y`, and the fitted encoder under `data/<encoder>_depth<d>/`. A stratified train/test split (seed `42`) is generated alongside and reused, so every encoder at the same depth is evaluated on the exact same rows. Classifiers then load the cached matrices and split indices — no re-encoding happens at training time.

### Related-products cache (for GraphSAGE)

For the graph model we also need the per-product co-purchase / co-view lists stored in the `related` field of the raw metadata. To avoid re-parsing the multi-GB raw dump on every run, `load_related_data(depth)` in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py) materialises the `(asin, related)` pairs into a pickle at `data/related_depth<d>.pkl` and reuses it from then on. The cache is depth-specific because the row order must align with the encoder caches built at the same depth — the GraphSAGE trainer enforces this with an explicit length check before constructing the edge index.

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
| MLP (2 hidden layers) | [src/models/mlp.py](src/models/mlp.py)                                 | `epochs`, `hidden_dim`, `lr`, `weight_decay`, `dropout`, `val_fraction`, `patience`, `min_delta`, `seed`. |
| GraphSAGE (2 layers)  | [src/models/sage.py](src/models/sage.py)                               | Same knobs as MLP. Builds an undirected `also_bought` graph from the cached `related` pickle and reduces feature dim to 128 via `TruncatedSVD` when the input is sparse or wider than 512. |

All three expose `train_*(X, y, [related,] encoder_name, train_idx, test_idx, ...)` and print Accuracy / Macro-F1 / Weighted-F1 at the end.

### MLP / SAGE early stopping

The MLP and SAGE loops are parameterised so long runs don't overfit:

- `epochs=200` — upper bound; in practice the loop almost never reaches it.
- `val_fraction=0.5` — 50 % of the **test** indices are held out (deterministic via `seed`) as a validation set; only the remaining test rows are used to report the final score.
- `patience=10` — stop after this many epochs without an improvement.
- `min_delta=1e-4` — minimum absolute drop in val loss to count as an improvement.

The best-val-loss `state_dict` is kept in memory and restored before the final test evaluation, so reported scores correspond to the best checkpoint, not the last epoch. Each epoch logs both `train_loss` and `val_loss`; an asterisk marks new best epochs in the log files under [logs/](logs).

## Repo layout

```
data/toys_games.json.gz           raw Amazon Toys & Games metadata (auto-downloaded)
data/<encoder>_depth<d>/          cached features + split indices for label depth <d>
data/related_depth<d>.pkl         cached (asin, related) pairs, used by GraphSAGE
logs/<model>/<encoder>.log        one log per (model, encoder) pair under lr/, mlp/, sage/
src/datasets/                     dataset loader + related-data cache + stratified splits
src/encoders/                     one module per encoder, identical fit/transform/save/load API
src/models/                       classifiers: logistic_regression, mlp, sage (GraphSAGE)
results.ipynb                     parses logs/ and renders the per-encoder comparison plots
```

## Quick start

```bash
pip install -r requirements.txt
python3 -m src.models.logistic_regression
python3 -m src.models.mlp
python3 -m src.models.sage
```

To change the label depth, edit `target_subcategory_depth` at the top of the `__main__` block in [src/models/mlp.py](src/models/mlp.py), [src/models/logistic_regression.py](src/models/logistic_regression.py) or [src/models/sage.py](src/models/sage.py) and re-run. Each encoder loader will look for `data/<encoder>_depth<d>/`; if the directory is missing it is built from scratch at the requested depth, so multiple depths can coexist without manually clearing caches. The same applies to the related-products cache at `data/related_depth<d>.pkl`.

## Results

Numerical scores are appended at the bottom of every log file in `logs/`. Per-encoder loss curves and final-score bar charts are generated by [results.ipynb](results.ipynb).
