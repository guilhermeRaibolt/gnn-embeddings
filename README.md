# GNN Embeddings

Comparison of text-classification baselines on Amazon product metadata. The goal is to check if more robust encoders (embeddings) result in more accurate models when dealing with Graph Neural Networks (GNNs).

>Plots comparing encoders across MLP and Logistic Regression for each label depth live in [results.ipynb](results.ipynb).

>ALERT: For now, this repo is comparing only the non-graph approaches.

## Scope

Each product is described by `title + description`; the label is its category at a configurable depth in the Amazon taxonomy (`SUBCATEGORY_DEPTH` in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py)). Three depths are evaluated:

| Depth | # classes (Musical Instruments subset) |
| ----- | -------------------------------------- |
| 1     | ~48                                    |
| 2     | ~239                                   |
| 3     | ~545                                   |

Class counts differ slightly between runs because rare classes (`<2` samples) are dropped at split time — see `min_class_count` in [src/datasets/splits.py](src/datasets/splits.py).

## Pipeline

```
raw .json.gz  ─►  load_*_features() ─►  cached X, y, encoder ─►  train_{mlp,logistic_regression}()
                       ▲                       │
                       └───── make_split / load_split (stratified, seeded)
```

- **Encoders** ([src/encoders/](src/encoders)) all expose the same `fit_transform(texts)` interface and cache outputs under `data/<encoder>/{X.*,y.npy,encoder.joblib,train_idx.npy,test_idx.npy}`.
- **Splits** ([src/datasets/splits.py](src/datasets/splits.py)) are generated once with `random_state=42`, stratified on `y`, then reused. Re-running an encoder will pick up the existing split, so all four encoders are evaluated on identical train/test rows.
- **Models** ([src/models/](src/models)) read the cached matrices; nothing in `train_*` re-encodes text.

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
data/<encoder>/        cached features + split indices
logs/<model>/depth<d>_<n>classes/<encoder>.log
src/datasets/          dataset loader (Amazon meta_*.json.gz) + stratified split helpers
src/encoders/          one module per encoder, identical fit/transform/save/load API
src/models/            classifiers (logistic_regression, mlp); gnn.py reserved for the graph model
results.ipynb          parses logs/ and renders the per-depth comparison plots
```

## Quick start

```bash
pip install -r requirements.txt
python3 -m src.models.logistic_regression
python3 -m src.models.mlp
```

To change the label depth, edit `SUBCATEGORY_DEPTH` in [src/datasets/amazon_dataset.py](src/datasets/amazon_dataset.py), then **delete the matching `data/<encoder>/` directory** (otherwise the cached `y.npy` is reused) and re-run.

## Results

Numerical scores are appended at the bottom of every log file in `logs/`. Per-encoder loss curves and final-score bar charts (per depth) are generated by [results.ipynb](results.ipynb).
