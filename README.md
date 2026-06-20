# Text-Embedding Scale vs. GNN Node Classification

A modular, reproducible pipeline that studies **how the scale of text embeddings
affects GNN performance** on a node-classification task built from the Amazon
**"Toys and Games"** product catalogue.

* **Nodes** — products (title + brand + description; category path is excluded
  from encoder input to avoid leaking the label).
* **Edges** — the `bought_together` co-purchase signal.
* **Labels** — the 2nd-level subcategory of the path rooted at *Toys & Games*
  (products only cross-listed under other top categories such as *Musical
  Instruments* are dropped).
* **Split** — per-class 60 / 20 / 20 train/val/test.

Encoders compared (all **frozen**, extracted offline once and cached):

| Family   | Encoder                         |
|----------|---------------------------------|
| Lexical  | Bag-of-Words, TF-IDF            |
| Dense    | sBERT `all-MiniLM-L6-v2` (batch 256) |
| LLM      | Qwen3-Embedding 0.6B → 4B → 8B  |

For each encoder the GNN (GraphSAGE or GCN) hyper-parameters are tuned
independently, then the best config is re-run over **5 seeds** to report the mean
test accuracy and a **95 % confidence interval**.

## Layout

```
trygnn/
├── main.py            # orchestrator (CLI)
├── run.sh             # Slurm sbatch template
├── requirements.txt
└── src/
    ├── logging_utils.py   # dual console + logs/gnn_experiment_<ts>.log
    ├── config.py          # encoder catalogue, dataset URLs, dataclasses
    ├── data.py            # download, parse, build & cache the graph
    ├── encoders.py        # BoW / TF-IDF / sBERT / Qwen3 extraction (+OOM retry)
    ├── models.py          # GCN / GraphSAGE
    ├── training.py        # training, search, multi-seed eval + CI
    ├── summary.py         # Markdown results table
    └── visualization.py   # accuracy bar chart, t-SNE, training curves
```

Generated artefacts:

```
data/raw/        meta_Toys_and_Games.json.gz
data/processed/  toys_and_games_graph_<text-version>.pt
data/embeddings/ <text-version>/<encoder>_embeds.pt  # computed once, reused
logs/            gnn_experiment_<timestamp>.log, slurm-<jobid>.{out,err}
results/         <encoder>_<gnn>_final.json, summary_<gnn>.json
figures/         accuracy_vs_scale_<gnn>.png,
                 <encoder>_<gnn>_tsne.png, <encoder>_<gnn>_training_curve.png
```

### Figures

* **`accuracy_vs_scale_<gnn>.png`** — headline bar chart of mean test accuracy
  per encoder with 95% CI error bars (Qwen3 scales vs. baselines).
* **`<encoder>_<gnn>_tsne.png`** — 2-D t-SNE of that encoder's node features,
  coloured by class (subsampled, see `--tsne-max-points`).
* **`<encoder>_<gnn>_training_curve.png`** — train/val loss and accuracy over epochs.

Disable all plotting with `--no-plots`.

## Running

On the cluster (recommended):

```bash
sbatch run.sh
```

Locally / interactively:

```bash
conda activate gnn-embedding        # env with torch, PyG, transformers, sklearn, sbert
python main.py --gnn sage --device cuda
```

Useful flags:

```bash
# Only the lexical + sBERT baselines (skip the large LLMs):
python main.py --encoders bow,tfidf,sbert_minilm

# Random search instead of full grid, capped at 6 trials:
python main.py --search-strategy random --max-trials 6

# Rebuild graph / recompute embeddings from scratch:
python main.py --force-rebuild-graph --force-recompute-embeddings
```

## Notes

* **Embedding cache** — re-running reuses `data/embeddings/*.pt`; delete a file
  (or pass `--force-recompute-embeddings`) to recompute.
* **CUDA OOM** — Qwen3 extraction automatically halves the batch size and retries
  on out-of-memory errors; if it still OOMs at batch size 1, request a larger GPU
  or lower `--max-length`.
* **Dataset access** — McAuley per-category files occasionally require manual
  download. If every mirror fails, place the archive at
  `data/raw/meta_Toys_and_Games.json.gz` and re-run.
* **Reference patterns** — PyG node-classification tutorial and OGB node-property
  conventions; Qwen3 embeddings use the official last-token pooling.
```
