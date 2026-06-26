# Graph Construction & Text/Multimodal Encoder Benchmarking

*Part of the group project "Better Embeddings for Better GNNs" (Télécom Paris, PRJ_4SD06_TP).*

This notebook covers my contribution to the project: building the co-purchase graph, benchmarking how different text encoders affect GraphSAGE node classification, and extending the pipeline to multimodal (text + image) features.

## What this part does

The project asks a single question: **does giving a GNN richer node features improve product classification?** My work focuses on the graph and the feature side of that question — constructing the graph the team trains on, and systematically swapping in different text and multimodal encoders while keeping the architecture fixed, so that any change in accuracy can be attributed to the features alone.

## Scope of my contribution

- **Graph construction.** Parsing the Amazon Toys & Games metadata, mapping products (ASINs) to node indices, building the co-purchase graph from `also_bought` links, removing self-loops and duplicates, and producing the PyTorch Geometric `Data` object with stratified train/val/test masks.
- **Text encoder benchmark.** Implementing and evaluating GraphSAGE with five text encoders spanning classical to modern: TF-IDF/BoW, BERT (MiniLM and Large), CLIP text, and Qwen3. This tests the encoder-quality axis (H3) together with the pretraining-objective and model-size sub-axes suggested by the supervisor.
- **Multimodal fusion.** Combining text and image embeddings (CLIP text + CLIP image, BERT Large + DINOv2) via concatenation and evaluating them on the same GraphSAGE backbone (H2). Image embeddings were extracted separately by a teammate and consumed here from a shared parquet checkpoint.

The non-graph baselines (MLP, Logistic Regression), the image extraction pipeline itself, and hyperparameter tuning were handled by other team members.

## Key design choices

- **Dataset:** Toys & Games rather than Musical Instruments — at the first subcategory level it gives 18 well-balanced classes instead of a long tail of sparse ones.
- **Labels:** the first subcategory level of the `Toys & Games` hierarchy, dropping products with no subcategory.
- **Edges:** only `also_bought` — using all four relation types produced a 14M-edge graph that was too dense and noisy; restricting to co-purchase kept the signal clean (~7.5M edges).
- **Fixed architecture:** a two-layer GraphSAGE with hidden size 256, trained with early stopping. Only the input features change between experiments, isolating the effect of the encoder.
- **Mini-batch training:** the full graph did not fit in GPU memory, so training uses `NeighborLoader` to sample a fixed neighbourhood per node.

## Results summary

Text encoders (GraphSAGE):

| Encoder | Dim | Test Acc |
|---------|-----|----------|
| TF-IDF / BoW | 128 | 73.1% |
| BERT MiniLM | 384 | 75.9% |
| BERT Large | 768 | 76.9% |
| CLIP text | 512 | **79.1%** |
| Qwen3 0.6B | 1024 | 75.8% |

Multimodal (GraphSAGE):

| Text + Image | Test Acc |
|--------------|----------|
| CLIP text + CLIP image | 78.6% |
| BERT Large + DINOv2 | 76.8% |

**Takeaways:** richer text encoders clearly help (CLIP text beats BoW by ~6 points), supporting H3. Adding images on top of text gave no consistent gain with simple concatenation, so H2 is not confirmed in this setting — the textual metadata already carries most of the discriminative signal.

## Running this notebook

```bash
pip install torch torchvision torch_geometric
pip install transformers sentence-transformers
pip install pandas scikit-learn numpy matplotlib
```

Run top to bottom: the graph and BoW features are built first, then `train_sage_full` (the shared training/early-stopping/plotting helper) is reused for every encoder. The multimodal cells expect the teammate's image embeddings in `embeddings_toys_checkpoint.parquet` (columns: `asin`, `emb_resnet`, `emb_clip`, `emb_dinov2`).
