import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected, coalesce

from src.datasets.splits import load_split
from src.datasets.amazon_dataset import load_related_data
from src.encoders.tf_idf import load_tfidf, TFIDF_DIR
from src.encoders.bow import load_bow, BOW_DIR
from src.encoders.sbert import load_sbert, SBERT_DIR
from src.encoders.qwen import load_qwen, QWEN_DIR


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def build_graph_from_amazon(related_df):
    asin2idx = {asin: i for i, asin in enumerate(related_df['asin'].to_numpy())}
    n = len(asin2idx)
    # edge_types = ['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing']
    edge_types = ['also_bought']
    src_chunks, dst_chunks = [], []
    for src_asin, rel in zip(related_df['asin'].to_numpy(), related_df['related'].to_numpy()):
        if not isinstance(rel, dict):
            continue
        src_idx = asin2idx[src_asin]
        for etype in edge_types:
            neighbors = rel.get(etype)
            if not neighbors:
                continue
            dst_ids = [asin2idx[a] for a in neighbors if a in asin2idx]
            if not dst_ids:
                continue
            dst_arr = np.asarray(dst_ids, dtype=np.int64)
            src_chunks.append(np.full(dst_arr.shape, src_idx, dtype=np.int64))
            dst_chunks.append(dst_arr)

    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)

    keep = src != dst
    src, dst = src[keep], dst[keep]

    edge_index = torch.from_numpy(np.stack([src, dst]))
    edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index = coalesce(edge_index, num_nodes=n)

    return edge_index

def train_sage(
    X,
    y,
    related,
    encoder_name,
    train_idx,
    test_idx,
    epochs=200,
    hidden_dim=256,
    lr=1e-2,
    weight_decay=5e-4,
    dropout=0.5,
    device=None,
    val_fraction=0.5, # 50% of the test set is for validation
    patience=10,
    min_delta=1e-4,
    seed=0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[TRAINING] Starting GraphSAGE training on {encoder_name} features using device: {device}")

    y = np.asarray(y)
    label_encoder = LabelEncoder()
    label_encoder.fit(y[np.concatenate([train_idx, test_idx])])

    print(f"\n[DATA INFO] {len(label_encoder.classes_)} classes.")

    if len(related) != X.shape[0]:
        raise ValueError(
            f"Related data rows ({len(related)}) do not align with feature rows ({X.shape[0]}); "
            "ensure the related cache was built with the same depth as the encoded features."
        )

    print(f"\n[GRAPH] Building Amazon co-purchase graph...")
    graph_start = time.perf_counter()
    edge_index = build_graph_from_amazon(related)
    print(f"[GRAPH] {X.shape[0]} nodes and {edge_index.shape[1]} edges built in {time.perf_counter() - graph_start:.2f}s")

    if sparse.issparse(X) or X.shape[1] > 512:
        svd = TruncatedSVD(n_components=128, random_state=42)
        X = svd.fit_transform(X)
    
    x_t = torch.from_numpy(np.asarray(X)).float()

    n = x_t.shape[0]
    y_full = np.full(n, -1, dtype=np.int64)
    selected = np.concatenate([train_idx, test_idx])
    y_full[selected] = label_encoder.transform(y[selected])
    y_t = torch.from_numpy(y_full).long()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(test_idx)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_idx_split, test_idx_split = shuffled[:n_val], shuffled[n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx_split] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[test_idx_split] = True

    print(
        f"\n[DATA SPLIT] Train: {int(train_mask.sum())} samples, "
        f"Val: {int(val_mask.sum())} samples, Test: {int(test_mask.sum())} samples"
    )

    data = Data(x=x_t, edge_index=edge_index, y=y_t)
    
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = GraphSAGE(
        input_dim=x_t.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder.classes_),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            val_logits = logits.detach()
            val_loss = criterion(val_logits[val_mask], data.y[val_mask]).item()
            val_acc = (
                val_logits[val_mask].argmax(dim=1) == data.y[val_mask]
            ).float().mean().item()
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # if epoch % 10 == 0:
        print(
            f"[EPOCH {epoch:02d}] train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_elapsed:.2f}s total={total_elapsed:.2f}s"
            + (" *" if improved else "")
        )

        if epochs_without_improvement >= patience:
            print(
                f"\n[EARLY STOP] No val improvement for {patience} epochs. "
                f"Best epoch {best_epoch} (val_loss={best_val_loss:.4f})."
            )
            break

    model.load_state_dict(best_state)
    train_elapsed = time.perf_counter() - train_start
    print(
        f"\n[TRAINING] Completed in {train_elapsed:.2f}s "
        f"(restored weights from epoch {best_epoch}, val_loss={best_val_loss:.4f})"
    )

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        y_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("-" * 40)
    print(f"[SCORES] GraphSAGE - Encoder: {encoder_name}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-" * 40)

    return model, label_encoder


if __name__ == "__main__":

    target_subcategory_depth = 1

    related = load_related_data(target_subcategory_depth)

    dir_tfidf = TFIDF_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_tfidf(dir_tfidf)
    train_idx, test_idx = load_split(dir_tfidf)
    train_sage(X, y, related, 'TF-IDF', train_idx, test_idx)

    dir_bow = BOW_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_bow(dir_bow)
    train_idx, test_idx = load_split(dir_bow)
    train_sage(X, y, related, 'BOW', train_idx, test_idx)

    dir_sbert = SBERT_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_sbert(dir_sbert)
    train_idx, test_idx = load_split(dir_sbert)
    train_sage(X, y, related, 'SBERT', train_idx, test_idx)

    dir_qwen = QWEN_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_qwen(dir_qwen)
    train_idx, test_idx = load_split(dir_qwen)
    train_sage(X, y, related, 'QWEN', train_idx, test_idx)