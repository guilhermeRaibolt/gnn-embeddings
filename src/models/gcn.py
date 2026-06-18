import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.datasets.splits import load_split
from src.encoders.tf_idf import load_tfidf, TFIDF_DIR
from src.encoders.bow import load_bow, BOW_DIR
from src.encoders.sbert import load_sbert, SBERT_DIR
from src.encoders.qwen import load_qwen, QWEN_DIR


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def build_knn_graph(X, k=10):
    nn_index = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn_index.fit(X)
    _, neighbors = nn_index.kneighbors(X)
    n = X.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = neighbors[:, 1:].reshape(-1)  # drop self-match
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)  # undirected
    return torch.from_numpy(edge_index).long()


def train_gcn(
    X,
    y,
    encoder_name,
    train_idx,
    test_idx,
    epochs=10,
    hidden_dim=128,
    k=10,
    lr=1e-3,
    weight_decay=1e-5,
    dropout=0.3,
    device=None,
    val_fraction=0.1,
    seed=0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[TRAINING] Starting GCN training on {encoder_name} features using device: {device}")

    y = np.asarray(y)
    label_encoder = LabelEncoder()
    label_encoder.fit(y[np.concatenate([train_idx, test_idx])])

    print(f"\n[DATA INFO] {len(label_encoder.classes_)} classes.")

    print(f"\n[GRAPH] Building kNN graph (k={k})...")
    graph_start = time.perf_counter()
    edge_index = build_knn_graph(X, k=k)
    print(f"[GRAPH] {edge_index.shape[1]} edges built in {time.perf_counter() - graph_start:.2f}s")

    if sparse.issparse(X):
        X = X.toarray()
    x_t = torch.from_numpy(np.asarray(X)).float()

    n = x_t.shape[0]
    y_full = np.full(n, -1, dtype=np.int64)
    selected = np.concatenate([train_idx, test_idx])
    y_full[selected] = label_encoder.transform(y[selected])
    y_t = torch.from_numpy(y_full).long()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_idx_split, train_idx_split = shuffled[:n_val], shuffled[n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx_split] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx_split] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[test_idx] = True

    print(
        f"\n[DATA SPLIT] Train: {int(train_mask.sum())} samples, "
        f"Val: {int(val_mask.sum())} samples, Test: {int(test_mask.sum())} samples"
    )

    data = Data(x=x_t, edge_index=edge_index, y=y_t).to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = GCN(
        input_dim=x_t.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder.classes_),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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
            val_logits = model(data.x, data.edge_index)
            val_loss = criterion(val_logits[val_mask], data.y[val_mask]).item()
            val_acc = (
                val_logits[val_mask].argmax(dim=1) == data.y[val_mask]
            ).float().mean().item()
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start
        print(
            f"[EPOCH {epoch:02d}] train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_elapsed:.2f}s total={total_elapsed:.2f}s"
        )

    train_elapsed = time.perf_counter() - train_start
    print(f"\n[TRAINING] Completed in {train_elapsed:.2f}s")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        y_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("-" * 40)
    print(f"[SCORES] GCN - Encoder: {encoder_name}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-" * 40)

    return model, label_encoder


if __name__ == "__main__":
    
    target_subcategory_depth = 1

    dir_tfidf = TFIDF_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_tfidf(dir_tfidf)
    train_idx, test_idx = load_split(dir_tfidf)
    train_gcn(X, y, 'TF-IDF', train_idx, test_idx)

    dir_bow = BOW_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_bow(dir_bow)
    train_idx, test_idx = load_split(dir_bow)
    train_gcn(X, y, 'BOW', train_idx, test_idx)

    dir_sbert = SBERT_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_sbert(dir_sbert)
    train_idx, test_idx = load_split(dir_sbert)
    train_gcn(X, y, 'SBERT', train_idx, test_idx)

    dir_qwen = QWEN_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_qwen(dir_qwen)
    train_idx, test_idx = load_split(dir_qwen)
    train_gcn(X, y, 'QWEN', train_idx, test_idx)
