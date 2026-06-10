import time
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from src.datasets.splits import load_split
from src.encoders.tf_idf import load_tfidf, TFIDF_DIR
from src.encoders.bow import load_bow, BOW_DIR
from src.encoders.sbert import load_sbert, SBERT_DIR
from src.encoders.qwen import load_qwen, QWEN_DIR


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.is_sparse = sparse.issparse(X)
        self.X = X.tocsr() if self.is_sparse else np.asarray(X)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            row = self.X[idx].toarray().squeeze(0)
        else:
            row = self.X[idx]
        return torch.from_numpy(row).float(), self.y[idx]


def train_mlp(
    X,
    y,
    encoder_name,
    train_idx,
    test_idx,
    epochs=200,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    val_fraction=0.1,
    patience=10,
    min_delta=1e-4,
    seed=0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[TRAINING] Starting MLP training on {encoder_name} features using device: {device}")

    y = np.asarray(y)

    label_encoder = LabelEncoder()
    label_encoder.fit(y[np.concatenate([train_idx, test_idx])])

    print(f"\n[DATA INFO] {len(label_encoder.classes_)} classes.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_idx_split, train_idx_split = shuffled[:n_val], shuffled[n_val:]

    X_train, X_val, X_test = X[train_idx_split], X[val_idx_split], X[test_idx]
    y_train = label_encoder.transform(y[train_idx_split])
    y_val = label_encoder.transform(y[val_idx_split])
    y_test = label_encoder.transform(y[test_idx])

    print(
        f"\n[DATA SPLIT] Train: {X_train.shape[0]} samples, "
        f"Val: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples"
    )

    train_loader = DataLoader(
        SparseDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SparseDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        SparseDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = MLP(input_dim=X.shape[1], num_classes=len(label_encoder.classes_)).to(device)
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
        running_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device)

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

        print(
            f"[EPOCH {epoch:02d}] train_loss={running_loss / n:.4f} "
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

    y_pred = predict(model, test_loader, device)
    
    print("-" * 40)
    print(f"[SCORES] MLP - Encoder: {encoder_name}")
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-" * 40)

    return model, label_encoder


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    denom = max(n, 1)
    return total_loss / denom, total_correct / denom


@torch.no_grad()
def predict_texts(model, encoder, label_encoder, texts, device=None):
    device = device or next(model.parameters()).device
    X = encoder.transform(texts)
    if sparse.issparse(X):
        X = X.toarray()
    xb = torch.from_numpy(X).float().to(device)
    model.eval()
    idx = model(xb).argmax(dim=1).cpu().numpy()
    return label_encoder.inverse_transform(idx)


if __name__ == "__main__":
    
    target_subcategory_depth = 1

    dir_tfidf = TFIDF_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_tfidf(dir_tfidf)
    train_idx, test_idx = load_split(dir_tfidf)
    train_mlp(X, y, 'TF-IDF', train_idx, test_idx)

    dir_bow = BOW_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_bow(dir_bow)
    train_idx, test_idx = load_split(dir_bow)
    train_mlp(X, y, 'BOW', train_idx, test_idx)

    dir_sbert = SBERT_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_sbert(dir_sbert)
    train_idx, test_idx = load_split(dir_sbert)
    train_mlp(X, y, 'SBERT', train_idx, test_idx)
    
    dir_qwen = QWEN_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_qwen(dir_qwen)
    train_idx, test_idx = load_split(dir_qwen)
    train_mlp(X, y, 'QWEN', train_idx, test_idx)
