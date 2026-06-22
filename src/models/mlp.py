import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

from src.datasets.splits import load_split
from src.encoders.tf_idf import load_tfidf, TFIDF_DIR
from src.encoders.bow import load_bow, BOW_DIR
from src.encoders.sbert import load_sbert, SBERT_DIR
from src.encoders.qwen import load_qwen, QWEN_DIR
from src.encoders.clip_text import load_clip_text, CLIP_TEXT_DIR


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


def train_mlp(
    X,
    y,
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

    print(f"\n[TRAINING] Starting MLP training on {encoder_name} features using device: {device}")

    y = np.asarray(y)

    label_encoder = LabelEncoder()
    label_encoder.fit(y[np.concatenate([train_idx, test_idx])])

    print(f"\n[DATA INFO] {len(label_encoder.classes_)} classes.")

    if sparse.issparse(X) or X.shape[1] > 512:
        svd = TruncatedSVD(n_components=128, random_state=42)
        X = svd.fit_transform(X)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(test_idx)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_idx_split, test_idx_split = shuffled[:n_val], shuffled[n_val:]

    X_train, X_val, X_test = X[train_idx], X[val_idx_split], X[test_idx_split]
    y_train = label_encoder.transform(y[train_idx])
    y_val = label_encoder.transform(y[val_idx_split])
    y_test = label_encoder.transform(y[test_idx_split])

    print(
        f"\n[DATA SPLIT] Train: {X_train.shape[0]} samples, "
        f"Val: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples"
    )

    x_train_t = torch.from_numpy(X_train).float().to(device)
    x_val_t = torch.from_numpy(X_val).float().to(device)
    x_test_t = torch.from_numpy(X_test).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    model = MLP(
        input_dim=X.shape[1], 
        hidden_dim=hidden_dim, 
        num_classes=len(label_encoder.classes_), 
        dropout=dropout
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
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        with torch.no_grad():
            model.eval()
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

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
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} "
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
        y_pred = model(x_test_t).argmax(dim=1).cpu().numpy()
    y_test = y_test_t.cpu().numpy()

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
    
    dir_cliptext = CLIP_TEXT_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_clip_text(dir_cliptext)
    train_idx, test_idx = load_split(dir_cliptext)
    train_mlp(X, y, 'CLIP-TEXT', train_idx, test_idx)
