import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from src.encoders.tf_idf import get_tfidf_features
from src.encoders.bow import get_bow_features


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
        self.X = X.tocsr()
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze(0)
        return torch.from_numpy(row).float(), self.y[idx]


def train_mlp(
    X,
    y,
    encoder_name,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    test_size=0.2,
    random_state=42,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[TRAINING] Starting MLP training on {encoder_name} features using device: {device}")

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    
    print(f"\n[DATA SPLIT] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    train_loader = DataLoader(
        SparseDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SparseDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = MLP(input_dim=X.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
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
        print(f"[EPOCH {epoch:02d}] train_loss={running_loss / n:.4f}")

    y_pred = predict(model, test_loader, device)
    print(f"\n[SCORES] {encoder_name} MLP Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

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
    X, y, encoder = get_tfidf_features()
    model, label_encoder = train_mlp(X, y, 'TF-IDF')
    
    # X, y, encoder = get_bow_features()
    # model, label_encoder = train_mlp(X, y, 'BOW')

    # print("\n[TEST] Predicting categories for new inputs:")
    # mock_data = [
    #     "Acoustic Guitar",
    #     "Truck 4x4",
    #     "Car Radio Sound System",
    # ]
    # preds = predict_texts(model, encoder, label_encoder, mock_data)
    # for text, pred in zip(mock_data, preds):
    #     print(f"Input: {text}\nPredicted Category: {pred}\n")
