import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.datasets.splits import load_split
from src.encoders.tf_idf import load_tfidf, TFIDF_DIR
from src.encoders.bow import load_bow, BOW_DIR
from src.encoders.sbert import load_sbert, SBERT_DIR
from src.encoders.qwen import load_qwen, QWEN_DIR


def train_logistic_regression(
    X,
    y,
    encoder_name,
    train_idx,
    test_idx,
    max_iter=1000,
    random_state=42,
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    device = "cpu"  # Logistic Regression runs on CPU
    print(f"\n[TRAINING] Starting Logistic Regression on {encoder_name} features using device: {device}")
    print(f"\n[DATA SPLIT] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    print(f"\n[DATA INFO] {len(set(y))} classes.")

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    train_start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - train_start
    print(f"\n[TRAINING] Completed in {train_elapsed:.2f}s")
    y_pred = clf.predict(X_test)

    print("-" * 40)
    print(f"[SCORES] Logistic Regression - Encoder: {encoder_name}")
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-" * 40)

    return clf


if __name__ == "__main__":
    
    # X, y, encoder = load_tfidf()
    # train_idx, test_idx = load_split(TFIDF_DIR)
    # train_logistic_regression(X, y, "TF-IDF", train_idx, test_idx)

    # X, y, encoder = load_bow()
    # train_idx, test_idx = load_split(BOW_DIR)
    # train_logistic_regression(X, y, "BOW", train_idx, test_idx)

    # X, y, encoder = load_sbert()
    # train_idx, test_idx = load_split(SBERT_DIR)
    # train_logistic_regression(X, y, "SBERT", train_idx, test_idx)
    
    X, y, encoder = load_qwen()
    train_idx, test_idx = load_split(QWEN_DIR)
    train_logistic_regression(X, y, 'QWEN', train_idx, test_idx)
