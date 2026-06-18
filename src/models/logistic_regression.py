import time
import numpy as np

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
    max_iter=500,
    random_state=42,
    seed=0,
):
    
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(test_idx)
    n_val = max(1, int(len(shuffled) * 0.5))
    _val_idx_split, test_idx_split = shuffled[:n_val], shuffled[n_val:]

    X_train, X_test = X[train_idx], X[test_idx_split]
    y_train, y_test = y[train_idx], y[test_idx_split]

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
    
    target_subcategory_depth = 1

    dir_tfidf = TFIDF_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_tfidf(dir_tfidf)
    train_idx, test_idx = load_split(dir_tfidf)
    train_logistic_regression(X, y, 'TF-IDF', train_idx, test_idx)

    dir_bow = BOW_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_bow(dir_bow)
    train_idx, test_idx = load_split(dir_bow)
    train_logistic_regression(X, y, 'BOW', train_idx, test_idx)

    dir_sbert = SBERT_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_sbert(dir_sbert)
    train_idx, test_idx = load_split(dir_sbert)
    train_logistic_regression(X, y, 'SBERT', train_idx, test_idx)
    
    dir_qwen = QWEN_DIR+"_depth"+str(target_subcategory_depth)
    X, y, encoder = load_qwen(dir_qwen)
    train_idx, test_idx = load_split(dir_qwen)
    train_logistic_regression(X, y, 'QWEN', train_idx, test_idx)
