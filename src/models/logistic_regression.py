from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.encoders.tf_idf import load_tfidf
import numpy as np


def train_logistic_regression(X, y, encoder_name, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
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
    X, y, encoder = load_tfidf()
    train_logistic_regression(X, y, "TF-IDF")
    
    # X, y, encoder = load_bow()
    # train_logistic_regression(X, y, "BOW")
    
    # X, y, encoder = load_sbert()
    # train_logistic_regression(X, y, "SBERT")
