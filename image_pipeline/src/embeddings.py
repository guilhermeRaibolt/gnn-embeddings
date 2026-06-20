import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _lookup_matrix(asin_order, img_lookup, col):
    sample = img_lookup[col].iloc[0]
    D = len(sample)
    X = np.zeros((len(asin_order), D), dtype=np.float32)
    for i, asin in enumerate(asin_order):
        if asin in img_lookup.index:
            X[i] = img_lookup.loc[asin, col]
    return X


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)


def load_image_embeddings(parquet_path, asin_order, normalize=True):
    images = pd.read_parquet(parquet_path)
    img_lookup = images.set_index('asin')

    X_resnet = _lookup_matrix(asin_order, img_lookup, 'emb_resnet')
    X_clip   = _lookup_matrix(asin_order, img_lookup, 'emb_clip')
    X_dino   = _lookup_matrix(asin_order, img_lookup, 'emb_dinov2')

    n_covered = sum(1 for a in asin_order if a in img_lookup.index)
    print(f"Image coverage: {n_covered}/{len(asin_order)} ({n_covered/len(asin_order):.1%})")
    print(f"ResNet: {X_resnet.shape} | CLIP: {X_clip.shape} | DINOv2: {X_dino.shape}")

    if normalize:
        X_resnet = l2_normalize(X_resnet)
        X_clip   = l2_normalize(X_clip)
        X_dino   = l2_normalize(X_dino)

    return {'resnet': X_resnet, 'clip': X_clip, 'dino': X_dino}


def load_text_embeddings(df, n_components=128):
    df = df.copy()
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['text'] = df['text'].str.strip()

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['text'])

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_bow = svd.fit_transform(X_tfidf).astype(np.float32)

    print(f"TF-IDF+SVD: {X_bow.shape} | Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return l2_normalize(X_bow)


def combine(*arrays):
    result = np.concatenate(arrays, axis=1)
    print(f"Combined embedding shape: {result.shape}")
    return result
