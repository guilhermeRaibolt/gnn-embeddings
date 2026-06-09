import os

import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_IDX_FILE = "train_idx.npy"
TEST_IDX_FILE = "test_idx.npy"


def make_split(y, test_size=0.2, random_state=42, min_class_count=2):
    y = np.asarray(y)
    indices = np.arange(len(y))

    classes, counts = np.unique(y, return_counts=True)
    rare = classes[counts < min_class_count]
    if len(rare) > 0:
        keep = ~np.isin(y, rare)
        dropped = int((~keep).sum())
        print(
            f"[SPLIT] Dropping {len(rare)} classes with <{min_class_count} samples "
            f"({dropped} rows removed)"
        )
        indices = indices[keep]
        y = y[keep]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_idx, test_idx


def save_split(out_dir, train_idx, test_idx):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, TRAIN_IDX_FILE), train_idx)
    np.save(os.path.join(out_dir, TEST_IDX_FILE), test_idx)
    print(
        f"[SAVE] Split indices: train={len(train_idx)}, test={len(test_idx)} -> {out_dir}"
    )


def load_split(out_dir):
    train_path = os.path.join(out_dir, TRAIN_IDX_FILE)
    test_path = os.path.join(out_dir, TEST_IDX_FILE)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Split indices not found in {out_dir}")
    print(f"\n[LOAD] Loading split indices from {out_dir}")
    return np.load(train_path), np.load(test_path)


def get_or_make_split(y, out_dir, test_size=0.2, random_state=42, min_class_count=2):
    train_path = os.path.join(out_dir, TRAIN_IDX_FILE)
    test_path = os.path.join(out_dir, TEST_IDX_FILE)
    if os.path.exists(train_path) and os.path.exists(test_path):
        return load_split(out_dir)
    train_idx, test_idx = make_split(y, test_size, random_state, min_class_count)
    save_split(out_dir, train_idx, test_idx)
    return train_idx, test_idx
