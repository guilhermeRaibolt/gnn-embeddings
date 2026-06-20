import gzip
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def parse_gz(path):
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                try:
                    yield eval(line)
                except:
                    pass


def build_graph(meta_path, cache_path=None):
    if cache_path and __import__('os').path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded from cache: {cache_path}")
        return cache['data'], cache['asin_order'], cache['label_encoder']

    records = [item for item in parse_gz(meta_path)]
    df = pd.DataFrame(records)

    def get_label(categories):
        if not categories:
            return None
        for path in categories:
            if path and path[0] == 'Toys & Games' and len(path) >= 2:
                return path[1]
        return None

    df['label'] = df['categories'].apply(get_label)
    df = df[df['label'].notna()].reset_index(drop=True)

    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label'])

    asin2idx = {asin: i for i, asin in enumerate(df['asin'])}

    src_list, dst_list = [], []
    for _, row in df.iterrows():
        related = row['related']
        if not isinstance(related, dict):
            continue
        for neighbor_asin in related.get('also_bought', []):
            if neighbor_asin in asin2idx:
                src_list.append(asin2idx[row['asin']])
                dst_list.append(asin2idx[neighbor_asin])

    src_all = src_list + dst_list
    dst_all = dst_list + src_list

    edges = set()
    src_clean, dst_clean = [], []
    for s, d in zip(src_all, dst_all):
        if s == d or (s, d) in edges:
            continue
        edges.add((s, d))
        src_clean.append(s)
        dst_clean.append(d)

    edge_index = torch.tensor([src_clean, dst_clean], dtype=torch.long)
    y = torch.tensor(df['label_idx'].values, dtype=torch.long)

    data = Data(x=torch.zeros(len(df), 1), edge_index=edge_index, y=y)

    idx = torch.arange(len(df))
    idx_train_val, idx_test = train_test_split(
        idx, test_size=0.15, random_state=42, stratify=df['label_idx']
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.15/0.85, random_state=42,
        stratify=df.loc[idx_train_val.numpy(), 'label_idx']
    )
    data.train_mask = torch.zeros(len(df), dtype=torch.bool)
    data.val_mask   = torch.zeros(len(df), dtype=torch.bool)
    data.test_mask  = torch.zeros(len(df), dtype=torch.bool)
    data.train_mask[idx_train] = True
    data.val_mask[idx_val]     = True
    data.test_mask[idx_test]   = True

    asin_order = df['asin'].tolist()

    print(f"Nodes: {len(asin2idx)} | Edges: {edge_index.shape[1]}")
    print(f"Train: {data.train_mask.sum()} | Val: {data.val_mask.sum()} | Test: {data.test_mask.sum()}")
    print(f"Classes: {df['label_idx'].nunique()}")

    if cache_path:
        torch.save({'data': data, 'asin_order': asin_order, 'label_encoder': le}, cache_path)
        print(f"Saved to cache: {cache_path}")

    return data, asin_order, le
