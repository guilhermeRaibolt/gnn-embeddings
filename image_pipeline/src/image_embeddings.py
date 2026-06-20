import os
import gzip
import json
import ast
import argparse
import urllib.request
import urllib.error
import requests
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoImageProcessor, AutoModel,
)

METADATA_URLS = {
    "meta_musical_instruments": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Musical_Instruments.json.gz",
    "meta_electronics":         "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz",
    "meta_books":               "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz",
    "meta_movies_tv":           "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz",
    "meta_cds_vinyl":           "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz",
    "meta_clothing_shoes_jewelry": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz",
    "meta_home_kitchen":        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz",
    "meta_sports_outdoors":     "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
    "meta_toys_games":          "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
    "meta_video_games":         "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz",
    "meta_beauty":              "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    "meta_health_personal_care":"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Health_and_Personal_Care.json.gz",
    "meta_pet_supplies":        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Pet_Supplies.json.gz",
    "meta_automotive":          "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Automotive.json.gz",
    "meta_grocery_gourmet_food":"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Grocery_and_Gourmet_Food.json.gz",
    "meta_baby":                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz",
    "meta_office_products":     "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz",
    "meta_tools_home_improvement": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Tools_and_Home_Improvement.json.gz",
    "meta_patio_lawn_garden":   "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Patio_Lawn_and_Garden.json.gz",
}


def ensure_metadata(local_path: str):
    if os.path.exists(local_path):
        print(f"[meta] Already exists: {local_path}")
        return
    stem = os.path.splitext(os.path.splitext(os.path.basename(local_path))[0])[0]
    url = METADATA_URLS.get(stem)
    if url is None:
        raise ValueError(f"No URL known for '{stem}'. Add it to METADATA_URLS or download manually.")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"[meta] Downloading {url} ...")
    urllib.request.urlretrieve(url, local_path)
    print(f"[meta] Saved to {local_path}")


def parse_json_gz(path: str):
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line.decode("utf-8"))
                except Exception:
                    pass


def get_label(categories, top_category: str):
    if not categories:
        return None
    for path in categories:
        if path and path[0] == top_category and len(path) >= 2:
            return path[1]
    return None


def load_metadata(path: str, category: str, max_items: int = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: asin, imUrl, label.
    Only rows with all three fields present are kept.
    """
    ensure_metadata(path)
    data = []
    for i, entry in enumerate(parse_json_gz(path)):
        if max_items is not None and i >= max_items:
            break
        asin    = entry.get("asin", "")
        img_url = entry.get("imUrl", "")
        label   = get_label(entry.get("categories", []), category)
        if asin and img_url and label is not None:
            data.append({"asin": asin, "imUrl": img_url, "label": label})

    df = pd.DataFrame(data)
    print(f"[meta] Loaded {len(df)} products with imUrl + label from {path}")
    return df


def download_image(asin: str, url: str, img_dir: str) -> str | None:
    path = os.path.join(img_dir, f"{asin}.jpg")
    if os.path.exists(path):
        return path
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
    except Exception:
        pass
    return None


def load_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_models(device: torch.device):
    print("[models] Loading ResNet50 ...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)

    resnet_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]) # Standard values from the model's source code

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval().to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
    dino_model.eval().to(device)
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    print("[models] All models loaded.\n")
    return (resnet, resnet_tf), (clip_model, clip_proc), (dino_model, dino_proc)


@torch.no_grad()
def embed_resnet(images, resnet, resnet_tf, device):
    tensors = torch.stack([resnet_tf(img) for img in images]).to(device)
    return resnet(tensors).cpu().numpy()


@torch.no_grad()
def embed_clip(images, clip_model, clip_proc, device):
    inputs = clip_proc(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy()


@torch.no_grad()
def embed_dinov2(images, dino_model, dino_proc, device):
    inputs = dino_proc(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = dino_model(**inputs)
    return outputs.last_hidden_state[:, 0].cpu().numpy()


def build_embeddings(df_meta, out_path, img_dir, models_tuple, device, batch_size, checkpoint_every):
    (resnet, resnet_tf), (clip_model, clip_proc), (dino_model, dino_proc) = models_tuple

    # Checkpoint logic
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        df_ckpt    = pd.read_parquet(out_path)
        done_asins = set(df_ckpt["asin"].tolist())
        records    = df_ckpt.to_dict("records")
        print(f"[ckpt] Resuming — {len(done_asins)} ASINs already embedded.")
    else:
        done_asins = set()
        records    = []

    df_todo = df_meta[~df_meta["asin"].isin(done_asins)].reset_index(drop=True)
    print(f"[ckpt] {len(df_todo)} ASINs remaining.\n")

    if df_todo.empty:
        print("[done] Nothing to do. Returning existing checkpoint.")
        return pd.read_parquet(out_path)

    os.makedirs(img_dir, exist_ok=True)
    batches_since_save = 0
    skipped = 0

    for i in tqdm(range(0, len(df_todo), batch_size), desc="Batches"):
        batch = df_todo.iloc[i : i + batch_size]
        images, asins, labels = [], [], []

        for _, row in batch.iterrows():
            path = download_image(row["asin"], row["imUrl"], img_dir)
            if path is None:
                skipped += 1
                continue
            img = load_image(path)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            asins.append(row["asin"])
            labels.append(row["label"])

        if not images:
            continue

        resnet_embs = embed_resnet(images, resnet, resnet_tf, device)
        clip_embs   = embed_clip(images, clip_model, clip_proc, device)
        dino_embs   = embed_dinov2(images, dino_model, dino_proc, device)

        for j, (asin, label) in enumerate(zip(asins, labels)):
            records.append({
                "asin":       asin,
                "label":      label,
                "emb_resnet": resnet_embs[j],
                "emb_clip":   clip_embs[j],
                "emb_dinov2": dino_embs[j],
            })

        batches_since_save += 1
        if batches_since_save >= checkpoint_every:
            pd.DataFrame(records).to_parquet(out_path, index=False)
            batches_since_save = 0
            tqdm.write(f"[ckpt] Saved {len(records)} records → {out_path}")

    # final save
    df_out = pd.DataFrame(records)
    df_out.to_parquet(out_path, index=False)
    print(f"\n[done] Final checkpoint saved → {out_path}")
    print(f"[done] Total embedded: {len(df_out)} | Skipped (no image): {skipped}")
    return df_out



def generate_embeddings(
    meta,
    out,
    imgdir,
    category="Toys & Games",
    batch=64,
    checkpoint_every=10,
    max_items=None,
    device=None,
):

    if device:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}\n")

    df_meta = load_metadata(meta, category, max_items)
    if df_meta.empty:
        print("No products found.")
        return

    models = build_models(device)

    build_embeddings(
        df_meta=df_meta,
        out_path=out,
        img_dir=imgdir,
        models_tuple=models_tuple,
        device=device,
        batch_size=batch,
        checkpoint_every=checkpoint_every,
    )
