import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cnn_rnn import build_model
from dataset import WikiArtDataset, get_transforms

CHECKPOINT  = 'checkpoints/phase2_best.pt'
DATA_DIR    = 'data/wikiart/'
SPLIT       = 'val'
TOP_K       = 50
BATCH_SIZE  = 64
NUM_WORKERS = 4
OUTPUT      = 'outliers.csv'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: str, device: torch.device):
    model = build_model({'freeze_backbone': False})
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_all(model, loader, device):
    embeddings, style_logits, artist_logits = [], [], []
    style_labels, artist_labels, paths = [], [], []

    total = len(loader)
    for i, batch in enumerate(loader, 1):
        if i % 10 == 0 or i == total:
            print(f'    batch {i}/{total}', end='\r')

        images = batch['image'].to(device)

        emb = model.get_embedding(images)
        out = model(images, task='all')

        embeddings.append(emb.cpu().float())
        style_logits.append(out['style'].cpu().float())
        artist_logits.append(out['artist'].cpu().float())
        style_labels.append(batch['style'])
        artist_labels.append(batch['artist'])
        paths.extend(batch['path'])

    print()
    return (
        torch.cat(embeddings).numpy(),
        torch.cat(style_logits).numpy(),
        torch.cat(artist_logits).numpy(),
        torch.cat(style_labels).numpy(),
        torch.cat(artist_labels).numpy(),
        paths,
    )


def cosine_distance_to_centroid(embeddings,labels):
    """Cosine distance of each sample to its class mean embedding (0=identical, 2=opposite)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    distances = np.zeros(len(embeddings), dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        centroid = normed[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        distances[mask] = 1.0 - (normed[mask] @ centroid)   # 1 - cosine_sim
    return distances


def gt_confidence(logits,labels):
    """Softmax probability the model assigns to the ground-truth label."""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return probs[np.arange(len(labels)), labels]


def zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def main():
    device = torch.device(DEVICE)
    print(f'[device]  {device}')

    dataset = WikiArtDataset(
        DATA_DIR,
        split=SPLIT,
        transform=get_transforms('val'),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )

    model = load_model(CHECKPOINT, device)
    print('loaded model')

    emb, s_log, a_log, s_lbl, a_lbl, paths = extract_all(model, loader, device)
    style_dist  = cosine_distance_to_centroid(emb, s_lbl)
    artist_dist = cosine_distance_to_centroid(emb, a_lbl)
    style_conf  = gt_confidence(s_log, s_lbl)
    artist_conf = gt_confidence(a_log, a_lbl)
    style_pred   = s_log.argmax(axis=1)
    artist_pred  = a_log.argmax(axis=1)
    style_wrong  = (style_pred  != s_lbl).astype(np.float32)
    artist_wrong = (artist_pred != a_lbl).astype(np.float32)

    score = (zscore(style_dist)+ zscore(artist_dist)- zscore(style_conf)- zscore(artist_conf)+ style_wrong+ artist_wrong)

    df = pd.DataFrame({
        'path':          paths,
        'filename':      [Path(p).name for p in paths],
        'true_style':    [dataset.idx_to_style[i]  for i in s_lbl],
        'true_artist':   [dataset.idx_to_artist[i] for i in a_lbl],
        'pred_style':    [dataset.idx_to_style[i]  for i in style_pred],
        'pred_artist':   [dataset.idx_to_artist[i] for i in artist_pred],
        'style_dist':    style_dist.round(4),
        'artist_dist':   artist_dist.round(4),
        'style_conf':    style_conf.round(4),
        'artist_conf':   artist_conf.round(4),
        'style_wrong':   style_wrong.astype(bool),
        'artist_wrong':  artist_wrong.astype(bool),
        'outlier_score': score.round(4),
    })

    df_sorted = df.sort_values('outlier_score', ascending=False).reset_index(drop=True)
    top = df_sorted.head(TOP_K)

    out_path = Path(OUTPUT)
    df_sorted.to_csv(out_path, index=False)

    top_path = out_path.with_name(out_path.stem + f'_top{TOP_K}.csv')
    top.to_csv(top_path, index=False)

if __name__ == '__main__':
    main()