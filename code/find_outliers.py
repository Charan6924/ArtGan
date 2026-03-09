import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from cnn_rnn import build_model
from dataset import WikiArtDataset, get_transforms

CHECKPOINT  = 'checkpoints/phase2_best.pt'
DATA_DIR    = 'data/wikiart/'
SPLIT       = 'val'
TOP_K       = 50
BATCH_SIZE  = 64
NUM_WORKERS = 3
OUTPUT      = 'outliers.csv'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: str, device: torch.device):
    model = build_model({'freeze_backbone': False})
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    embeddings, style_labels, artist_labels, paths = [], [], [], []

    total = len(loader)
    for i, batch in enumerate(loader, 1):
        if i % 10 == 0 or i == total:
            print(f'    batch {i}/{total}', end='\r')

        emb = model.get_embedding(batch['image'].to(device))
        embeddings.append(emb.cpu().float())
        style_labels.append(batch['style'])
        artist_labels.append(batch['artist'])
        paths.extend(batch['path'])

    print()
    return (
        torch.cat(embeddings).numpy(),
        torch.cat(style_labels).numpy(),
        torch.cat(artist_labels).numpy(),
        paths,
    )


def cosine_distance_to_centroid(embeddings, labels):
    """Cosine distance of each sample to its class mean embedding."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    distances = np.zeros(len(embeddings), dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        centroid = normed[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        distances[mask] = 1.0 - (normed[mask] @ centroid)
    return distances


def main():
    device = torch.device(DEVICE)
    print(f'[device]  {device}')

    dataset = WikiArtDataset(DATA_DIR, split=SPLIT, transform=get_transforms('val'))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    print(f'[dataset] {len(dataset):,} images')

    model = load_model(CHECKPOINT, device)
    print('[model]   loaded')

    print('[extract] running forward pass...')
    emb, s_lbl, a_lbl, paths = extract_embeddings(model, loader, device)

    style_dist  = cosine_distance_to_centroid(emb, s_lbl)
    artist_dist = cosine_distance_to_centroid(emb, a_lbl)

    # combined score: average of both centroid distances
    score = (style_dist + artist_dist) / 2

    df = pd.DataFrame({
        'path':          paths,
        'filename':      [Path(p).name for p in paths],
        'true_style':    [dataset.idx_to_style[i]  for i in s_lbl],
        'true_artist':   [dataset.idx_to_artist[i] for i in a_lbl],
        'style_dist':    style_dist.round(4),
        'artist_dist':   artist_dist.round(4),
        'outlier_score': score.round(4),
    })

    df_sorted = df.sort_values('outlier_score', ascending=False).reset_index(drop=True)

    out_path = Path(OUTPUT)
    df_sorted.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()