import sys
import os
from dataset import create_dataloaders
import matplotlib.pyplot as plt
import numpy as np

train_loader, val_loader, test_loader, dataset = create_dataloaders( #type: ignore
      root_dir='/mnt/vstor/courses/csds312/cvx166/ArtGan/data/wikiart',
      batch_size=32
  )
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

d = next(iter(train_loader))
img = d['image'][0].permute(1, 2, 0).numpy()
img = img * std + mean
img = np.clip(img, 0, 1)
plt.imshow(img)
plt.title(f"Style: {dataset.idx_to_style[d['style'][0].item()]}, Artist: {dataset.idx_to_artist[d['artist'][0].item()]}")
plt.axis('off')
plt.savefig('sample_image.png')

