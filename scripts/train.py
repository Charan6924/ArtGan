import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from dataset import create_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from cnn_rnn import ArtClassifier
import time
import tqdm

train_loader, val_loader, test_loader, dataset = create_dataloaders( #type: ignore
      root_dir='/mnt/vstor/courses/csds312/cvx166/ArtGan/data/wikiart',
      batch_size=1024
  )

device = 'cuda'
model = ArtClassifier().to(device)
model = torch.compile(model)
print('created model')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()
num_epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)                                                                                                

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, d in enumerate(pbar):
        t1 = time.time()
        images = d['image'].to(device)
        style_labels = d['style'].to(device)
        artist_labels = d['artist'].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(images)
            style_loss = loss(outputs['style'], style_labels)
            artist_loss = loss(outputs['artist'], artist_labels)
            combined_loss = style_loss + artist_loss
        combined_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += combined_loss.item()
        t2 = time.time()
        pbar.set_postfix({
            'loss': f'{combined_loss.item():.4f}',
            'time': f'{t2-t1:.3f}s'
        })

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')