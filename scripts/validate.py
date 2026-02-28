import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
device = 'cuda'

def validate(model,val_loader,loss):
    model.eval()
    for d in val_loader:
        images = d['image'].to(device)
        style_labels = d['style'].to(device)
        artist_labels = d['artist'].to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(images)
            style_loss = loss(outputs['style'], style_labels)
            artist_loss = loss(outputs['artist'], artist_labels)
            combined_loss = style_loss + artist_loss

        total_loss += combined_loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

    