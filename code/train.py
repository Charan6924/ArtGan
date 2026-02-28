import torch
import torch.nn as nn
from dataset import create_dataloaders
from cnn_rnn import ArtClassifier
import time
import tqdm
from validate import validate
import csv
import os

train_loader, val_loader, test_loader, dataset = create_dataloaders(
    root_dir='/mnt/vstor/courses/csds312/cvx166/ArtGan/data/wikiart',
    batch_size=1024
)

device = 'cuda'
model = ArtClassifier().to(device)
compiled_model = torch.compile(model)
print('created model')

os.makedirs('checkpoints', exist_ok=True)

def train_phase(model, compiled_model, train_loader, val_loader, optimizer, scheduler,
               loss_fn, num_epochs, patience, phase_name, log_file):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        compiled_model.train()
        total_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f'{phase_name} Epoch {epoch+1}/{num_epochs}')

        for i, d in enumerate(pbar):
            t1 = time.time()
            images = d['image'].to(device)
            style_labels = d['style'].to(device)
            artist_labels = d['artist'].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = compiled_model(images)
                style_loss = loss_fn(outputs['style'], style_labels)
                artist_loss = loss_fn(outputs['artist'], artist_labels)
                combined_loss = style_loss + artist_loss
            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.item()
            t2 = time.time()
            pbar.set_postfix({
                'loss': f'{combined_loss.item():.4f}',
                'time': f'{t2-t1:.3f}s'
            })

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        val_loss, style_acc, artist_acc = validate(compiled_model, val_loader, loss_fn)

        print(f'{phase_name} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Style Acc: {style_acc:.4f} | Artist Acc: {artist_acc:.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoints/{phase_name}_epoch_{epoch+1}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'style_acc': style_acc,
                'artist_acc': artist_acc,
            }, f'checkpoints/{phase_name}_best.pt')
        else:
            patience_counter += 1

        lr = optimizer.param_groups[-1]['lr']
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([phase_name, epoch + 1, train_loss, val_loss, style_acc, artist_acc, lr])

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    return best_val_loss

with open('logs.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['phase', 'epoch', 'train_loss', 'val_loss', 'style_acc', 'artist_acc', 'lr'])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs_phase1 = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_phase1)

train_phase(model, compiled_model, train_loader, val_loader,optimizer, scheduler, loss_fn,num_epochs=num_epochs_phase1,patience=12,phase_name='phase1',log_file='logs.csv')

checkpoint = torch.load('checkpoints/phase1_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'loaded best phase1 model (val_loss: {checkpoint["val_loss"]:.4f})')

model.unfreeze_backbone(layers=-3)
print('unfroze backbone')

backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

num_epochs_phase2 = 50
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_phase2)

compiled_model = torch.compile(model)

train_phase(model, compiled_model, train_loader, val_loader,optimizer, scheduler, loss_fn,num_epochs=num_epochs_phase2,patience=10,phase_name='phase2',log_file='logs.csv')
