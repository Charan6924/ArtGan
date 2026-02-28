import torch

device = 'cuda'

def validate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    style_correct = 0
    artist_correct = 0
    total_samples = 0

    with torch.no_grad():
        for d in val_loader:
            images = d['image'].to(device)
            style_labels = d['style'].to(device)
            artist_labels = d['artist'].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(images)
                style_loss = loss_fn(outputs['style'], style_labels)
                artist_loss = loss_fn(outputs['artist'], artist_labels)
                combined_loss = style_loss + artist_loss

            total_loss += combined_loss.item()
            style_correct += (outputs['style'].argmax(1) == style_labels).sum().item()
            artist_correct += (outputs['artist'].argmax(1) == artist_labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / len(val_loader)
    style_acc = style_correct / total_samples
    artist_acc = artist_correct / total_samples
    return avg_loss, style_acc, artist_acc

    