import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, top_k_accuracy_score
)
from collections import defaultdict
from cnn_rnn import ArtClassifier
from dataset import create_dataloaders
import argparse


def evaluate(model, loader, device='cuda'):
    model.eval()

    all_style_preds = []
    all_style_labels = []
    all_artist_preds = []
    all_artist_labels = []
    all_style_probs = []
    all_artist_probs = []

    with torch.no_grad():
        for d in loader:
            images = d['image'].to(device)
            style_labels = d['style']
            artist_labels = d['artist']

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(images)

            style_probs = torch.softmax(outputs['style'].float(), dim=1).cpu().numpy()
            artist_probs = torch.softmax(outputs['artist'].float(), dim=1).cpu().numpy()

            all_style_preds.extend(outputs['style'].argmax(1).cpu().numpy())
            all_artist_preds.extend(outputs['artist'].argmax(1).cpu().numpy())
            all_style_labels.extend(style_labels.numpy())
            all_artist_labels.extend(artist_labels.numpy())
            all_style_probs.append(style_probs)
            all_artist_probs.append(artist_probs)

    return {
        'style_preds': np.array(all_style_preds),
        'style_labels': np.array(all_style_labels),
        'artist_preds': np.array(all_artist_preds),
        'artist_labels': np.array(all_artist_labels),
        'style_probs': np.vstack(all_style_probs),
        'artist_probs': np.vstack(all_artist_probs),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path, figsize=(12, 10)):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved {save_path}')


def compute_metrics(y_true, y_pred, y_probs, class_names, task_name):
    accuracy = (y_pred == y_true).mean()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    top3_acc = top_k_accuracy_score(y_true, y_probs, k=3)
    top5_acc = top_k_accuracy_score(y_true, y_probs, k=5)

    print(f'\n{"="*50}')
    print(f'{task_name.upper()} METRICS')
    print(f'{"="*50}')
    print(f'Accuracy:     {accuracy:.4f}')
    print(f'Top-3 Acc:    {top3_acc:.4f}')
    print(f'Top-5 Acc:    {top5_acc:.4f}')
    print(f'\nMacro Avg:    P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}')
    print(f'Weighted Avg: P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}')

    per_class = list(zip(class_names, precision, recall, f1, support))
    per_class.sort(key=lambda x: x[3], reverse=True)

    print(f'\nTop 10 classes by F1:')
    print(f'{"Class":<25} {"Prec":>8} {"Recall":>8} {"F1":>8} {"Support":>8}')
    print('-' * 60)
    for name, p, r, f, s in per_class[:10]:
        print(f'{name:<25} {p:>8.3f} {r:>8.3f} {f:>8.3f} {int(s):>8}')

    print(f'\nBottom 10 classes by F1:')
    print(f'{"Class":<25} {"Prec":>8} {"Recall":>8} {"F1":>8} {"Support":>8}')
    print('-' * 60)
    for name, p, r, f, s in per_class[-10:]:
        print(f'{name:<25} {p:>8.3f} {r:>8.3f} {f:>8.3f} {int(s):>8}')

    return {
        'accuracy': accuracy,
        'top3_acc': top3_acc,
        'top5_acc': top5_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': per_class
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase2_best.pt')
    parser.add_argument('--data_root', type=str, default='/mnt/vstor/courses/csds312/cvx166/ArtGan/data/wikiart')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size
    )

    loader = test_loader if args.split == 'test' else val_loader

    model = ArtClassifier().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Checkpoint val_loss: {checkpoint.get("val_loss", "N/A")}')

    print(f'\nEvaluating on {args.split} set...')
    results = evaluate(model, loader, device)

    style_names = [dataset.idx_to_style[i] for i in range(dataset.num_styles)]
    artist_names = [dataset.idx_to_artist[i] for i in range(dataset.num_artists)]

    style_metrics = compute_metrics(
        results['style_labels'], results['style_preds'],
        results['style_probs'], style_names, 'Style'
    )

    artist_metrics = compute_metrics(
        results['artist_labels'], results['artist_preds'],
        results['artist_probs'], artist_names, 'Artist'
    )

    plot_confusion_matrix(
        results['style_labels'], results['style_preds'],
        style_names, 'Style Confusion Matrix', 'style_confusion.png'
    )

    top_artists = [x[0] for x in artist_metrics['per_class'][:30]]
    mask = np.isin(artist_names, top_artists)
    top_indices = np.where(mask)[0]

    filtered_mask = np.isin(results['artist_labels'], top_indices)
    if filtered_mask.sum() > 0:
        label_map = {old: new for new, old in enumerate(top_indices)}
        filtered_labels = np.array([label_map.get(l, -1) for l in results['artist_labels'][filtered_mask]])
        filtered_preds = np.array([label_map.get(p, -1) for p in results['artist_preds'][filtered_mask]])

        valid = (filtered_labels >= 0) & (filtered_preds >= 0)
        if valid.sum() > 0:
            plot_confusion_matrix(
                filtered_labels[valid], filtered_preds[valid],
                [artist_names[i] for i in top_indices],
                'Artist Confusion Matrix (Top 30)', 'artist_confusion_top30.png',
                figsize=(14, 12)
            )


if __name__ == '__main__':
    main()
