from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
Image.MAX_IMAGE_PIXELS = None


class WikiArtDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, val_ratio=0.1, test_ratio=0.1, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform

        all_images = list(self.root_dir.glob('*/*.jpg'))

        styles = sorted(set(p.parent.name for p in all_images))
        self.style_to_idx = {s: i for i, s in enumerate(styles)}
        self.idx_to_style = {i: s for s, i in self.style_to_idx.items()}

        artists = sorted(set(p.stem.split('_')[0] for p in all_images))
        self.artist_to_idx = {a: i for i, a in enumerate(artists)}
        self.idx_to_artist = {i: a for a, i in self.artist_to_idx.items()}

        random.seed(seed)
        shuffled = all_images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        test_idx = int(n * test_ratio)
        val_idx = test_idx + int(n * val_ratio)

        if split == 'test':
            self.images = shuffled[:test_idx]
        elif split == 'val':
            self.images = shuffled[test_idx:val_idx]
        else:
            self.images = shuffled[val_idx:]

        self.num_styles = len(styles)
        self.num_artists = len(artists)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path)
            if image.size[0] * image.size[1] > 50_000_000:  # skip >50MP images
                return self.__getitem__((idx + 1) % len(self))
            image = image.convert('RGB')

            style = img_path.parent.name
            artist = img_path.stem.split('_')[0]

            style_idx = self.style_to_idx[style]
            artist_idx = self.artist_to_idx[artist]

            if self.transform:
                image = self.transform(image)

            return {
                'image': image,
                'style': style_idx,
                'artist': artist_idx,
                'path': str(img_path)
            }
        except (OSError, Exception):
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(split='train', image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(root_dir, batch_size=512, num_workers=3, image_size=224):
    train_dataset = WikiArtDataset(
        root_dir,
        split='train',
        transform=get_transforms('train', image_size)
    )

    val_dataset = WikiArtDataset(
        root_dir,
        split='val',
        transform=get_transforms('val', image_size)
    )

    test_dataset = WikiArtDataset(
        root_dir,
        split='test',
        transform=get_transforms('val', image_size)
    )

    for ds in [val_dataset, test_dataset]:
        ds.style_to_idx = train_dataset.style_to_idx
        ds.artist_to_idx = train_dataset.artist_to_idx
        ds.idx_to_style = train_dataset.idx_to_style
        ds.idx_to_artist = train_dataset.idx_to_artist

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset


