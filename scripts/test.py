# run this as a separate script to find huge images
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

root = '/mnt/vstor/courses/csds312/cvx166/ArtGan/data/wikiart'
for path in Path(root).rglob('*'):
    if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            img = Image.open(path)
            pixels = img.width * img.height
            if pixels > 50_000_000:
                print(f"{pixels/1e6:.1f}MP - {path}")
        except Exception as e:
            print(f"ERROR: {path} - {e}")