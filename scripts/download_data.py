import os
import zipfile
import urllib.request
import sys
from pathlib import Path


DATASET_URL = "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
BACKUP_URL = "https://drive.google.com/uc?id=1vTChp3nU5GQeLkPwotrybpUGUXj12BTK&export=download"


def download_with_progress(url, dest_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    print(f"Destination: {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[{'=' * int(percent // 2)}{' ' * (50 - int(percent // 2))}] "
                        f"{percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print("\nDownload complete!")


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress."""
    print(f"Extracting to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members):
            zf.extract(member, extract_to)
            if i % 1000 == 0:
                percent = (i + 1) * 100 / total
                sys.stdout.write(f"\rExtracting: {percent:.1f}% ({i + 1}/{total} files)")
                sys.stdout.flush()
    print("\nExtraction complete!")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "wikiart.zip"

    if (data_dir / "wikiart").exists():
        print("Dataset already exists at data/wikiart")
        return

    if not zip_path.exists():
        try:
            download_with_progress(DATASET_URL, zip_path)
        except Exception as e:
            print(f"Primary URL failed: {e}")
            print("Try downloading manually from Google Drive:")
            print("https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK")
            return

    extract_zip(zip_path, data_dir)

    delete = input("Delete zip file to save space? [y/N]: ").lower()
    if delete == 'y':
        zip_path.unlink()
        print("Zip file deleted.")

    print("\nDataset ready at data/wikiart")
    print("Structure:")
    for item in sorted((data_dir / "wikiart").iterdir())[:10]:
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
