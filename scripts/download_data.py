import os
import zipfile
import urllib.request
import sys
from pathlib import Path


dataset_url = "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
backup_url = "https://drive.google.com/uc?id=1vTChp3nU5GQeLkPwotrybpUGUXj12BTK&export=download"


def download(url, dest_path):
    print(f"downloading from {url}")
    print(f"destination: {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[{'=' * int(percent // 2)}{' ' * (50 - int(percent // 2))}] "
                        f"{percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print("complete!")


def extract_zip(zip_path, extract_to):
    print(f"extracting to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members):
            zf.extract(member, extract_to)
            if i % 1000 == 0:
                percent = (i + 1) * 100 / total
                sys.stdout.write(f"\rextracting: {percent}% ({i + 1}/{total} files)")
                sys.stdout.flush()
    print("complete!")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "wikiart.zip"

    if (data_dir / "wikiart").exists():
        print("data already exists at")
        return

    if not zip_path.exists():
        try:
            download(dataset_url, zip_path)
        except Exception as e:
            print('failed')
            print(f"Error: {e}")
            return

    extract_zip(zip_path, data_dir)
    zip_path.unlink()
    print("zip file deleted.")

    print("\nDataset ready at data/wikiart")
    print("Structure:")
    for item in sorted((data_dir / "wikiart").iterdir())[:10]:
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
