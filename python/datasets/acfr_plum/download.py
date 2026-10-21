"""Download and safely extract ACFR's day/night plum detection archives."""
import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

BASE_URL = "https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/dataset"


def extract_archive(archive: Path, target: Path):
    """从官方压缩包提取 RGB 图像、VOC 标注和划分清单。"""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with ZipFile(archive) as zf:
        for item in zf.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Archive path escapes target: {item.filename}")
        for item in zf.infolist():
            # RGB images and VOC XML are sufficient; depth maps are not used.
            if any(part in item.filename for part in ("/JPEGImages/", "/Annotations/", "/ImageSets/")):
                zf.extract(item, root)


def main():
    """下载并整理 ACFR 日间或夜间原始数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", choices=("Day", "Night"), default=["Day"],
                        help="默认只下载日间 78 MB；可选 Day Night")
    parser.add_argument("--root", type=Path, default=Path("data/acfr_plum"))
    args = parser.parse_args()
    archives = args.root / "archives"
    raw = args.root / "raw"
    archives.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        archive = archives / f"{name}.zip"
        if not archive.is_file():
            temporary = archive.with_suffix(".download")
            with urlopen(f"{BASE_URL}/{name}.zip", timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(archive)
        extract_archive(archive, raw)
        print(f"Extracted {name}: {archive} -> {raw}", flush=True)


if __name__ == "__main__":
    main()
