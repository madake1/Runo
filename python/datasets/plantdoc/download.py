"""下载官方 PlantDoc 检测数据。固定版本、失败重试、可继续下载。"""
import argparse
import csv
import hashlib
import random
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets.common import local_filename, write_json

REPO = "pratikkayal/PlantDoc-Object-Detection-Dataset"
REVISION = "4730a233a555b30ee98e0879c63ad25d82407455"
BASE = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"


def fetch(remote, target):
    """按固定上游版本下载文件并在失败时重试。"""
    target = Path(target)
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            request = urllib.request.Request(BASE + urllib.parse.quote(remote, safe="/"),
                                             headers={"User-Agent": "Runo-learning"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data:
                raise ValueError("下载内容为空")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
            return
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 404:
                raise
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    """下载 PlantDoc 标注和图片并记录校验信息。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/plantdoc/raw"))
    parser.add_argument("--max-images", type=int, default=0, help="每个官方划分上限；0 下载全量")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_images < 0 or args.workers < 1:
        parser.error("max-images >= 0 且 workers >= 1")
    root = args.output
    for name in ["train_labels.csv", "test_labels.csv", "LICENSE.txt"]:
        fetch(name, root / name)
    files = []
    for split, csv_name in [("TRAIN", "train_labels.csv"), ("TEST", "test_labels.csv")]:
        rows = list(csv.DictReader((root / csv_name).open(encoding="utf-8-sig")))
        names = sorted({row["filename"] for row in rows})
        if args.max_images:
            random.Random(42).shuffle(names)
            names = sorted(names[:args.max_images])
        files.extend((f"{split}/{name}", root / split / local_filename(name)) for name in names)
    errors, unavailable = [], []
    def job(item):
        """下载一张图片并返回校验值或错误。"""
        remote, target = item
        try:
            fetch(remote, target)
            return remote, hashlib.sha256(target.read_bytes()).hexdigest(), None, False
        except Exception as error:
            return remote, None, str(error), isinstance(error, urllib.error.HTTPError) and error.code == 404
    checksums = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (remote, digest, error, missing) in enumerate(pool.map(job, files), 1):
            if error:
                (unavailable if missing else errors).append({"path": remote, "error": error})
            else:
                checksums[remote] = digest
            if index % 50 == 0 or index == len(files):
                print(f"Downloaded {index}/{len(files)}, failures={len(errors)}", flush=True)
    write_json(root / "download_report.json", {"repository": REPO, "revision": REVISION,
               "max_images_per_split": args.max_images, "sha256": checksums,
               "upstream_unavailable": unavailable, "errors": errors})
    if errors:
        raise SystemExit("部分文件下载失败，重新执行可继续；查看 download_report.json")


if __name__ == "__main__":
    main()
