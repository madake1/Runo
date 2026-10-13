"""将官方 CSV 转成 COCO，保留官方测试集，训练集再分出验证集。"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from datasets.common import clipped_box, group_split, inspect_image, local_filename, save_coco, write_json


def read_split(root, folder, csv_name, issues):
    by_file = defaultdict(list)
    with (root / csv_name).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            by_file[row["filename"]].append(row)
    records = []
    for filename, rows in sorted(by_file.items()):
        path = root / folder / local_filename(filename)
        if not path.exists():
            issues.append({"path": str(path), "reason": "not_downloaded"})
            continue
        try:
            width, height, digest = inspect_image(path)
            boxes = []
            for row in rows:
                # CSV 尺寸可能与真实文件不同，将坐标映射回实际原图。
                source_width, source_height = float(row["width"]), float(row["height"])
                if source_width <= 0 or source_height <= 0:
                    raise ValueError("官方 CSV 图片尺寸为零或负数，无法可靠换算框坐标")
                sx, sy = width / source_width, height / source_height
                box = [float(row["xmin"])*sx, float(row["ymin"])*sy,
                       float(row["xmax"])*sx, float(row["ymax"])*sy]
                boxes.append((row["class"].strip(), clipped_box(box, width, height)))
            records.append({"path": f"{folder}/{local_filename(filename)}", "width": width, "height": height,
                            "digest": digest, "group": digest, "boxes": boxes})
        except Exception as error:
            issues.append({"path": str(path), "reason": str(error)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/plantdoc/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/plantdoc/annotations"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    issues = []
    train = read_split(args.raw, "TRAIN", "train_labels.csv", issues)
    test = read_split(args.raw, "TEST", "test_labels.csv", issues)
    if not train or not test:
        raise ValueError("需要已下载的官方训练和测试图片")
    # 所有 CSV 的类别均纳入映射，避免子集和全量使用不同类别编号。
    classes = set()
    for filename in ["train_labels.csv", "test_labels.csv"]:
        with (args.raw / filename).open(encoding="utf-8-sig") as stream:
            classes.update(row["class"].strip() for row in csv.DictReader(stream))
    classes = sorted(classes)
    test_hashes = {record["digest"] for record in test}
    leaked = [record for record in train if record["digest"] in test_hashes]
    train = [record for record in train if record["digest"] not in test_hashes]
    train, val = group_split(train, seed=args.seed)
    for split, records in [("train", train), ("val", val), ("test", test)]:
        save_coco(records, classes, args.output / f"{split}.json")
    summary = {"classes": classes, "images": {"train": len(train), "val": len(val), "test": len(test)},
               "removed_train_test_duplicates": len(leaked), "issues": issues,
               "limitation": "No tree/session identity available; pixel deduplication cannot remove all near-duplicates."}
    write_json(args.output / "preparation_report.json", summary)
    print(summary["images"], "classes:", len(classes), "issues:", len(issues))


if __name__ == "__main__":
    main()
