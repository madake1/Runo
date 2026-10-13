"""将 LabelMe 矩形标注转为 COCO；按树/拍摄批次划分，支持健康负样本。"""
import argparse
import json
from pathlib import Path

from datasets.common import clipped_box, group_split, inspect_image, save_coco, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/fengtangli/raw"))
    parser.add_argument("--groups", type=Path, required=True,
                        help="JSON: 相对图片路径 -> 果园/树/连拍或视频来源组 ID")
    parser.add_argument("--classes", type=Path, required=True, help="JSON 字符串列表，专家确认后制定")
    parser.add_argument("--output", type=Path, default=Path("data/fengtangli/annotations"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    classes = json.loads(args.classes.read_text(encoding="utf-8"))
    groups = json.loads(args.groups.read_text(encoding="utf-8"))
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("类别列表不能为空或包含重复类别")
    records = []
    for label_path in sorted(args.raw.rglob("*.json")):
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        if "shapes" not in annotation:
            continue
        path = (label_path.parent / annotation["imagePath"]).resolve()
        if not path.is_relative_to(args.raw.resolve()):
            raise ValueError(f"图片必须位于 raw 下：{path}")
        relative = path.relative_to(args.raw.resolve()).as_posix()
        if relative not in groups:
            raise ValueError(f"缺少来源组：{relative}")
        width, height, digest = inspect_image(path)
        boxes = []
        for shape in annotation["shapes"]:
            if shape["shape_type"] != "rectangle":
                raise ValueError(f"仅支持矩形框，不能静默丢弃其他形状：{label_path}")
            name = shape["label"]
            if name not in classes:
                raise ValueError(f"未定义类别：{name}")
            (x1, y1), (x2, y2) = shape["points"]
            box = clipped_box([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], width, height)
            boxes.append((name, box))
        records.append({"path": relative, "width": width, "height": height, "digest": digest,
                        "group": str(groups[relative]), "boxes": boxes})
    if len(records) < 3:
        raise ValueError("至少需要三个独立来源组的已标注图片")
    train_val, test = group_split(records, val_ratio=0.15, seed=args.seed)
    train, val = group_split(train_val, val_ratio=0.15/0.85, seed=args.seed+1)
    for split, subset in [("train", train), ("val", val), ("test", test)]:
        present = {name for record in subset for name, _ in record["boxes"]}
        missing = sorted(set(classes) - present)
        if missing:
            raise ValueError(f"{split} 缺少类别 {missing}；增加独立来源组数据或调整划分种子")
        save_coco(subset, classes, args.output / f"{split}.json")
    write_json(args.output / "preparation_report.json", {
        "classes": classes, "seed": args.seed,
        "images": {"train": len(train), "val": len(val), "test": len(test)},
        "groups": {split: sorted({r['group'] for r in subset})
                   for split, subset in [("train", train), ("val", val), ("test", test)]}})
    print("Prepared:", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
