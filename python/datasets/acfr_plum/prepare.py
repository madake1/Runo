"""Convert ACFR Pascal VOC plum boxes to the project's COCO format.

Consecutive frames are kept in chronological blocks, with a small discarded gap
between splits. These are still frames from the same collection, not independent
orchards; the result is for plum localization research only.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets.common import clipped_box, inspect_image, save_coco, write_json


def split_ids(ids):
    """按连续帧分段并丢弃划分边界附近帧。"""
    ids = sorted(ids)
    if len(ids) < 20:
        raise ValueError("至少需要 20 帧才能划分训练、验证和测试")
    first = round(len(ids) * 0.70)
    second = round(len(ids) * 0.85)
    gap = min(5, max(1, len(ids) // 20))
    return {"train": ids[:first], "val": ids[first+gap:second],
            "test": ids[second+gap:]}, gap


def parse_image(folder, image_id):
    """读取一张 ACFR 图片及其 VOC 检测框。"""
    xml = folder / "Annotations" / f"{image_id}.xml"
    tree = ET.parse(xml).getroot()
    image_path = folder / "JPEGImages" / f"{image_id}.jpg"
    width, height, digest = inspect_image(image_path)
    size = tree.find("size")
    if int(size.findtext("width")) != width or int(size.findtext("height")) != height:
        raise ValueError(f"XML 与图片尺寸不一致：{xml}")
    boxes = []
    for obj in tree.findall("object"):
        if obj.findtext("name", "").strip().lower() != "plum":
            raise ValueError(f"未知类别：{xml}")
        box = obj.find("bndbox")
        coords = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append(("plum", clipped_box(coords, width, height)))
    return {"path": f"{folder.name}/JPEGImages/{image_id}.jpg", "width": width,
            "height": height, "digest": digest, "boxes": boxes}


def main():
    """转换 ACFR 标注为带来源隔离的 COCO 划分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/acfr_plum/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/acfr_plum/annotations"))
    args = parser.parse_args()
    records = {"train": [], "val": [], "test": []}
    gaps, sets = {}, []
    for name in ("Day", "Night"):
        folder = args.raw / name
        if not folder.exists():
            continue
        sets.append(name)
        ids = [path.stem for path in (folder / "Annotations").glob("*.xml")]
        groups, gap = split_ids(ids)
        gaps[name] = gap
        for split, selected in groups.items():
            records[split].extend(parse_image(folder, image_id) for image_id in selected)
    if not sets:
        raise FileNotFoundError(f"{args.raw}/Day 或 Night 不存在；先运行 python -m datasets.acfr_plum.download")
    # Guard against identical frames across sessions/splits.
    seen = set()
    for split in ("test", "val", "train"):
        unique = []
        for record in records[split]:
            if record["digest"] not in seen:
                unique.append(record)
                seen.add(record["digest"])
        records[split] = unique
    if any(not records[split] for split in records):
        raise ValueError("至少一个划分为空")
    for split, data in records.items():
        save_coco(data, ["plum"], args.output / f"{split}.json")
    summary = {"source_sets": sets, "images": {split: len(data) for split, data in records.items()},
               "classes": ["plum"], "excluded_gap_frames_per_boundary": gaps,
               "limitation": "Chronological separation reduces adjacent-frame leakage, but all captures may come from the same orchard. No disease annotations."}
    write_json(args.output / "preparation_report.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
