"""数据准备公用逻辑：验证图片、按来源组划分、输出 COCO。"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def local_filename(name):
    """官方部分文件名含 ? 等 Windows 禁用字符；用哈希保留可重复映射。"""
    if any(char in name for char in '<>:"/\\|?*') or name.endswith((".", " ")):
        return hashlib.sha256(name.encode("utf-8")).hexdigest() + ".jpg"
    return name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect_image(path):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        digest = hashlib.sha256(image.convert("RGB").tobytes() + str(image.size).encode()).hexdigest()
    return width, height, digest


def group_split(records, val_ratio=0.2, seed=42):
    """同一个 group（同树/连拍/视频）和像素重复图片不可跨划分。"""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    # 并查集将来源组和重复图片关联成不可拆开的连通组。
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]

    for record in records:
        a, b = root("g:" + record["group"]), root("h:" + record["digest"])
        parent[a] = b
    groups = defaultdict(list)
    for record in records:
        groups[root("g:" + record["group"])].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("至少需要两个独立来源组才能划分训练和验证集")
    n = max(1, min(len(keys) - 1, round(len(keys) * val_ratio)))
    val_keys = set(keys[:n])
    train, val = [], []
    for key in keys:
        (val if key in val_keys else train).extend(groups[key])
    return train, val


def save_coco(records, classes, output):
    """内部 boxes 使用原图像素 xyxy；COCO 写出 xywh，类别从 1 开始。"""
    class_ids = {name: i + 1 for i, name in enumerate(classes)}
    images, annotations = [], []
    for image_id, record in enumerate(records, 1):
        images.append({"id": image_id, "file_name": record["path"],
                       "width": record["width"], "height": record["height"]})
        for name, box in record["boxes"]:
            x1, y1, x2, y2 = box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": class_ids[name], "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    write_json(output, {"images": images, "annotations": annotations,
                        "categories": [{"id": i, "name": name} for name, i in class_ids.items()]})


def clipped_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效框：{box}")
    return [x1, y1, x2, y2]
