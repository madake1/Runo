"""最小 COCO 检测 Dataset。图片不提前归一化，模型内部完成缩放和归一化。"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class DetectionDataset(Dataset):
    def __init__(self, image_root, annotation_path, augment=False):
        self.root = Path(image_root).resolve()
        data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda item: item["id"])
        if [c["id"] for c in categories] != list(range(1, len(categories)+1)):
            raise ValueError("类别 ID 必须从 1 连续编号，0 留给背景")
        self.classes = [c["name"] for c in categories]
        self.images = data["images"]
        self.annotations = defaultdict(list)
        self.augment = augment
        for annotation in data["annotations"]:
            if annotation.get("iscrowd", 0):
                raise ValueError("当前教学框架不支持 crowd 标注")
            if not 1 <= annotation["category_id"] <= len(categories):
                raise ValueError("标注类别编号超出范围")
            self.annotations[annotation["image_id"]].append(annotation)
        if not self.images:
            raise ValueError("数据集为空")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = (self.root / info["file_name"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("图片路径超出数据根目录")
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (info["width"], info["height"]):
            raise ValueError(f"图片尺寸与标注不一致：{path}")
        boxes, labels = [], []
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > image.width+0.01 or y+h > image.height+0.01:
                raise ValueError(f"框无效：{path}")
            boxes.append([x, y, x+w, y+h])
            labels.append(annotation["category_id"])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = F.hflip(image)
            boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor(info["id"]),
                  "area": (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                  "iscrowd": torch.zeros(len(labels), dtype=torch.int64)}
        return F.to_tensor(image), target


def collate_fn(batch):
    # 检测图片尺寸和框数不相同，不能直接 stack。
    images, targets = zip(*batch)
    return list(images), list(targets)
