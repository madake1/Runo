"""Named dataset factory for the common COCO detection interface."""
from pathlib import Path

from configs.defaults import DATASET_CHOICES
from datasets.coco import DetectionDataset


def build_dataset(name: str, split: str, config: dict, *, augment: bool = False):
    if name not in DATASET_CHOICES:
        raise ValueError(f"未知数据集 {name!r}，可选 {DATASET_CHOICES}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"未知划分 {split!r}")
    annotation = Path(config["annotation_dir"]) / f"{split}.json"
    if not annotation.is_file():
        raise FileNotFoundError(
            f"缺少 {name} 的 {split} 标注：{annotation}。先准备数据；步骤见 datasets/{name}/README.md")
    return DetectionDataset(config["image_root"], annotation, augment=augment)
