"""Cross-check named factories and source-frame split boundaries."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from configs.defaults import DATASET_CHOICES, MODEL_CHOICES, default_config
from datasets.acfr_plum.prepare import split_ids
from datasets.factory import build_dataset


class FactoryTests(unittest.TestCase):
    """验证数据集及默认配置工厂。"""
    def test_acfr_source_frames_do_not_touch_split_boundaries(self):
        """验证 ACFR 连续帧划分边界隔离。"""
        splits, gap = split_ids([f"{index:04d}" for index in range(1, 351)])
        self.assertEqual(gap, 5)
        self.assertEqual([len(splits[key]) for key in ("train", "val", "test")], [245, 48, 47])
        self.assertGreater(int(splits["val"][0]) - int(splits["train"][-1]), gap)
        self.assertGreater(int(splits["test"][0]) - int(splits["val"][-1]), gap)

    def test_named_dataset_uses_its_coco_annotations(self):
        """验证工厂按名称加载对应 COCO 标注。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as folder:
            root = Path(folder)
            Image.new("RGB", (8, 8)).save(root / "sample.jpg")
            coco = {"images": [{"id": 1, "file_name": "sample.jpg", "width": 8, "height": 8}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                     "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0}],
                    "categories": [{"id": 1, "name": "plum"}]}
            (root / "train.json").write_text(json.dumps(coco), encoding="utf-8")
            config = default_config(dataset="acfr_plum")
            config.update(image_root=str(root), annotation_dir=str(root))
            dataset = build_dataset("acfr_plum", "train", config)
            self.assertEqual(dataset.classes, ["plum"])
            self.assertEqual(dataset[0][1]["boxes"].tolist(), [[1.0, 1.0, 4.0, 4.0]])
            with self.assertRaises(FileNotFoundError):
                build_dataset("acfr_plum", "test", config)

    def test_default_choices_are_registered(self):
        """验证默认模型和数据集名称已注册。"""
        self.assertIn(default_config()["model"], MODEL_CHOICES)
        self.assertIn(default_config()["dataset"], DATASET_CHOICES)


if __name__ == "__main__":
    unittest.main()
