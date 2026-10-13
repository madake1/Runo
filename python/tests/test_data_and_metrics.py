"""验证容易出错的坐标、来源组和检测 AP，不依赖下载或预训练权重。"""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image

from datasets.coco import DetectionDataset
from datasets.common import group_split, local_filename, save_coco
from metrics.detection import detection_metrics


class DetectionTests(unittest.TestCase):
    def test_duplicate_sources_are_bound(self):
        records = [{"group": "tree_a", "digest": "same"},
                   {"group": "tree_b", "digest": "same"},
                   {"group": "tree_b", "digest": "other"},
                   {"group": "tree_c", "digest": "third"}]
        train, val = group_split(records)
        self.assertFalse({r['group'] for r in train} & {r['group'] for r in val})
        self.assertFalse({r['digest'] for r in train} & {r['digest'] for r in val})

    def test_windows_filename_mapping(self):
        self.assertNotIn("?", local_filename("image.jpg?size=100.jpg"))
        self.assertEqual(local_filename("image.jpg"), "image.jpg")

    def test_dataset_and_negative_sample(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "green").save(root / "a.jpg")
            Image.new("RGB", (20, 10), "blue").save(root / "b.jpg")
            records = [{"path": "a.jpg", "width": 20, "height": 10,
                        "boxes": [("disease", [1, 2, 8, 9])]},
                       {"path": "b.jpg", "width": 20, "height": 10, "boxes": []}]
            save_coco(records, ["disease"], root / "labels.json")
            dataset = DetectionDataset(root, root / "labels.json")
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 10, 20))
            self.assertEqual(target["boxes"].tolist(), [[1, 2, 8, 9]])
            self.assertEqual(tuple(dataset[1][1]["boxes"].shape), (0, 4))
            dataset.augment = True
            with patch("datasets.coco.random.random", return_value=0):
                self.assertEqual(dataset[0][1]["boxes"].tolist(), [[12, 2, 19, 9]])

    def test_fengtangli_labelme_conversion(self):
        from datasets.fengtangli.prepare import main
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            groups = {}
            for index in range(8):
                name = f"image_{index}"
                Image.new("RGB", (20, 10), (index*20, 10, 30)).save(raw / f"{name}.png")
                (raw / f"{name}.json").write_text(json.dumps({
                    "imagePath": f"{name}.png", "shapes": [
                        {"label": "leaf_symptom", "shape_type": "rectangle", "points": [[1, 1], [8, 8]]},
                        {"label": "fruit_symptom", "shape_type": "rectangle", "points": [[10, 1], [18, 8]]}]}))
                groups[f"{name}.png"] = f"tree_{index}"
            (root / "groups.json").write_text(json.dumps(groups))
            (root / "classes.json").write_text(json.dumps(["leaf_symptom", "fruit_symptom"]))
            argv = ["prepare", "--raw", str(raw), "--groups", str(root / "groups.json"),
                    "--classes", str(root / "classes.json"), "--output", str(root / "annotations")]
            with patch("sys.argv", argv):
                main()
            subsets = []
            for split in ["train", "val", "test"]:
                data = DetectionDataset(raw, root / "annotations" / f"{split}.json")
                subsets.append({info["file_name"] for info in data.images})
                self.assertEqual(data.classes, ["leaf_symptom", "fruit_symptom"])
            self.assertFalse(subsets[0] & subsets[1] or subsets[0] & subsets[2] or subsets[1] & subsets[2])

    def test_ap_perfect_and_duplicate_false_positive(self):
        truth = [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1])},
                 {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1])}]
        perfect = [{**t, "scores": torch.tensor([0.9])} for t in truth]
        self.assertAlmostEqual(detection_metrics(perfect, truth, ["disease"])["map_50"], 1)
        repeated = [{"boxes": torch.tensor([[0.,0.,10.,10.], [0.,0.,10.,10.]]),
                     "labels": torch.tensor([1, 1]), "scores": torch.tensor([0.99, 0.98])}, perfect[1]]
        score = detection_metrics(repeated, truth, ["disease"])["map_50"]
        self.assertGreater(score, 0)
        self.assertLess(score, 1)


if __name__ == "__main__":
    unittest.main()
