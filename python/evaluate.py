"""独立评估：只读取模型，不恢复优化器，也不更新参数。"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.coco import DetectionDataset, collate_fn
from datasets.factory import build_dataset
from engine import evaluate
from models import build_model
from utils import choose_device, load_checkpoint, write_json


def main():
    """解析评估参数，加载检查点并输出指定划分的指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--image-root", type=Path, help="可指定外部数据集，但类别映射必须相同")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    torch.set_num_threads(config.get("cpu_threads", 4))
    data = (DetectionDataset(args.image_root or config["image_root"],
                             args.annotations or Path(config["annotation_dir"]) / f"{args.split}.json")
            if args.image_root or args.annotations else
            build_dataset(config.get("dataset", "plantdoc"), args.split, config))
    if data.classes != checkpoint["classes"]:
        raise ValueError("评估数据与模型类别映射不同")
    device = choose_device(args.device)
    model = build_model(checkpoint["model_name"], len(data.classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(data, batch_size=config["batch_size"], collate_fn=collate_fn)
    metrics = evaluate(model, loader, device, data.classes)
    output = args.output or args.checkpoint.parent / f"evaluation_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
