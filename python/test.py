"""训练后测试：测试集 AP、逐类指标和真实框/预测框对照图。"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeRemainingColumn
from rich.table import Table

from datasets.coco import collate_fn
from datasets.factory import build_dataset
from metrics import detection_metrics
from models import build_model
from training_log import console, setup_logging
from utils import choose_device, load_checkpoint, write_json


def comparison(image, target, prediction, classes, threshold):
    """绿色是真实标注，红色是超过显示阈值的预测；指标使用全部模型输出。"""
    width, height = image.size
    result = Image.new("RGB", (width * 2, height + 32), "white")
    result.paste(image, (0, 32))
    result.paste(image, (width, 32))
    draw = ImageDraw.Draw(result)
    draw.text((8, 8), "GROUND TRUTH (green)", fill="green")
    draw.text((width + 8, 8), f"PREDICTION (red), score >= {threshold:.2f}", fill="red")
    for data, offset, color in [(target, 0, "green"), (prediction, width, "red")]:
        for index, (box, label) in enumerate(zip(data["boxes"], data["labels"])):
            score = float(data["scores"][index]) if "scores" in data else None
            if score is not None and score < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle((x1 + offset, y1 + 32, x2 + offset, y2 + 32), outline=color, width=3)
            text = f"{int(label)}: {classes[int(label)-1]}"
            if score is not None:
                text += f" {score:.2f}"
            position = (x1 + offset, y1 + 32)
            draw.rectangle(draw.textbbox(position, text), fill="white")
            draw.text(position, text, fill=color)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=8, help="保存前 N 张对照图，0 不保存")
    parser.add_argument("--threshold", type=float, default=0.4, help="仅控制对照图显示，不影响 AP")
    parser.add_argument("--max-images", type=int, help="仅快速检查；省略时评估完整划分")
    args = parser.parse_args()
    if args.samples < 0 or not 0 <= args.threshold <= 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("samples 必须非负，threshold 在 [0,1] 内，max-images 必须为正数")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    torch.set_num_threads(config.get("cpu_threads", 4))
    dataset = build_dataset(config.get("dataset", "plantdoc"), args.split, config)
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("测试集与模型的类别映射不同")
    count = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=config["batch_size"], collate_fn=collate_fn)
    output = args.output or args.checkpoint.resolve().parent / "tests" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    log = setup_logging(output / "test.log")
    device = choose_device(args.device)
    log.info("Checkpoint: %s | split: %s | images: %d/%d | device: %s", args.checkpoint, args.split, count, len(dataset), device)
    partial = count < len(dataset)
    if partial:
        log.warning("PARTIAL CHECK: these metrics do not represent the full test set")
    if config.get("smoke_max_steps"):
        log.warning("Checkpoint was trained with limited smoke batches; scores are for pipeline checks")
    model = build_model(checkpoint["model_name"], len(dataset.classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets, previews = [], [], []
    with Progress("[cyan]{task.description}", BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("Testing", total=count)
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            for prediction, target in zip(outputs, batch_targets):
                prediction = {key: value.cpu() for key, value in prediction.items()}
                index = len(targets)
                if index < args.samples:
                    info = dataset.images[index]
                    path = dataset.root / info["file_name"]
                    with Image.open(path) as source:
                        visual = comparison(source.convert("RGB"), target, prediction, dataset.classes, args.threshold)
                    filename = f"sample_{index+1:03d}.jpg"
                    visual.save(output / filename)
                    previews.append({"image": info["file_name"], "comparison": filename})
                predictions.append(prediction)
                targets.append(target)
                progress.advance(task)
        progress.update(task, description="Computing AP")
        metrics = detection_metrics(predictions, targets, dataset.classes)
    write_json(output / "report.json", {"checkpoint": str(args.checkpoint.resolve()), "split": args.split,
               "images": count, "total_images": len(dataset), "partial": partial,
               "checkpoint_epoch": checkpoint["epoch"]+1, "smoke_checkpoint": bool(config.get("smoke_max_steps")),
               "visualization_threshold": args.threshold, "metrics": metrics, "previews": previews})
    table = Table(title=f"{args.split.upper()} results" + (" · PARTIAL" if partial else ""))
    table.add_column("Images")
    table.add_column("AP50")
    table.add_column("AP50:95 (simple)")
    table.add_row(str(count), f"{metrics['map_50']:.4f}", f"{metrics['map_50_95_simple']:.4f}")
    console.print(table)
    per_class = Table(title="Per-class AP50")
    per_class.add_column("Class")
    per_class.add_column("AP50", justify="right")
    for name, value in metrics["per_class_ap50"].items():
        per_class.add_row(name, f"{value:.4f}" if value is not None else "N/A (no ground truth)")
    console.print(per_class)
    log.info("AP50 %.4f | AP50:95 %.4f | report and %d comparisons: %s",
             metrics['map_50'], metrics['map_50_95_simple'], len(previews), output)


if __name__ == "__main__":
    main()
