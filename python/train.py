"""训练入口。阅读顺序：配置 -> Dataset -> 模型 -> engine -> checkpoint。"""
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from configs.defaults import (DATASET_CHOICES, DATASET_PATHS, DEFAULT_DATASET,
                              MODEL_BATCH_SIZE, MODEL_CHOICES, default_config)
from datasets.coco import collate_fn
from datasets.factory import build_dataset
from engine import evaluate, train_one_epoch
from models.network import build_model, freeze_backbone
from training_log import TrainingLog, setup_logging, console
from utils import (BASE, choose_device, load_checkpoint, load_config,
                   save_checkpoint, seed_everything, seed_worker, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="可选：加载旧实验 YAML，再由命令行覆盖模型或数据集")
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES,
                        help="模型名称；默认值在 configs/defaults.py")
    parser.add_argument("--dataset", type=str, choices=DATASET_CHOICES,
                        help="数据集名称；默认值在 configs/defaults.py")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="覆盖总训练轮数")
    parser.add_argument("--batch-size", type=int, help="覆盖模型对应的默认 batch size")
    parser.add_argument("--max-steps", type=int, help="每轮训练和验证限制步数，仅用于冒烟检查")
    parser.add_argument("--no-pretrained", action="store_true", help="离线调试；正式训练推荐预训练")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    log = setup_logging()
    log.info("Loading configuration, datasets and model...")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = (load_config(args.config) if args.config else
              dict(checkpoint["config"]) if checkpoint else default_config())
    if args.dataset:
        config["dataset"] = args.dataset
        image_root, annotation_dir = DATASET_PATHS[args.dataset]
        config["image_root"] = str(BASE / image_root)
        config["annotation_dir"] = str(BASE / annotation_dir)
    config.setdefault("dataset", DEFAULT_DATASET)
    if args.model:
        config["model"] = args.model
        config["batch_size"] = MODEL_BATCH_SIZE[args.model]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config["model"] not in MODEL_CHOICES or config["dataset"] not in DATASET_CHOICES:
        parser.error(f"model 和 dataset 必须分别属于 {MODEL_CHOICES}、{DATASET_CHOICES}")
    if args.model or args.dataset:
        config["run_dir"] = str(BASE / "runs" / f"{config['dataset']}_{config['model']}")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.no_pretrained:
        config["pretrained"] = False
    if args.run_dir:
        config["run_dir"] = str(args.run_dir.resolve())
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs 和 batch_size 必须为正数")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps 必须为正数")
    if config["batch_size"] == 1 and not config["freeze_bn"]:
        raise ValueError("batch_size=1 时请启用 freeze_bn")
    if args.max_steps:
        config["smoke_max_steps"] = args.max_steps
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = choose_device(config["device"])
    train_set = build_dataset(config["dataset"], "train", config, augment=True)
    val_set = build_dataset(config["dataset"], "val", config)
    if train_set.classes != val_set.classes:
        raise ValueError("训练与验证类别映射不一致")
    classes = train_set.classes
    generator = torch.Generator().manual_seed(config["seed"])
    common = dict(batch_size=config["batch_size"], num_workers=config["num_workers"],
                  collate_fn=collate_fn, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    if checkpoint:
        if checkpoint["classes"] != classes or checkpoint["model_name"] != config["model"]:
            raise ValueError("恢复训练的类别或模型与配置不同")
        if checkpoint["config"].get("dataset", DEFAULT_DATASET) != config["dataset"]:
            raise ValueError("恢复训练的数据集与断点不同")
        # 仅允许修改总轮数和设备等运行设置；防止意外换数据/优化策略。
        for key in ["image_root", "annotation_dir", "lr", "weight_decay", "freeze_backbone_epochs", "freeze_bn", "batch_size"]:
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"恢复训练配置不一致：{key}")
    torch.hub.set_dir(str(BASE / "runs/_cache/pretrained"))
    model = build_model(config["model"], len(classes), pretrained=config["pretrained"] and not checkpoint).to(device)
    # 冻结参数仍保留在 optimizer 内，后续解冻时无需重新创建优化器。
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9,
                                weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step"], gamma=0.1)
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start, best = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"]+1, checkpoint["best_metric"]
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        run = args.resume.resolve().parent
    else:
        run = Path(config["run_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    write_json(run / "classes.json", {str(i+1): name for i, name in enumerate(classes)})
    write_json(run / "environment.json", {"torch": str(torch.__version__), "device": str(device)})
    reporter = TrainingLog(run, config, device, len(train_set), len(val_set), len(classes), start)
    steps_per_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    global_step = checkpoint.get("global_step", start * steps_per_epoch) if checkpoint else 0
    if args.max_steps:
        reporter.log.warning("SMOKE CHECK: limited to %d batches per phase; metrics are partial", args.max_steps)
    with reporter:
        for epoch in range(start, config["epochs"]):
            frozen = epoch < config["freeze_backbone_epochs"]
            freeze_backbone(model, frozen)
            reporter.log.info("Epoch %d/%d | backbone %s", epoch+1, config['epochs'], "frozen" if frozen else "unfrozen")
            loss = train_one_epoch(model, train_loader, optimizer, device, scaler,
                                   amp=amp, freeze_bn=config["freeze_bn"] or frozen, max_steps=args.max_steps,
                                   reporter=reporter, epoch=epoch+1, global_step=global_step)
            global_step += steps_per_epoch
            metrics = evaluate(model, val_loader, device, classes, max_steps=args.max_steps,
                               reporter=reporter, epoch=epoch+1)
            score = metrics["map_50_95_simple"]
            improved = score > best
            best = max(best, score)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            state = np.random.get_state()
            checkpoint = {"model": model.state_dict(), "model_name": config["model"], "classes": classes,
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "epoch": epoch, "best_metric": best, "global_step": global_step,
                          "config": config, "torch_rng": torch.get_rng_state(),
                          "loader_rng": generator.get_state(), "python_rng": random.getstate(),
                          "numpy_rng": (state[0], state[1].tolist(), state[2], state[3], state[4]),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            save_checkpoint(run / "last.pt", checkpoint)
            if improved:
                save_checkpoint(run / "best.pt", checkpoint)
            record = {"epoch": epoch+1, "lr": lr, "loss": loss, "validation": metrics,
                      "partial_smoke_check": bool(args.max_steps)}
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            reporter.metrics({**{f"train/epoch_{key}": value for key, value in loss.items()},
                              "val/AP50": metrics['map_50'], "val/AP50_95_simple": score,
                              "val/best": best, "lr": lr, "epoch": epoch+1}, global_step)
            reporter.epoch(epoch+1, loss, metrics, lr, best, improved)
    reporter.log.info("Finished | checkpoints: %s", run)
    console.print(f"[cyan]TensorBoard:[/] python -m tensorboard.main --logdir \"{run / 'tensorboard'}\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("runo").warning("Training stopped by user; resume from the last completed epoch with last.pt")
        raise SystemExit(130)
    except Exception:
        logging.getLogger("runo").exception("Training failed")
        raise SystemExit(1)
