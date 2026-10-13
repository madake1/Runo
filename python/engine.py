"""手写训练和评估循环：模型给损失，Runo 负责反向传播和训练状态。"""
import math
from contextlib import nullcontext

import torch
from torch import nn
from metrics import detection_metrics


def train_one_epoch(model, loader, optimizer, device, scaler, amp=False, freeze_bn=False,
                    max_steps=None, reporter=None, epoch=1, global_step=0):
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    totals, steps = {}, 0
    total = min(len(loader), max_steps) if max_steps else len(loader)
    phase = reporter.phase(f"[cyan]Train · Epoch {epoch}[/]", total) if reporter else nullcontext(None)
    with phase as task:
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"非有限损失：{losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            scaler.step(optimizer)
            scaler.update()
            values = {key: value.item() for key, value in {**losses, "total": loss}.items()}
            for key, value in values.items():
                totals[key] = totals.get(key, 0) + value
            steps += 1
            if reporter:
                reporter.metrics({f"train/step_{key}": value for key, value in values.items()}, global_step + steps)
                reporter.advance(task, f"loss={values['total']:.4f} avg={totals['total']/steps:.4f} lr={optimizer.param_groups[0]['lr']:.3g}")
            if max_steps and steps >= max_steps:
                break
    if not steps:
        raise ValueError("没有训练批次，请检查 batch_size 和数据集")
    return {key: value/steps for key, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, device, classes, max_steps=None, reporter=None, epoch=1):
    model.eval()
    predictions, targets = [], []
    total = min(len(loader), max_steps) if max_steps else len(loader)
    phase = reporter.phase(f"[magenta]Validate · Epoch {epoch}[/]", total) if reporter else nullcontext(None)
    with phase as task:
        for step, (images, batch_targets) in enumerate(loader, 1):
            outputs = model([image.to(device) for image in images])
            predictions.extend({key: value.cpu() for key, value in output.items()} for output in outputs)
            targets.extend(batch_targets)
            if reporter:
                reporter.advance(task, f"images={len(targets)}")
            if max_steps and step >= max_steps:
                break
        if reporter:
            reporter.progress.update(task, metrics="Computing AP…", refresh=True)
        return detection_metrics(predictions, targets, classes)
