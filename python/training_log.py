"""Rich terminal reporting and Lightning Fabric experiment loggers."""
import logging
from contextlib import contextmanager

from lightning_fabric.loggers import CSVLogger, TensorBoardLogger
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (Progress, TextColumn, BarColumn, MofNCompleteColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

console = Console()


def setup_logging(path=None):
    """配置终端与文件日志输出。"""
    logger = logging.getLogger("runo")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    logger.addHandler(RichHandler(console=console, show_path=False, markup=False))
    if path:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


class TrainingLog:
    """统一训练进度条、终端日志和指标输出。"""
    def __init__(self, run, config, device, train_size, val_size, classes, start):
        """初始化训练进度显示和指标记录器。"""
        self.log = setup_logging(run / "train.log")
        self.writers = [TensorBoardLogger(str(run), name="tensorboard", version=""),
                        CSVLogger(str(run), name="csv", flush_logs_every_n_steps=50)]
        self.writers[0].log_hyperparams(config)
        table = Table.grid(padding=(0, 2))
        gpu = str(device)
        if device.type == "cuda":
            import torch
            gpu += " | " + torch.cuda.get_device_name(device)
        for label, value in [("Model", config["model"]), ("Device", gpu),
                             ("Data", f"train {train_size:,} | val {val_size:,} | classes {classes}"),
                             ("Training", f"epochs {start+1}-{config['epochs']} | batch {config['batch_size']} | lr {config['lr']}"),
                             ("Run", str(run))]:
            table.add_row(f"[bold cyan]{label}[/]", str(value))
        console.print(Panel(table, title="[bold cyan]Runo | Detection Training[/]", border_style="cyan"))
        self.progress = Progress(
            TextColumn("{task.description}"), BarColumn(complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(), TextColumn("{task.percentage:>5.1f}%"),
            TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
            TextColumn("{task.fields[speed]}"), TextColumn("{task.fields[metrics]}"),
            console=console, refresh_per_second=4,
        )
        self.epoch_task = None

    def __enter__(self):
        """启动进度界面并返回日志对象。"""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """结束进度界面并释放日志资源。"""
        self.progress.stop()
        status = "success" if exc_type is None else "failed"
        for writer in self.writers:
            writer.finalize(status)
        if exc_type:
            self.log.error("Training interrupted: %s", exc)

    @contextmanager
    def phase(self, name, total):
        """创建一个训练或验证阶段的进度任务。"""
        if self.epoch_task is not None:
            self.progress.remove_task(self.epoch_task)
        task = self.progress.add_task(name, total=total, speed="", metrics="")
        self.epoch_task = task
        yield task

    def advance(self, task, metrics=""):
        """推进批次进度并更新当前指标文本。"""
        self.progress.update(task, advance=1, metrics=metrics)
        speed = self.progress.tasks[0].speed
        self.progress.update(task, speed=f"{speed:.2f} batch/s" if speed else "")

    def metrics(self, values, step):
        """把当前指标写入 CSV 和 TensorBoard。"""
        for writer in self.writers:
            writer.log_metrics(values, step=step)

    def epoch(self, epoch, loss, metrics, lr, best, improved):
        """汇总一轮训练结果并记录最佳指标状态。"""
        for writer in self.writers:
            writer.save()
        if self.epoch_task is not None:
            self.progress.update(self.epoch_task, metrics=f"AP50={metrics['map_50']:.4f} AP50:95={metrics['map_50_95_simple']:.4f}")
        table = Table(title=f"Epoch {epoch} | Validation", border_style="dim")
        for name in ["Loss", "AP50", "AP50:95 (simple)", "Best", "LR"]:
            table.add_column(name, justify="right")
        table.add_row(f"{loss['total']:.4f}", f"{metrics['map_50']:.4f}",
                      f"{metrics['map_50_95_simple']:.4f}", f"{best:.4f}", f"{lr:.3g}")
        console.print(table)
        self.log.info("Epoch %d | loss %.4f | AP50 %.4f | AP50:95 %.4f | lr %.3g",
                      epoch, loss['total'], metrics['map_50'], metrics['map_50_95_simple'], lr)
        if improved:
            console.print("[bold green]New best checkpoint saved[/]")
            self.log.info("Saved best.pt | score %.4f", best)
