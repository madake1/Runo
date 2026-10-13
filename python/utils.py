"""配置、随机种子、设备选择和权重读写。"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

BASE = Path(__file__).resolve().parent


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ["image_root", "annotation_dir", "run_dir"]:
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else BASE / value)
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请设置 device: cpu 或 auto")
    return torch.device(name)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # 本项目仅保存 tensors 和普通容器，保持 weights_only 安全加载。
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
