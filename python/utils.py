"""配置、随机种子、设备选择和权重读写。"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

BASE = Path(__file__).resolve().parent


def load_config(path):
    """读取 YAML 配置并合并项目默认参数。"""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ["image_root", "annotation_dir", "run_dir"]:
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else BASE / value)
    return config


def seed_everything(seed):
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """为数据加载器工作进程设置可复现的随机种子。"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def choose_device(name):
    """根据配置与硬件状态选择 CPU 或 CUDA。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请设置 device: cpu 或 auto")
    return torch.device(name)


def save_checkpoint(path, checkpoint):
    """原子写入训练检查点。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # 本项目仅保存 tensors 和普通容器，保持 weights_only 安全加载。
    """从磁盘读取训练检查点。"""
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, value):
    """把结构化结果以 UTF-8 JSON 保存到磁盘。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
