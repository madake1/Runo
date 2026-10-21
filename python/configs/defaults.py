"""Edit this file to change the default training experiment.

Model and dataset names are stable CLI choices. The paths below are relative to
python/, and old YAML experiments can still be selected with --config.
"""
from pathlib import Path

MODEL_CHOICES = (
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn_v2",
)
DATASET_CHOICES = ("plantdoc", "fengtangli", "acfr_plum")

DEFAULT_MODEL = MODEL_CHOICES[0]
DEFAULT_DATASET = DATASET_CHOICES[0]

TRAIN_DEFAULTS = {
    "pretrained": True,
    "device": "auto",
    "cpu_threads": 4,
    "seed": 42,
    "epochs": 20,
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.005,
    "weight_decay": 0.0005,
    "lr_step": 10,
    "freeze_backbone_epochs": 2,
    "freeze_bn": True,
    "amp": True,
}

# Keep model size and memory needs visible when choosing defaults.
MODEL_BATCH_SIZE = {
    "ssdlite320_mobilenet_v3_large": 4,
    "fasterrcnn_mobilenet_v3_large_320_fpn": 2,
    "fasterrcnn_resnet50_fpn_v2": 2,
}

DATASET_PATHS = {
    "plantdoc": ("data/plantdoc/raw", "data/plantdoc/annotations"),
    "fengtangli": ("data/fengtangli/raw", "data/fengtangli/annotations"),
    "acfr_plum": ("data/acfr_plum/raw", "data/acfr_plum/annotations"),
}


def default_config(model: str = DEFAULT_MODEL, dataset: str = DEFAULT_DATASET) -> dict:
    if model not in MODEL_CHOICES or dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported model/dataset: {model}/{dataset}")
    root = Path(__file__).resolve().parents[1]
    image_root, annotation_dir = DATASET_PATHS[dataset]
    return {**TRAIN_DEFAULTS, "model": model, "dataset": dataset,
            "batch_size": MODEL_BATCH_SIZE[model],
            "image_root": str(root / image_root),
            "annotation_dir": str(root / annotation_dir),
            "run_dir": str(root / "runs" / f"{dataset}_{model}")}
