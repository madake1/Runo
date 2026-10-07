"""Model loading and inference; camera and drawing code live elsewhere."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


MODEL_FILE = "yolo11n.pt"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class ProduceDetector:
    def __init__(self, conf: float = 0.25, imgsz: int = 640) -> None:
        self.conf = conf
        self.imgsz = imgsz
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Ultralytics downloads missing weights relative to the working directory.
        original_dir = Path.cwd()
        try:
            os.chdir(model_dir)
            self.model = YOLO(MODEL_FILE)
        finally:
            os.chdir(original_dir)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        found = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            found.append(
                Detection(
                    label=result.names[class_id],
                    confidence=float(box.conf.item()),
                    xyxy=tuple(int(v) for v in box.xyxy[0].tolist()),
                )
            )
        return found
