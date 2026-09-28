"""OpenCV overlays; replace this module without changing model inference."""

import cv2
import numpy as np

from detector import Detection


COLORS = {"apple": (60, 60, 240), "banana": (0, 220, 220),
          "orange": (0, 140, 255)}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for item in detections:
        x1, y1, x2, y2 = item.xyxy
        color = COLORS.get(item.label, (60, 200, 60))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{item.label} {item.confidence:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_status(frame: np.ndarray, fps: float, count: int) -> None:
    text = f"FPS: {fps:.1f}  detections: {count}  Q: quit"
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
