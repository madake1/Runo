"""图片和未来视频共享的逐帧推理接口。

Android 可实现相同结果协议：原图像素 xyxy、类别、分数；视频逐帧调用，
跟踪、抽帧和时序去抖留给上层应用。
"""
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from models import build_model
from utils import choose_device, load_checkpoint


@dataclass
class Detection:
    box_xyxy: list[float]
    category_id: int
    category_name: str
    score: float


class Detector:
    def __init__(self, checkpoint_path, device="auto", score_threshold=0.4):
        if not 0 <= score_threshold <= 1:
            raise ValueError("阈值必须在 0 和 1 之间")
        checkpoint = load_checkpoint(checkpoint_path)
        self.classes = checkpoint["classes"]
        self.device = choose_device(device)
        torch.set_num_threads(checkpoint["config"].get("cpu_threads", 4))
        self.model = build_model(checkpoint["model_name"], len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def predict_frame(self, image: Image.Image) -> list[Detection]:
        """输入 RGB PIL 图片；输出框为输入原图坐标，模型内已完成 NMS。"""
        output = self.model([to_tensor(image.convert("RGB")).to(self.device)])[0]
        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            category = int(label)
            if float(score) >= self.score_threshold and 1 <= category <= len(self.classes):
                detections.append(Detection(box.cpu().tolist(), category,
                                            self.classes[category-1], float(score)))
        return detections

    def predict_frames(self, frames: Iterable[Image.Image]) -> Iterator[list[Detection]]:
        """预留视频接口：调用方负责读取/抽帧，此处不依赖摄像头或 UI。"""
        for frame in frames:
            yield self.predict_frame(frame)
