"""单图检测，输出 JSON 和绘制检测框的图片。"""
import argparse
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from inference import Detector
from utils import write_json


def main():
    """读取图片与检查点，保存检测框图片和结构化结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/prediction"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    detector = Detector(args.checkpoint, args.device, args.threshold)
    results = detector.predict_frame(image)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "detections.json", {"image": str(args.image.resolve()),
               "width": image.width, "height": image.height, "coordinate_format": "original_pixel_xyxy",
               "detections": [asdict(result) for result in results]})
    draw = ImageDraw.Draw(image)
    # 数字类别 ID 避免系统字体不支持中文；完整类别名保存在 JSON。
    for result in results:
        draw.rectangle(result.box_xyxy, outline="red", width=3)
        draw.text(result.box_xyxy[:2], f"{result.category_id}: {result.score:.2f}", fill="red")
    image.save(args.output / "prediction.jpg")
    print(f"Detections: {len(results)}; output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
