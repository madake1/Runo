"""Train a small plum detector and export a fixed-shape ONNX for Android."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/mobile_plum_dataset/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/mobile_plum"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Missing {args.data}; run mobile/prepare_yolo.py first")
    model = YOLO(args.model)
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                         imgsz=args.imgsz, batch=args.batch, device=args.device,
                         project=str(args.project.resolve()), name="train", exist_ok=True,
                         seed=42, workers=0)
    best = Path(result.save_dir) / "weights" / "best.pt"
    detector = YOLO(str(best))
    exported = Path(detector.export(format="onnx", imgsz=args.imgsz,
                                    dynamic=False, simplify=False, opset=17, nms=False))
    metadata = {"input_width": args.imgsz, "input_height": args.imgsz,
                "input_layout": "NCHW", "input_color": "RGB",
                "input_range": "0..1", "output": "yolo11_detect_raw",
                "classes": [detector.names[i] for i in sorted(detector.names)],
                "confidence_threshold": 0.35, "nms_iou_threshold": 0.45,
                "source_checkpoint": best.name}
    exported.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(exported)


if __name__ == "__main__":
    main()
