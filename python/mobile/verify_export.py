"""Check ONNX output shape and compare one image against the PyTorch model."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/mobile_plum/train/weights/best.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("runs/mobile_plum/train/weights/best.onnx"))
    parser.add_argument("--image", type=Path, default=Path("data/acfr_plum/raw/Day/JPEGImages/0304.jpg"))
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    torch_output = model.predict(str(args.image), imgsz=640, conf=0.35, iou=0.45,
                                 rect=False, device="cpu", verbose=False)[0]
    onnx_output = YOLO(str(args.onnx)).predict(str(args.image), imgsz=640, conf=0.35,
                                               iou=0.45, device="cpu", verbose=False)[0]
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1 and inputs[0].shape == [1, 3, 640, 640]
    assert len(outputs) == 1 and outputs[0].shape == [1, 5, 8400]
    a = torch_output.boxes.xyxy.cpu().numpy()
    b = onnx_output.boxes.xyxy.cpu().numpy()
    assert len(a) == len(b), f"Different box counts: PyTorch={len(a)}, ONNX={len(b)}"
    assert np.allclose(a, b, atol=2.0), "ONNX boxes differ from PyTorch"
    print(f"ONNX parity passed: {len(a)} boxes; input {inputs[0].shape}; output {outputs[0].shape}")


if __name__ == "__main__":
    main()
