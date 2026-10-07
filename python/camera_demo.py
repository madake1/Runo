"""Webcam demo: python python/camera_demo.py --camera 0"""

import argparse
import time

import cv2

from detector import ProduceDetector
from display import draw_detections, draw_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained YOLO detection on a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (try 1 if 0 is wrong)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    detector = ProduceDetector(conf=args.conf, imgsz=args.imgsz)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}; try --camera 1")

    previous = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera returned no frame.")
                break

            detections = detector.predict(frame)
            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            draw_detections(frame, detections)
            draw_status(frame, fps, len(detections))
            cv2.imshow("Runo | YOLO11 COCO | Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
