# Python 模型实验

建议使用独立虚拟环境。在仓库根目录执行（Windows PowerShell）：

```powershell
python -m venv python/.venv
python/.venv/Scripts/python.exe -m pip install -r python/requirements.txt
python/.venv/Scripts/python.exe python/camera_demo.py --camera 0
```

需要 Python 3.10 或以上版本，以及可用的摄像头。按 Q 或 Esc 退出；摄像头索引不正确时尝试 `--camera 1`。

- `camera_demo.py`：摄像头读取、检测循环及 FPS 计算。
- `detector.py`：加载模型并解析检测结果。
- `display.py`：绘制检测框、类别、置信度和 FPS。
- `models/yolo11n.pt`：当前使用的预训练 COCO 模型。
- `models/yolov8s-worldv2.pt`：已有实验权重，当前入口未使用。

模型路径相对于本目录解析，不依赖启动时的工作目录。缺少当前权重时 Ultralytics 会尝试下载；首次安装依赖也需要网络。

目前没有独立图片检测、训练或 ONNX 导出脚本。此环境仅用于实验，不参与 C++ 编译或 Android 打包。后续通过导出的模型和明确的输入输出约定与 C++ 对接。
