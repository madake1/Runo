# Python 模型工程

本目录负责数据准备、目标检测训练、评估、预测，以及移动端模型导出。`datasets/` 读取和转换数据，`models/`、`engine.py`、`train.py` 实现手写 PyTorch 实验流程，`mobile/` 负责 YOLO 训练与 ONNX 导出，`runs/` 保存本地实验产物。

命令、各模块职责和数据流见 [Python 模块说明](../docs/python.md)。
