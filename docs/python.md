# Python 模块与数据流

`python/` 包含两条检测实验路径：手写 PyTorch 训练框架和移动端 YOLO 导出流程。它们共用原始 ACFR 数据及划分，但模型和输出目录分开。

| 模块 | 职责 |
| --- | --- |
| `datasets/`、`data/` | 下载或读取 PlantDoc、ACFR 和自采蜂糖李数据；转换为 COCO，维护来源组与训练/验证/测试划分 |
| `configs/` | 默认参数和各实验 YAML 配置 |
| `models/`、`engine.py`、`train.py` | 构建 torchvision 检测器，执行训练/验证并保存断点 |
| `metrics/`、`evaluate.py`、`test.py` | 计算 AP，保存测试报告与框图对照 |
| `inference.py`、`predict.py` | 加载手写框架的权重，输出图片或逐帧检测结果 |
| `training_log.py`、`runs/` | Rich 进度、CSV/TensorBoard、配置、权重和评估产物 |
| `mobile/` | ACFR COCO→YOLO 转换、YOLO11n 训练、ONNX 导出和 PyTorch/ONNX 单图核对 |

手写框架数据流：`原图和标注 → 数据集转换/COCO → Dataset → torchvision 检测模型 → 损失与优化 → checkpoint → 测试指标/框图`。

移动端数据流：`ACFR COCO → YOLO 图片/标签 → YOLO11n 训练 → best.pt → best.onnx + best.json → Android assets`。ONNX 约定输入为 `1×3×640×640`、RGB、0–1；输出是未内置 NMS 的检测张量，Android 侧负责后处理。

在 `python/` 目录使用已有的 `Runo` Conda 环境。手写框架可运行 `python train.py --dataset acfr_plum --model fasterrcnn_resnet50_fpn_v2`、`python evaluate.py --checkpoint <run>/best.pt --split test`、`python test.py --checkpoint <run>/best.pt`。移动端流程依次执行：

```powershell
python -m pip install -r mobile/requirements.txt
python -m datasets.acfr_plum.download
python -m datasets.acfr_plum.prepare
python mobile/prepare_yolo.py
python mobile/train_export.py --epochs 30
python mobile/verify_export.py
Copy-Item runs/mobile_plum/train/weights/best.onnx ../android/app/src/main/assets/best.onnx
Copy-Item runs/mobile_plum/train/weights/best.json ../android/app/src/main/assets/best.json
```

PlantDoc 使用 `python -m datasets.plantdoc.download` 和 `python -m datasets.plantdoc.prepare`；蜂糖李自采数据使用 LabelMe 矩形框、`data/fengtangli/metadata/classes.json` 类别列表和 `groups.json` 来源组映射，再运行 `python -m datasets.fengtangli.prepare`。同树连拍应保持在同一划分。数据来源和物种核验见 [数据调研](dataset-research.md)，实验结果见 [开发日志](Logs.md)。
