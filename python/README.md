# Python 病害检测训练框架

用于学习手写 PyTorch 检测训练流程。最终任务是贵州蜂糖李果实与叶片病害定位，支持用户随手拍照片，预留视频逐帧接口。当前公开数据只用于学习流程，尚无真实蜂糖李训练数据。

## 目录职责

- `configs/`：实验配置，例如模型名称、数据路径、学习率和训练轮数。
- `data/`：数据文件，包括原始图片、标注和数据划分。
- `datasets/`：数据读取、Dataset 和图像增强代码。
- `models/`：检测模型构建、预训练权重加载和类别预测头适配代码。
- `losses/`：自定义损失函数代码。
- `metrics/`：检测评估指标代码。
- `runs/`：按实验保存配置、日志、指标和模型权重。

## 学习入口

- `train.py`：组装配置、数据、模型和优化器，启动训练。
- `engine.py`：每轮训练与验证逻辑，供不同模型复用。
- `evaluate.py`：加载权重，在指定数据集上独立评估。
- `predict.py`：加载权重，预测单张图片。

`inference.py` 提供 `Detector.predict_frame()` 和 `predict_frames()`，图片与未来视频共用。帧输入使用 RGB，检测结果为原图像素 xyxy、类别 ID、类别名和分数。

## 数据按数据集分目录

```text
data/
├── plantdoc/          # 已下载的通用叶片检测练习数据
│   ├── raw/
│   ├── source/
│   └── annotations/
├── fengtangli/        # 自采蜂糖李数据预留目录
│   ├── raw/
│   ├── metadata/
│   └── annotations/
└── plantvillage/      # 仅调研信息，无图片

datasets/
├── plantdoc/          # 独立下载、提取方案与脚本
├── fengtangli/        # 独立 LabelMe 提取方案与脚本
├── common.py
└── coco.py            # 框架统一读取 COCO
```

详细调研见 [数据调研](../docs/dataset-research.md)，具体方案见各数据集 README。PlantDoc 标的是整片叶子，不是精细病斑；它的模型不能当作蜂糖李病害模型。

## 环境与运行

使用已有的 Conda 环境，不另建环境。从仓库根目录执行：

```powershell
conda activate Runo
cd python
python -m datasets.plantdoc.download
python -m datasets.plantdoc.prepare
python train.py --config configs/plantdoc_ssdlite.yaml
```

当前本机 Runo 环境已安装 `torch 2.14.0+cu130` 与 `torchvision 0.29.0+cu130`，已验证 RTX 5090 D v2 上的 CUDA 训练。配置 `device: auto` 在 CUDA 可用时选择显卡，否则使用 CPU。该环境通过 `runo_env_priority.pth` 优先加载环境内的包，避免用户共享目录中的 CPU 版覆盖。需要补依赖时才运行 `python -m pip install -r requirements.txt`，torch 与 torchvision 须成对匹配。首次预训练模型会下载权重，缓存到 `runs/_cache/pretrained/`；`data/` 只管理数据集。

先做小规模离线检查（随机初始化，仅验证流程，不代表识别能力）：

```powershell
python train.py --config configs/plantdoc_ssdlite.yaml --no-pretrained --epochs 1 --max-steps 2 --run-dir runs/smoke
```

每次训练输出实际路径。以下 `<run>` 替换为对应实验目录：

```powershell
python evaluate.py --checkpoint "<run>/best.pt" --split test
python predict.py --checkpoint "<run>/best.pt" --image "你的图片.jpg" --output runs/prediction
python train.py --config configs/plantdoc_ssdlite.yaml --resume "<run>/last.pt" --epochs 30
```

换模型：使用 `configs/plantdoc_fasterrcnn.yaml`。换蜂糖李数据：先按照 `datasets/fengtangli/README.md` 采集标注并提取，再使用 `configs/fengtangli_ssdlite.yaml`。

## 训练日志

运行 `python train.py` 会显示 Rich 彩色进度条，包括训练/验证批次、百分比、耗时、ETA、批次速度、实时与平均 loss、学习率。每轮结束显示验证指标表，最佳模型用绿色提示；警告和错误使用分级彩色日志。终端不支持交互或重定向输出时，Rich 自动降级为静态输出。

每个实验目录保存 `train.log`、原有 `metrics.jsonl`，以及 Lightning Fabric logger 生成的 `csv/version_*/metrics.csv` 和 `tensorboard/` 事件文件。CSV 每次启动创建新版本，续训不覆盖历史记录。TensorBoard 记录分项训练损失、验证 AP、最佳指标和学习率，断点保存 `global_step` 以延续指标步数。查看曲线：

```powershell
python -m tensorboard.main --logdir runs/plantdoc_ssdlite
```

打开终端提示的本地地址。训练循环仍由 Runo 手写，实验记录使用 Lightning Fabric 的 logger。

## 训练设计

- 默认 SSDlite320 + MobileNetV3-Large，轻量检测学习基线；Faster R-CNN MobileNetV3 320 FPN 作对照。
- 复用 torchvision 网络和检测损失实现；数据加载、优化器、训练/验证循环、日志、断点恢复由 Runo 编写。
- COCO 前景类别从 1 开始，0 为背景。Dataset 输出 RGB `[0,1]`；模型内部完成缩放和归一化，不要重复归一化。
- 训练随机水平翻转，同时修改框坐标；验证不随机增强。
- SGD、StepLR、梯度裁剪；CUDA 才启用 AMP。先冻结主干，再解冻；小批量固定 BatchNorm 统计。
- 验证指标选择最佳模型，测试集用于最终评估。保存模型、优化器、调度器、AMP、epoch、类别映射和随机状态。
- 指标为全评估集 AP50、简化 AP50:95 和逐类 AP50。没有 crowd/area 分组，不宣称等价官方 COCO evaluator；缺失真值类别在报告中明确列出。

## 当前边界

尚未实现病斑分割、视频解码与追踪、ONNX 导出、量化和 Android 部署。预留逐帧协议不是已经实现视频模式。SSDLite 的 320 输入可能漏掉远景小病斑，后续需要在蜂糖李真实数据上验证分辨率与模型选择。

手机部署阶段需验证运行时算子、NMS、坐标和预处理一致性，再测速度与量化后精度。当前不承诺该 PyTorch 教学模型可直接转换到任何手机运行时。

## 检查

```powershell
python -m unittest discover -s tests -v
```

测试覆盖来源组与重复图片隔离、Windows 文件名映射、坐标转换、健康负样本和 AP 匹配。
