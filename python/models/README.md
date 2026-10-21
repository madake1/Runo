# 检测模型接入

放置检测模型构建、预训练权重加载、类别预测头替换和主干冻结/解冻代码。

`network.py` 提供模型工厂 `build_model(name, num_classes, pretrained)`，支持 SSDlite320 MobileNetV3-Large、Faster R-CNN MobileNetV3 320 FPN 和更大的 Faster R-CNN ResNet50 FPN V2。`num_classes` 为前景数，内部增加背景类别。支持的名称及默认选择集中在 `configs/defaults.py`，模型工厂复用 torchvision 预训练权重，并替换数据集分类头。

检测模型接收图像张量列表。训练时传入 target 列表，返回损失字典；推理时返回每张图的 boxes、labels、scores，框为原图像素 xyxy。模型内部负责归一化和 NMS。

训练生成的权重放在 `runs/` 对应实验目录中。
