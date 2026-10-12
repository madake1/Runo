# 模型接入

放置模型构建、预训练权重加载、分类头替换和主干冻结/解冻代码。

开始可以在 `network.py` 中统一提供 `build_model(name, num_classes, pretrained)`，逐步接入 MobileNetV3-Small、MobileNetV3-Large 和 EfficientNet-B0。

分类模型统一接收 `[B, C, H, W]` 图像张量，返回 `[B, num_classes]` 原始 logits，方便复用训练循环。使用 CrossEntropyLoss 时，模型输出不提前做 softmax。

训练生成的权重放在 `runs/` 对应实验目录中。
