# 数据集适配与加载

每个数据集一个子目录，放独立提取方案和脚本：`plantdoc/`、`fengtangli/`。通用检查和划分位于 `common.py`，统一 COCO 检测加载位于 `coco.py`。

图片与标注按数据集放在 `data/<dataset>/`。检测样本返回图像张量和 target 字典，包含 boxes、labels 等；健康负样本支持空框。训练、验证和推理使用相同类别映射，随机增强仅用于训练。
