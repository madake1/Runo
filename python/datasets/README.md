# 数据集适配与加载

每个数据集一个子目录，放独立提取方案和脚本：`plantdoc/`、`fengtangli/`、`acfr_plum/`。`factory.py` 按名称选择数据集，所有数据通过 `coco.py` 统一读取。通用检查和划分位于 `common.py`。

图片与标注按数据集放在 `data/<dataset>/`。检测样本返回图像张量和 target 字典，包含 boxes、labels 等；健康负样本支持空框。训练、验证和推理使用相同类别映射，随机增强仅用于训练。
