# ACFR 李子果实定位：Faster R-CNN ResNet50 FPN V2

实验日期：2026-10-17（Asia/Shanghai）  
运行目录：[20261017_223048_435584](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/)  
状态：**完整训练与独立测试已完成**，2026-10-17 22:30:48 至 22:33:00，训练约 2 分 12 秒。

## 目的与任务

用较大的检测模型验证 ACFR 果园李子数据上的单类果实定位能力，并与当前 SSDlite 学习基线区分。标签只有 `plum`，本实验**不能评估蜂糖李病害识别能力**。

## 数据

- 来源：[University of Sydney ACFR Plum Harvesting Detection Dataset](https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/)，本次使用日间 RGB 图像与 Pascal VOC 框。
- 原始数据：350 张 640×480 图像，官方标注的红色李子果实。
- 转换：`python -m datasets.acfr_plum.prepare` 生成 COCO JSON，只有 `plum` 一个前景类别，背景为 0。
- 划分：训练 **245 张 / 2,659 框**；验证 **48 张 / 785 框**；测试 **47 张 / 845 框**。按帧序号分连续区段，两处边界各舍弃 5 帧；跨划分像素重复会去重。
- 局限：来自同一数据来源，时间隔离不能证明跨果园泛化；与贵州蜂糖李的品种、光照、病害任务均有差异。
- 原始日间包 SHA256：`574381B28945C2B9746151D1DDA5A6F1961251BACD78CE15644C7C3CA007F311`。

## 模型与训练参数

配置文件：[acfr_plum_resnet50_fpn_v2.yaml](../../python/configs/acfr_plum_resnet50_fpn_v2.yaml)。

| 参数 | 值 |
| --- | --- |
| 模型 | `fasterrcnn_resnet50_fpn_v2`，torchvision COCO 预训练 |
| 数据集 | `acfr_plum`，日间单类检测 |
| 训练轮数 | 12 |
| batch size | 2 |
| 优化器 | SGD，momentum 0.9，weight decay 0.0005 |
| 初始学习率 | 0.005 |
| 调度器 | StepLR，8 轮后学习率乘 0.1 |
| 主干 | 前 2 轮冻结，之后解冻 |
| BatchNorm | 固定运行统计 |
| AMP | CUDA 时启用 |
| 随机种子 | 42 |
| 图像增强 | 训练集随机水平翻转 50%，验证/测试不增强 |
| 最佳权重选择 | 验证集简化 AP50:95 最高值 |
| 硬件 / 运行时 | NVIDIA GeForce RTX 5090 D v2，PyTorch 2.14.0+cu130，CUDA |

预训练权重为 torchvision 官方 `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`，SHA256：`DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF`。模型载入 COCO 权重后，将分类与回归预测头改为本数据集的单前景类别。输入图片由模型内部缩放、归一化；验证与测试不参与参数更新。

启动命令（在 `python/` 目录）：

```powershell
conda activate Runo
python train.py --config configs/acfr_plum_resnet50_fpn_v2.yaml
```

## 逐轮训练记录

以下数值来自原始 [metrics.jsonl](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/metrics.jsonl)。AP50:95 是项目的简化实现，未做 COCO crowd/area 等分组，**不能等同官方 COCO mAP**。

| Epoch | 学习率 | 训练 loss | 验证 AP50 | 验证 AP50:95（简化） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.005 | 0.7061 | 0.6334 | 0.3244 |
| 2 | 0.005 | 0.5703 | 0.6965 | 0.3670 |
| 3 | 0.005 | 0.6463 | 0.6977 | 0.3484 |
| 4 | 0.005 | 0.5602 | 0.7427 | 0.3967 |
| 5 | 0.005 | 0.5177 | 0.7891 | 0.3999 |
| 6 | 0.005 | 0.4725 | 0.7930 | 0.4325 |
| 7 | 0.005 | 0.4570 | 0.7172 | 0.3662 |
| 8 | 0.005 | 0.4296 | 0.8167 | 0.4430 |
| 9 | 0.0005 | 0.3750 | 0.8574 | 0.4838 |
| 10 | 0.0005 | 0.3347 | 0.8637 | 0.4926 |
| 11 | 0.0005 | 0.3226 | 0.8564 | 0.4909 |
| **12** | **0.0005** | **0.3075** | **0.8567** | **0.4943** |

第 12 轮的简化 AP50:95 最高，因此 [best.pt](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/best.pt) 保存这一轮；[last.pt](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/last.pt) 是恢复训练用的最后断点，本次恰好同为第 12 轮。训练 loss 从 0.7061 降到 0.3075；中途验证指标有波动，最佳权重按验证指标而非 loss 选出。

## 独立测试

使用 `best.pt` 对完整测试集 **47 张 / 845 个真实框**做一次评估，无 `--max-images` 限制。原始 [report.json](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/report.json) 记录了样本数、类别、指标与对照图清单。

| 指标 | 结果 |
| --- | ---: |
| AP50 | **0.8155** |
| AP50:95（简化） | **0.4234** |
| `plum` 类 AP50 | **0.8155** |

测试 AP50 比最佳验证 AP50 低约 0.041，简化 AP50:95 低约 0.071。测试集仍来自同一来源的相邻时段，结果只能说明此数据来源上的果实定位能力。原始测试对照图阈值为 0.4；**阈值只控制图上显示哪些预测框，不改变 AP 计算**。抽查 [sample_001.jpg](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_001.jpg) 和 [sample_008.jpg](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_008.jpg)：多数果实位置对齐，密集枝叶图可见额外预测框。

![第 1 张测试图：绿色真实框与红色预测框](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_001.jpg)

## 日志和复现

- 训练事件：[train.log](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/train.log)
- 逐轮完整指标：[metrics.jsonl](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/metrics.jsonl)
- 批次与逐轮 CSV：[csv/version_0/metrics.csv](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/csv/version_0/metrics.csv)
- TensorBoard 曲线：[tensorboard/](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tensorboard/)
- 测试日志：[tests/final/test.log](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/test.log)
- 实际运行配置：[config.yaml](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/config.yaml)；[环境信息](../../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/environment.json)

```powershell
cd python
python test.py --checkpoint runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/best.pt --split test --samples 8 --output runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/recheck
python -m tensorboard.main --logdir runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tensorboard
```

本实验可用作果实定位基线；要评估蜂糖李病害，必须再有经确认的蜂糖李病叶、病果或病斑位置标注与独立果园测试集。
