# 开发与实验日志

## 2026-10-12：框架验证

以下是当日的手写框架验证快照；“尚未验证”仅指当时。

使用用户现有 Conda Runo 环境：PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu，无 CUDA。没有另行安装或替换环境依赖。

### 数据处理

PlantDoc 下载 2,578 张图片，原始数据与标注约 995 MB。3 个上游引用文件返回 404，4 张 CSV 尺寸无效，全部记录在报告。移除跨训练/测试像素重复图片 12 张。

最终训练 1,860、验证 466、测试 236 张，29 类。原图保留，COCO 标注引用原图，不复制或去背景。没有真实蜂糖李样本。

### 已通过的验证

- 5 项单元测试，覆盖来源组/重复图片隔离、Windows 文件名、COCO 框坐标和翻转、健康负样本、蜂糖李 LabelMe 提取、AP 匹配。
- SSDlite 随机初始化：少量训练和验证批次，保存 best/last，恢复到下一轮。
- SSDlite 官方 COCO 预训练：少量训练与验证，恢复至第3轮，验证主干从冻结到解冻。权重 SHA256 与文件名中的 `a79551df` 前缀一致。
- Faster R-CNN MobileNetV3：随机初始化的训练与验证批次，对照模型接口正常。
- 独立测试集评估、单图 JSON 与框图输出、逐帧迭代接口。
- Git diff 格式检查。

短训练只确认流程，未训练正式可用模型，其指标不代表病害识别能力。没有蜂糖李精度结论。

尚未验证：完整训练收敛、真实蜂糖李场景、视频解码/追踪、ONNX、手机算子兼容、量化和 Android 应用。运行方式与学习入口见 `python/README.md`。

---

## 2026-10-17：ACFR 李子果实定位实验

实验日期：2026-10-17（Asia/Shanghai）

运行目录：[20261017_223048_435584](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/)
状态：**完整训练与独立测试已完成**，2026-10-17 22:30:48 至 22:33:00，训练约 2 分 12 秒。

### 目的与任务

用较大的检测模型验证 ACFR 果园李子数据上的单类果实定位能力，并与当前 SSDlite 学习基线区分。标签只有 `plum`，本实验**不能评估蜂糖李病害识别能力**。

### 数据

- 来源：[University of Sydney ACFR Plum Harvesting Detection Dataset](https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/)，本次使用日间 RGB 图像与 Pascal VOC 框。
- 原始数据：350 张 640×480 图像，官方标注的红色李子果实。
- 转换：`python -m datasets.acfr_plum.prepare` 生成 COCO JSON，只有 `plum` 一个前景类别，背景为 0。
- 划分：训练 **245 张 / 2,659 框**；验证 **48 张 / 785 框**；测试 **47 张 / 845 框**。按帧序号分连续区段，两处边界各舍弃 5 帧；跨划分像素重复会去重。
- 局限：来自同一数据来源，时间隔离不能证明跨果园泛化；与贵州蜂糖李的品种、光照、病害任务均有差异。
- 原始日间包 SHA256：`574381B28945C2B9746151D1DDA5A6F1961251BACD78CE15644C7C3CA007F311`。

### 模型与训练参数

配置文件：[acfr_plum_resnet50_fpn_v2.yaml](../python/configs/acfr_plum_resnet50_fpn_v2.yaml)。

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

### 逐轮训练记录

以下数值来自原始 [metrics.jsonl](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/metrics.jsonl)。AP50:95 是项目的简化实现，未做 COCO crowd/area 等分组，**不能等同官方 COCO mAP**。

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

第 12 轮的简化 AP50:95 最高，因此 [best.pt](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/best.pt) 保存这一轮；[last.pt](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/last.pt) 是恢复训练用的最后断点，本次恰好同为第 12 轮。训练 loss 从 0.7061 降到 0.3075；中途验证指标有波动，最佳权重按验证指标而非 loss 选出。

### 独立测试

使用 `best.pt` 对完整测试集 **47 张 / 845 个真实框**做一次评估，无 `--max-images` 限制。原始 [report.json](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/report.json) 记录了样本数、类别、指标与对照图清单。

| 指标 | 结果 |
| --- | ---: |
| AP50 | **0.8155** |
| AP50:95（简化） | **0.4234** |
| `plum` 类 AP50 | **0.8155** |

测试 AP50 比最佳验证 AP50 低约 0.041，简化 AP50:95 低约 0.071。测试集仍来自同一来源的相邻时段，结果只能说明此数据来源上的果实定位能力。原始测试对照图阈值为 0.4；**阈值只控制图上显示哪些预测框，不改变 AP 计算**。抽查 [sample_001.jpg](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_001.jpg) 和 [sample_008.jpg](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_008.jpg)：多数果实位置对齐，密集枝叶图可见额外预测框。

![第 1 张测试图：绿色真实框与红色预测框](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/sample_001.jpg)

### 日志和复现

- 训练事件：[train.log](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/train.log)
- 逐轮完整指标：[metrics.jsonl](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/metrics.jsonl)
- 批次与逐轮 CSV：[csv/version_0/metrics.csv](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/csv/version_0/metrics.csv)
- TensorBoard 曲线：[tensorboard/](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tensorboard/)
- 测试日志：[tests/final/test.log](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/final/test.log)
- 实际运行配置：[config.yaml](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/config.yaml)；[环境信息](../python/runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/environment.json)

```powershell
cd python
python test.py --checkpoint runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/best.pt --split test --samples 8 --output runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tests/recheck
python -m tensorboard.main --logdir runs/acfr_plum_resnet50_fpn_v2/20261017_223048_435584/tensorboard
```

本实验可用作果实定位基线；要评估蜂糖李病害，必须再有经确认的蜂糖李病叶、病果或病斑位置标注与独立果园测试集。

---

## 2026-10-21：YOLO11n 从训练到 Android 真机

### 目标与范围

这一版先验证“公开李子果实数据 → 训练 → ONNX 导出 → 手机摄像头实时画框”的完整链路。检测类别只有 `plum`；它不是蜂糖李或青脆李专用模型，也没有病害类别。当前任务是果实检测原型，病害框和病名仍需要相应数据训练。

### 今天完成

1. 在 `python/mobile/` 加入 ACFR COCO 标注转 YOLO、YOLO11n 训练与 ONNX 导出、PyTorch/ONNX 单图结果核对脚本。保留原有手写 PyTorch 检测实验，两条流程各有用途。
2. 使用 ACFR 日间李子数据训练 30 轮。训练集 245 张、验证集 48 张、测试集 47 张；导出的 FP32 ONNX 约 10.6 MB，输入为 RGB、NCHW、640×640，输出为未做 NMS 的单类检测结果。在 47 张同源测试图上，ONNX AP50 为 0.752、AP50:95 为 0.386；单图 PyTorch/ONNX 结果核对通过。这些数值不能说明跨果园或非李子场景表现。
3. 建立 `android/` Kotlin 工程：CameraX 获取后置摄像头画面，Kotlin 做旋转、裁剪、缩放、归一化、结果解析和 NMS；ONNX Runtime Android 在手机 CPU 上执行 `best.onnx`；叠加框和类别分数，按钮可定格画面。
4. 在 `D:\Dev\Android` 集中安装 Android Studio、SDK、Java 17 和 Gradle 缓存；`assembleDebug` 构建成功，生成 `android/app/build/outputs/apk/debug/app-debug.apk`。Android Studio 首次同步还会下载供阅读和调试的 `-sources.jar`，与命令行编译所需的依赖不同。
5. 用户已在真机启动并看到实时画框。主观反馈为运行流畅、没有明显卡顿；也观察到大量非李子物体被标成 `plum`。尚未量化 FPS、延迟、误检率，也没有完成框坐标精度或连续运行验证。

### 技术调用栈

| 阶段 | 实际使用 | 执行位置 |
| --- | --- | --- |
| 数据准备与训练 | Python、Ultralytics YOLO11n、PyTorch | Windows 电脑 |
| 模型交付 | `best.onnx` 权重和计算图、`best.json` 类别与阈值 | 电脑导出，随后打包进 APK assets |
| Android 构建 | Android Studio、Gradle、Android SDK 35、Java 17 | Windows 电脑 |
| 相机与界面 | Kotlin、CameraX、Android View | Android 手机 |
| 模型推理 | Kotlin 调用 ONNX Runtime Android 的原生库，当前使用 CPU | Android 手机 |
| 前后处理 | Kotlin 图像变换、RGB 张量、置信度筛选、NMS、坐标映射与画框 | Android 手机 |

ONNX 是模型格式，不是解释器。手机不运行训练脚本、PyTorch 或 Python 解释器；APK 中是 ONNX 模型和 ONNX Runtime 的 Android 库。ONNX Runtime 底层有 C/C++ 原生实现，所以模型可以在手机上运行，但项目目前没有自行编写 Android NDK/JNI/C++ 推理代码。仓库里的 `cpp/` 是独立的桌面 OpenCV 学习代码，不参与本次 APK 构建。日后若测出明确的端侧性能瓶颈，再评估是否需要 C++。

### 当前问题与下次行动

- `plum` 误检很多：保存若干真机画面及框分数，先核对手机与桌面的同图输入、颜色通道、旋转/裁剪、输出解析、NMS 和坐标映射，再比较不同阈值；此时不能断定是模型还是端侧实现单方面造成的。
- 收集蜂糖李、青脆李及普通非李子场景做独立测试。ACFR 图片来自同一果园，现有测试集无法代表这些场景。
- 根据实测结果决定补充负样本或目标果园标注并重训。病害识别需另备病害类别及位置标注，不能通过重命名当前 `plum` 类别获得。
- 记录手机型号、单帧耗时或 FPS、长时间运行情况后再判断是否需要加速；目前只有“感觉流畅”的反馈。

相关操作说明：[Android 工程](../android/README.md)、[Python 训练](../python/README.md)、[数据调研](dataset-research.md)。

### 文档与目录重构

按语言收拢工程：根目录的 CMake 入口移除，CMake 预设和 vcpkg 清单移入 `cpp/`，C++ 独立配置/编译成功；旧根目录 `out/` 构建产物已清理。根 README 只保留项目定位，每个语言目录只保留一个职责简介 README；功能模块、数据流和构建命令放在 `docs/cpp.md`、`docs/python.md`、`docs/android.md`。2026-10-12 框架验证与 2026-10-17 ACFR 实验记录并入本日志，删除原分散日志；数据来源调研保留为独立参考资料。

为 Python 模块和函数、Kotlin 类和函数、C++ 入口补充文件/函数说明。PlantDoc 下载器不再在 Python 数据目录写入上游 README，缺标注错误提示改指向集中模块文档。验证：Python `compileall` 与 8 个单元测试通过；`cpp/` 的 CMake/Ninja/vcpkg 构建成功；Android `assembleDebug` 构建成功。文档本地链接核对通过。未因整理文档重新做真机识别质量测试。
