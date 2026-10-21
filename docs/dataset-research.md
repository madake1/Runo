# 数据集与模型调研

更新：2026-10-21。依据 [Runo 的产品定位](../README.md)，本项目先让普通 Android 手机离线、实时地画出目标位置，再逐步支持果实和病害识别。数据集是否能训练**果实框**、**病害区域**或**病名**，需要分别判断；名称含 “plum” 也不能直接推断它适用于蜂糖李、青脆李。

## 选型结论

1. **眼下继续用 YOLO11n 检测模型作为可复现基线。** 它已完成训练、ONNX 导出和 Android 运行；先查清大量 `plum` 误检来自模型泛化、输入处理还是输出解析。当前本机实验和真机反馈见 [开发日志](Logs.md)。
2. **果实数据先用 ACFR 练链路，自采绿色李子数据验证和改进。** ACFR 的官方数据是红李果实框，日/夜各 350 张，没有病害类别；它与家中绿色李子的颜色和拍摄场景不同。[ACFR 原始数据页](https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/)
3. **病害任务以自采和专家核对的标注为主。** PlantDoc 可练多作物叶片检测，PlantSeg 可研究病斑分割，但都不能直接充当蜂糖李病名训练集。现有已核验资料中，没有发现可直接下载、规模足够且含蜂糖李病害位置与可靠病名的公开数据。[PlantDoc 官方仓库](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)、[PlantSeg 官方仓库](https://github.com/tqwei05/PlantSeg)
4. **YOLO26n 作为同数据对照，不直接替换线上模型。** 官方支持 ONNX 和可选的无 NMS 输出，但该输出协议与当前 Android 解析器不同；必须重新核对输出、精度、误检和真机速度。[YOLO26 官方说明](https://docs.ultralytics.com/models/yolo26)、[输出模式说明](https://docs.ultralytics.com/guides/end2end-detection)

## 数据集清单

| 候选 | 标注与获取情况 | 对 Runo 的用途 | 决定 |
| --- | --- | --- | --- |
| [ACFR Plum Harvesting Detection](https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/) | 官方可下载；日/夜各 350 张，Pascal VOC 红李果实框，分别有 4,449 / 1,402 个实例；无病害标签 | 果实检测与端侧链路基线 | **已使用**；不作为绿色李子或病害效果证明 |
| 自家果园的蜂糖李、青脆李与非李子画面 | 需要拍摄和标注；含健康果实、叶片、相似绿色物体、空画面及复杂光照 | 真实场景测试、误检分析、后续微调 | **最高优先级**；按果园/树/日期分组留出独立测试 |
| [PlantDoc Object Detection](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) | 官方仓库有图片和 CSV 框，项目许可为 CC BY 4.0；多作物、多病害，缺蜂糖李类别 | 学习病叶检测数据处理与训练 | **练习用**；不把其类别名直接展示为蜂糖李诊断 |
| [PlantSeg](https://github.com/tqwei05/PlantSeg) | 官方仓库指向可下载数据；11,400 多张野外病害图，115 类，带病害区域分割标注。论文列出若干 plum 类别，但各类仅约 10–42 张，品种及本地适用性待核验。[论文及类别表](https://www.nature.com/articles/s41597-025-06513-4) | 研究“框病斑还是分割病斑”、验证标签粒度 | **候选辅助数据**；不能单独训练可靠的本地病名模型 |
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | 官方库有 54,306 张健康/病叶图和图像级标签；没有目标框 | 分类基线与数据划分方法参考 | **暂不用于实时画框**；分类标签不能提供病害位置 |
| [四川古蔺 YOLOv7-Plum 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10420999/) | 论文报告 1,220 张图、8,745 个果实框；数据注明向通讯作者申请，无病害标注 | 地理与果园场景更接近的果实检测候选 | **可申请**；拿到许可和数据后再接入 |
| [青脆李 MAE-YOLOv8 研究](https://www.sciencedirect.com/science/article/pii/S0168169924008494) | 论文使用自建青脆李小目标检测数据；本次未核实到可直接下载的原始图片和标注包 | 参考遮挡、小目标和绿色果实问题 | **可联系作者**；不把论文结果当作现成数据集 |

这些来源的拍摄对象、标注粒度、许可和划分方式各不相同，不能混合后直接报告一个“蜂糖李病害准确率”。尤其 ACFR 原始划分有相邻帧风险，项目现有转换流程按连续帧分段；即便如此，测试仍来自同一数据来源，不能代表跨果园泛化。详见 [Python 模块说明](python.md) 和 [历史实验日志](Logs.md)。

### 自采数据的第一版约定

- **先定义标签**：果实框与病叶/病果框分开；若要画病斑边界，则需要病斑框或掩码。病名须由可靠资料或农技人员确认；不能确定的样本保留“待确认”，不硬塞到某一病类。
- **保留负样本**：空画面、叶子、枝条、其他绿色果实、强反光和模糊帧。当前大量误检使负样本与独立验证集尤其重要。
- **记录来源**：品种、果园、树或拍摄区域、日期、手机、距离、光照、器官、标注人；同树连拍或视频帧不要跨训练与测试。
- **先建小型验证集**：覆盖至少两种光照、多个拍摄距离、健康与非目标场景；先逐张检查标签质量，再扩充训练集。病害类别样本不足时只验证“疑似异常区域”，暂不承诺具体病名。

## 模型候选与推荐顺序

| 模型 | 适合的任务 | 与现有 Android 链路的关系 | 建议 |
| --- | --- | --- | --- |
| [YOLO11n Detect](https://docs.ultralytics.com/models/yolo11) | 实时果实框；有足量病害框后也可训练病害区域检测 | 当前使用的基线，ONNX 输入、输出和 Kotlin NMS 已接通 | **首先保留**；用真机误检样本校验和改进数据 |
| [YOLO26n Detect](https://docs.ultralytics.com/models/yolo26) | 同一数据上的轻量检测对照 | 可导出 ONNX；默认一对多输出与可选无 NMS 输出不同，Android 必须按导出协议适配 | **第二候选**；同数据、同划分、同手机对比后决定 |
| [YOLO11n-seg](https://docs.ultralytics.com/models/yolo11) | 需要精细标出病斑范围时的实例分割 | 需要掩码标注和新的 Android 掩码解析/绘制 | **后续阶段**；仅当框无法满足使用需求时引入 |
| [SSDLite320 MobileNetV3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) | 手写训练框架的轻量检测对照 | `python/` 已有训练路径，尚需独立验证导出与 Android 输出协议 | **学习基线**；小病斑可能受 320 输入限制 |

对手机部署，应在**同一真实测试集**上比较误检、漏检、框位置和设备性能，而不是拿不同论文的 mAP 或桌面 CPU 速度排序。ONNX Runtime 官方建议先在设备上测模型大小、延迟和功耗，再按结果考虑 XNNPACK、NNAPI 或量化；加速收益取决于手机与算子支持。[ONNX Runtime 移动端指南](https://onnxruntime.ai/docs/tutorials/mobile/)

模型输出是检测建议，不等于植物病理诊断。病名和处理资料要能追溯到资料来源；当模型分数不足、图像模糊或类别不在训练范围内，界面应显示“无法判断/建议补拍”，避免把猜测显示为确定结论。

### 发布前的许可核对

PlantDoc 仓库标注 CC BY 4.0；其他数据源应分别核对下载页和原作者的使用、再分发条件。Ultralytics 对其 YOLO 代码和模型提供 AGPL-3.0 与企业许可选项；若未来要把应用作为闭源产品分发，需要在发布前确认相应授权或更换可用的模型实现。[PlantDoc 许可](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)、[Ultralytics 许可说明](https://www.ultralytics.com/license)
