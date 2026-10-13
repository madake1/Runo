# 蜂糖李数据调研（2026-10-12）

目标：贵州蜂糖李果实与叶片病害定位，支持近照和复杂背景，未来扩展视频。

本次未找到已核实可直接下载、带蜂糖李病害框、规模足够训练的公开数据。不能把名字含 Plum 的数据直接当成国内李子。

| 数据 | 核实结果 | 决定 |
| --- | --- | --- |
| PlantDoc Object Detection | 官方 CSV 框、多作物叶片、CC BY 4.0 | 下载用于学习检测流程，不当作蜂糖李数据 |
| PlantVillage | 通用叶片分类，缺少蜂糖李位置标注 | 仅保留调研清单 |
| Plum leaf and fruit (w7sdx55m7z) | 论文明确 Ziziphus mauritiana（印度枣）；原图标签与加工版本病害名也有差异 | 排除 |
| Java Plum / Hog Plum | 其他植物；Hog Plum 文献明确 Spondias mombin | 排除 |
| PlantSeg | 病斑分割，论文表中 plum bacterial spot 仅 10 张 | 可后续核验物种，覆盖不足，不下载约 3GB 全量 |
| 李子红点病两阶段论文 | 研究李子病害，本次未找到已验证的公开原始下载入口 | 可联系作者，不声称已取得数据 |

## 原始来源

- PlantDoc：https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
- PlantDoc 论文：https://arxiv.org/abs/1911.10317
- PlantVillage：https://github.com/spMohanty/PlantVillage-Dataset
- 印度枣数据论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12145562/
- Hog Plum 论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC11808602/
- Java Plum 论文：https://www.sciencedirect.com/science/article/pii/S2772375524001059
- PlantSeg：https://github.com/tqwei05/PlantSeg
- PlantSeg 论文：https://www.nature.com/articles/s41597-025-06513-4
- 李子红点病论文：https://www.mdpi.com/2073-4395/14/7/1589

蜂糖李需要自采或向机构获取对应数据。先统一框病叶/病果或框病斑，覆盖不同树、日期、距离、健康背景和遮挡，保留来源组并请专家确认类别。

SSDLite 是轻量检测学习起点，320 输入可能丢失远景小病斑。真实能力必须用真实蜂糖李测试集衡量，无法承诺任意照片中找出所有病害。
