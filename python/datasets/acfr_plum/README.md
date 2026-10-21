# ACFR 果园李子检测数据

来源：[University of Sydney ACFR Plum Harvesting Detection Dataset](https://data.acfr.usyd.edu.au/Agriculture/PlumDetection/)。官方提供 Pascal VOC 检测框，日间和夜间各 350 张；项目只使用 RGB 图片和框，忽略深度图。标注目标是果园里的李子果实，**没有病害或蜂糖李类别**。数据页面没有清晰列出再分发许可，原始包保留在本机 `data/`，不要将图片提交到 Git。

在 `python/` 下运行：

```powershell
python -m datasets.acfr_plum.download
python -m datasets.acfr_plum.prepare
python train.py --dataset acfr_plum --model fasterrcnn_resnet50_fpn_v2
```

默认下载日间包（78 MB）；若需夜间包，可运行 `python -m datasets.acfr_plum.download --sets Day Night` 后重新执行 `prepare`。下载脚本只提取 JPEG、VOC XML 与划分清单，数据转成 `train.json`、`val.json`、`test.json`。当前已准备日间包：245/48/47 张。

官方 train/val/test 清单按相邻帧交错分配，容易让高度相似帧跨划分。这里改为按帧序号连续切段，划分边界各舍弃 5 帧，并去除跨划分的像素重复图。所有图像仍可能来自同一果园，这个测试不能证明跨果园泛化。训练出的单类模型只能定位李子；蜂糖李病害识别仍需真实病叶/病果标注。
