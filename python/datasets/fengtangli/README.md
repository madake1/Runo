# 蜂糖李提取方案

第一版使用矩形检测，定位果实/叶片病害。如果要求病斑轮廓和面积，需要分割任务。

```text
data/fengtangli/
├── raw/orchard_a/day_01/tree_01/
│   ├── image_001.jpg
│   └── image_001.json       # LabelMe
├── metadata/
│   ├── classes.json
│   └── groups.json
└── annotations/            # 自动生成
```

先请专家确认类别，并统一“框病叶/病果”或“框病斑”的粒度。标注所有可见目标，避免同图遗漏。LabelMe 使用 rectangle，`imagePath` 指向原图；健康照片也保留 JSON，`shapes` 为空，作为负样本。

`classes.json` 是有序的 JSON 字符串列表，内容为实际确认的类别。`groups.json` 将相对 raw 的图片路径映射到来源组，例如：

```json
{
  "orchard_a/day_01/tree_01/image_001.jpg": "orchard_a_tree_01"
}
```

同一树、连拍或视频的帧不能跨划分；更严格评估可按整个果园分组，但需要足够独立果园。

```powershell
python -m datasets.fengtangli.prepare --groups data/fengtangli/metadata/groups.json --classes data/fengtangli/metadata/classes.json
```

验证图片、标签、坐标和来源组。通过连通分组绑定来源组与像素重复图片，按组近似分成 70/15/15。某划分缺类别时明确报错，要求补数据或调整种子；不保证自动类别平衡。

保留原图，输出 COCO。标注后不要改变图片方向和尺寸；没有标注 JSON 的图片不进入训练。

后续视频先抽帧标注，同一视频使用同组。推理预留 `Detector.predict_frames()`，解码、抽帧、追踪和 Android 按钮交给上层。
