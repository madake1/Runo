# PlantDoc 提取方案

来源：[官方检测仓库](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)，CC BY 4.0。保留作者 README 与 LICENSE。固定 commit `4730a233a555b30ee98e0879c63ad25d82407455`。

在 `python/` 目录的 Conda Runo 环境执行：

```powershell
python -m datasets.plantdoc.download
python -m datasets.plantdoc.prepare
```

下载到 `data/plantdoc/raw/`，记录 SHA256，失败可重试。`--max-images 100` 可限制每个官方划分下载 100 张；默认全量。含 Windows 禁用字符的文件名映射为确定性哈希文件名，不改变上游下载 URL。

按图片聚合所有框，验证解码、按实际尺寸换算坐标、裁剪越界框，并记录缺失或无效样本。类别映射来自全部官方 CSV，前景 ID 从 1 开始。生成 COCO JSON，不复制图片。

保留官方 TEST，TRAIN 按固定种子拆训练/验证。像素重复图片归同组，TRAIN 中与 TEST 重复的图片移除。缺少树/拍摄批次元数据，像素去重不能消除所有近似重复，不能保证来源隔离。

官方标签通常标记整片叶子，不代表精细病斑位置。该数据用于学习流程，成绩不代表蜂糖李能力。健康叶类别作为官方目标保留；蜂糖李任务可使用无病害框的健康照片作负样本。
