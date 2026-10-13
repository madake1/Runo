# 框架检查

运行 `python -m unittest discover -s tests -v`。验证来源组隔离、Windows 文件名、COCO 框坐标、翻转、健康负样本、蜂糖李提取器和检测 AP。

训练集成检查见上级 README 的冒烟命令。少量批次只验证流程，不能用作精度结论；正式训练使用独立实验目录。
