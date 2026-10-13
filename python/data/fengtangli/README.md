# 贵州蜂糖李自采数据

预留给真实蜂糖李果实和叶片数据，目前没有真实样本。

- `raw/`：原始图片及同名 LabelMe 矩形标注，可按果园/日期/树编号组织。
- `metadata/`：人工整理的 `classes.json` 和 `groups.json`。
- `annotations/`：生成的 COCO 标注与划分报告。

方案见 `../../datasets/fengtangli/README.md`。病斑框、整片病叶框与整颗病果框是不同粒度，标注前必须统一定义。
