# 实验产物

每次实验使用独立子目录，保存实际配置、类别映射、训练日志、评估指标、最佳权重和最新 checkpoint。

例如：

```text
runs/
└── mobilenet_v3_small_001/
    ├── config.yaml
    ├── classes.json
    ├── metrics.csv
    ├── best.pt
    └── last.pt
```

需要恢复训练时，checkpoint 应包含模型、优化器、调度器（如有）和训练进度等状态。

实验产物默认不提交到 Git，本目录的 README 保留在版本管理中。
