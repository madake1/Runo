# Runo Android：李子实时检测

首个部署任务是**离线框出果园里的李子果实**，验证训练、ONNX 导出、相机取帧、端侧推理、坐标映射和拍照定格。当前模型不识别病害，也不估计整树产量。

## 模型准备

在 `python/` 下执行：

```powershell
python -m pip install -r mobile/requirements.txt
python -m datasets.acfr_plum.download
python -m datasets.acfr_plum.prepare
python mobile/prepare_yolo.py
python mobile/train_export.py --epochs 30
python mobile/verify_export.py
Copy-Item runs/mobile_plum/train/weights/best.onnx ../android/app/src/main/assets/best.onnx
Copy-Item runs/mobile_plum/train/weights/best.json ../android/app/src/main/assets/best.json
```

`prepare_yolo.py` 将已有 ACFR COCO 标注转为 YOLO 格式。`train_export.py` 使用 YOLO11n 预训练权重训练，导出固定 640×640 的 FP32 ONNX，另写入类别、阈值和预处理约定。首次运行会下载预训练权重；导出需要 `onnx` 包。模型和图片数据都不进入 Git。

本机本次训练 30 轮后，47 张测试图上的 ONNX 检测 AP50 为 0.752、AP50:95 为 0.386；最佳 `.pt` 约 5.5 MB，ONNX 约 10.6 MB。`verify_export.py` 检查输入/输出张量，并将单图 ONNX 框与 PyTorch 框对齐。

## Android 工程

用 Android Studio 打开 `android/`，安装 SDK 35 和 JDK 17，同步 Gradle 后在 Android 8.0 及以上的真机运行 `app`。工程使用 CameraX 后置摄像头和 ONNX Runtime CPU。相机帧按方向转正，等比例缩放并灰色填充到 640×640，按 RGB、NCHW、0–1 输入；应用解析未内置 NMS 的 YOLO11 输出，执行置信度筛选和 NMS，再把框映射回预览画面。点击“拍照定格”冻结当前画面与框，再点击继续预览。

工程包含 Gradle wrapper；也可从 `android/` 运行 `./gradlew.bat assembleDebug`。若 assets 缺少模型，应用会显示明确错误而不会把随机框当作识别结果。

本机已将 Android 开发环境集中在 `D:\Dev\Android`：`studio/` 是 Android Studio，`sdk/` 是 SDK，`jdk/` 是 Java 17，`gradle/` 存放 Gradle 下载与缓存，`config/` 存放 Studio 配置。启动程序位于 `D:\Dev\Android\studio\bin\studio64.exe`；用它打开本目录即可。SDK 路径写在本机的 `local.properties` 中，Gradle JDK 路径写在 `D:\Dev\Android\gradle\gradle.properties` 中。已通过 `assembleDebug` 编译，APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。

## 2026-10-21 进展与真机反馈

- 使用公开 ACFR 李子果实数据训练 YOLO11n 30 轮，导出约 10.6 MB 的 FP32 ONNX；在 47 张测试图上得到 ONNX AP50 0.752、AP50:95 0.386。这些离线指标不代表目标果园的实测效果。
- 完成 `CameraX → Kotlin 图像预处理 → ONNX Runtime Android → Kotlin 解析输出/NMS → 画框`，支持点击按钮定格当前画面。模型和元数据作为 `best.onnx`、`best.json` 放在 App 的 assets 内，离线运行。
- 安装 Android Studio、SDK 35、Build Tools、Platform Tools、Java 17 和 Gradle；命令行 `assembleDebug` 构建成功，Android Studio 同步并在真机运行成功。Studio 额外下载的 `-sources.jar` 是查看和调试依赖源码用的，不是手机运行时需要的 Python 包。
- 真机初测反馈：画面流畅，未观察到明显卡顿；但许多非李子物体也被画框并标为 `plum`。这版只能证明端到端运行，不能作为可靠的果实识别或病害识别。当前只有 `plum` 一个类别，阈值为 0.35；误检原因尚未定位，可能涉及数据分布、阈值或端侧预处理，需要用实际误检画面逐项核对。

### 为什么手机上不用运行 Python 或自行编写 C++

Python、PyTorch 和 Ultralytics 只在电脑上负责训练与导出。ONNX 保存模型结构和权重，本身不执行推理。Android APK 内置 ONNX Runtime Android 的原生推理库，由 Kotlin API 调用它执行 `best.onnx`；相机取帧、预处理、后处理、画框由 Kotlin 完成。手机运行时不启动 Python，也不打包 Python 解释器。底层推理库包含 C/C++ 原生实现，但本项目当前没有自己编写 JNI/C++ 推理代码。若以后测速发现具体瓶颈，再决定是否引入 NDK/C++。

下一步先保存真机误检截图和对应置信度，核对相机图像通道、旋转/裁剪、ONNX 输出解析与桌面结果是否一致；再用蜂糖李、青脆李和非李子场景建立小型真实测试集，衡量误检并补充训练数据。病害任务要另行收集有病害类别和位置标注的样本。

## 验收边界

真机已证明 APK、模型加载、预览与画框可运行，流畅度仅有主观反馈，尚无单帧耗时或连续运行数据；框位置与误检仍需验证。现有 ACFR 训练图来自同一果园，模型对青脆李、蜂糖李及不同环境的泛化需要另测。后续更换病害模型时，若保持相同的 ONNX 输入、输出和 `best.json` 协议，Android 取帧与画框代码可复用；病害训练仍需病害位置标注。
