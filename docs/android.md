# Android 模块与数据流

`android/` 是独立的 Kotlin/Gradle 应用，最低 Android API 26。当前 assets 约定为 `best.onnx` 与 `best.json`；模型只有 `plum` 果实类别，没有病害类别。

| 模块 | 职责 |
| --- | --- |
| `MainActivity.kt` | 请求相机权限，连接 CameraX 预览与分析流，管理定格按钮和状态文字 |
| `PlumDetector.kt` | 读取模型和元数据，图像缩放/填充/RGB 张量，调用 ONNX Runtime，解析原始 YOLO 输出并做 NMS |
| `DetectionOverlay.kt` | 把检测框从分析图像坐标映射到预览区域并绘制类别、分数 |
| `app/build.gradle.kts` | Android SDK、Kotlin、CameraX、ONNX Runtime 依赖和 APK 配置 |

数据流：`后置摄像头 → CameraX RGBA 帧 → 旋转和裁剪 → 640×640 RGB/NCHW 张量 → ONNX Runtime CPU 推理 → 置信度筛选与 NMS → 预览坐标映射 → 画框`。按钮冻结当前画面与框，再按恢复预览。输入约定和类别/阈值由 `best.json` 描述。

ONNX 是模型文件格式。电脑上的 Python、PyTorch 和 Ultralytics 负责训练和导出；APK 不包含 Python 解释器。Android 端由 Kotlin 调用 ONNX Runtime 自带的原生推理库，其底层含 C/C++ 实现；本工程没有自行编写 JNI/C++ 推理模块。

在 `android/` 目录运行 `./gradlew.bat assembleDebug`，APK 输出到 `app/build/outputs/apk/debug/app-debug.apk`。用 Android Studio 打开 `android/` 本身。当前本机工具链集中在 `D:\Dev\Android`：`studio/`、`sdk/`、`jdk/`、`gradle/`、`config/`。本机 SDK 路径在不提交的 `local.properties`，Gradle JDK 在 `D:\Dev\Android\gradle\gradle.properties`。模型 assets 不提交到 Git，构建前需由 Python 导出流程复制进来。

真机效果与未解决问题统一记录在 [开发日志](Logs.md)。
