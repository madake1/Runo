# Android 应用工程

本目录是独立的 Kotlin/Gradle Android 工程。应用通过 CameraX 读取相机画面，在设备上使用 ONNX Runtime 加载模型，并显示检测框与定格画面。`app/` 存放界面、推理代码和模型 assets；Gradle wrapper 用于构建。

构建方法、模块职责和相机到画框的数据流见 [Android 模块说明](../docs/android.md)。
