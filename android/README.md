# Android 应用（规划中）

最终交付物是可离线运行的 Android App。此目录预留给 Android Studio / Gradle 工程，目前没有可构建的应用。

未来 Kotlin 负责界面、权限与 CameraX 输入；NDK / JNI 连接可复用的 C++ 推理模块。桌面摄像头窗口代码不直接移植到手机，Python 实验环境也不打包进 App。

开始移动端阶段后，再引入 Gradle、Android SDK / NDK 和模型资源；当前桌面 CMake 构建不依赖这些工具。
