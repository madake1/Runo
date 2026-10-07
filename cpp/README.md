# C++ 桌面验证

`src/main.cpp` 使用 OpenCV 打开摄像头 0 并显示画面，按 Esc 退出。目前没有 FPS 统计或模型推理。

从仓库根目录用 Visual Studio 打开项目，选择 `Windows x64` 和 `Runo.exe`。根 CMake 文件委托本目录构建，输出仍在 `out/build/windows-x64/Runo.exe`。

`examples/opencv_version.cpp` 保存早期 OpenCV 版本检查代码（当前已注释），不参与构建。

未来完成 C++ 推理后，再把与平台无关的推理逻辑提取成库，由桌面程序和 Android JNI 共同调用。当前不提前创建空模块。
