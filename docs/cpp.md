# C++ 模块与数据流

`cpp/` 是独立的桌面 OpenCV 工程，不参与 Android APK 构建。

| 文件 | 职责 |
| --- | --- |
| `src/main.cpp` | 打开默认摄像头，逐帧显示图像，Esc 退出 |
| `examples/opencv_version.cpp` | 保留 OpenCV 版本查询示例，不参加默认构建 |
| `CMakeLists.txt` | 查找 OpenCV，生成 C++17 的 `Runo` 可执行文件 |
| `CMakePresets.json` | Windows x64/Ninja 预设，指定 vcpkg 工具链和输出目录 |
| `vcpkg.json` | OpenCV 依赖清单 |

数据流：`摄像头 → cv::VideoCapture → cv::Mat → cv::imshow → 屏幕`。目前没有 C++ 模型推理。

在 `cpp/` 目录的 VS x64 开发者终端构建：

```powershell
cmake --preset windows-x64
cmake --build out/build/windows-x64
./out/build/windows-x64/Runo.exe
```

本机若 `cmake` 未加入 PATH，可使用 `D:/Dev/VS/visualstudio/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe` 执行上述两条命令。预设使用 `D:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake` 和本机 Visual Studio 自带的 Ninja；其他机器需修改 `CMakePresets.json` 中这两个绝对路径。构建需要 MSVC、CMake、Ninja 和通过 vcpkg 安装的 OpenCV。
