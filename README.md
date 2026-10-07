# Runo

> **Let me run!**

Runo 是一个用于学习 **C++、计算机视觉、模型推理与端侧 AI 部署** 的长期个人项目。

项目以农业视觉为主要应用场景，从最基础的摄像头与目标检测开始，逐步完成从 **模型训练 → 模型导出 → C++ 推理 → 端侧优化 → Android 部署 → 多模态视觉应用** 的完整 AI 工程链路。

Runo 的目标不是一开始就构建一个完整的农业产品，而是在真实、可落地的应用场景中逐步学习和验证计算机视觉与 AI 工程技术，最终做出一个能够在普通 Android 手机上运行的离线视觉应用。

未来条件允许时，将进一步使用真实果园数据进行验证和迭代。

---

## 🎯 Project Goals

Runo 将逐步探索：

- 使用 C++ / OpenCV 构建实时视觉程序
- 使用 PyTorch / YOLO 完成视觉模型实验
- 将模型导出为 ONNX 等通用模型格式
- 使用 C++ 推理引擎完成模型推理
- 理解图像预处理、Tensor、模型输出与 NMS 等完整推理流程
- 探索 FP16 / INT8 等模型轻量化与端侧优化技术
- 使用 Android NDK / JNI 将 C++ 推理模块部署到移动端
- 在普通 Android 设备上实现完全离线的视觉识别
- 针对果实、叶片病害等农业场景进行模型训练与微调
- 探索 VLM 等多模态模型在端侧视觉应用中的使用方式

---

## 仓库结构与启动

```text
Runo/
├── cpp/                       # C++ 桌面验证，未来提取可复用推理库
│   ├── CMakeLists.txt
│   ├── src/main.cpp           # 摄像头入口
│   └── examples/              # 早期学习代码，不参与默认构建
├── python/                    # Python 模型实验，使用独立环境
│   ├── camera_demo.py
│   ├── detector.py
│   ├── display.py
│   ├── requirements.txt
│   └── models/                # 当前实验使用的 .pt 权重
├── android/                   # 最终 App 的预留位置，目前仅有说明
├── docs/                      # 学习资料和历史提示
├── CMakeLists.txt             # VS 桌面构建入口，委托 cpp/
├── CMakePresets.json          # Windows 桌面配置
└── vcpkg.json                 # 桌面 C++ 的 OpenCV 依赖
```

`out/` 为本地构建产物，不属于源代码。C++、Python 与 Android 分别管理自己的依赖；Python 不参与桌面编译，也不会作为运行环境打包进 Android App。

### C++ / Visual Studio

从 Visual Studio 打开仓库根目录，选择 **Windows x64** 配置，等待 CMake 配置完成，再生成并运行 **Runo.exe**。按 Esc 退出摄像头窗口。

当前预设中的 vcpkg 路径为 `D:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake`；其他电脑需在 `CMakePresets.json` 中调整该路径，并安装 MSVC、CMake 和 Ninja。

也可以在已加载 x64 MSVC 环境的开发者终端中，从仓库根目录运行：

```powershell
cmake --preset windows-x64
cmake --build out/build/windows-x64
./out/build/windows-x64/Runo.exe
```

CMake 配置成功不代表已生成可执行文件；构建后程序位于 `out/build/windows-x64/Runo.exe`。旧的 `out/build/x64-Debug/` 不再是当前预设的输出目录。

### Python

环境创建、依赖安装和摄像头启动命令见 [Python 说明](python/README.md)。现有入口使用 YOLO11n 的 COCO 预训练权重，尚未进行农业数据训练；目录中的 YOLO-World 权重未被当前入口使用。

### 最终 Android App

[Android 目录](android/README.md) 当前只是规划占位，没有 Gradle 工程或 APK。后续以 Kotlin 实现界面和摄像头输入，通过 JNI 调用 C++ 推理库。桌面窗口与摄像头代码留在桌面端，模型实验通过导出模型与运行时对接。

下面的技术栈与路线图包含未来计划，不代表仓库已实现。当前未包含模型训练、ONNX 导出、C++ 模型推理、量化或移动端代码。`docs/initial-learning-prompt.md` 是历史资料。

---

## 🛠 Tech Stack（当前 + 规划）

**Model Development**

Python · PyTorch · YOLO · OpenCV

↓  

**Model Deployment**

ONNX · C++ · OpenCV · Inference Runtime

↓  

**Edge Optimization**

FP16 / INT8 · Model Quantization · Performance Profiling

↓  

**Mobile Deployment**

Android · Kotlin · NDK · JNI

↓  

**Multimodal Exploration**

VLM · Vision-Language Interaction · Local AI

---

## 🗺 Roadmap

### v0.1 — Hello Vision

让 Runo 睁开眼睛。

- [x] 搭建 C++ 项目
- [x] 配置 CMake / OpenCV
- [x] USB 摄像头读取
- [x] 实时显示摄像头画面
- [ ] C++ 基础 FPS 统计（当前仅 Python 摄像头入口实现）

---

### v0.2 — Hello YOLO

让 Runo 第一次看懂东西。

- [x] 提供 Python 实验代码及依赖清单（环境需本机安装）
- [x] 接入 YOLO
- [ ] 独立图片目标检测入口（检测器已支持图像帧输入）
- [x] 摄像头实时目标检测
- [x] 绘制类别、置信度与检测框
- [x] 整理 Python 模型实验代码结构（摄像头 / 检测器 / 绘制分离）

---

### v0.3 — C++ Inference

让模型离开 Python。

- [ ] 导出 YOLO 模型至 ONNX
- [ ] C++ 加载模型
- [ ] OpenCV 图像预处理
- [ ] C++ 模型推理
- [ ] 解析模型输出
- [ ] NMS 后处理
- [ ] 绘制检测结果
- [ ] C++ 摄像头实时检测

---

### v0.4 — Faster Runo

让 Runo 跑得更快。

- [ ] 统计模型推理延迟与 FPS
- [ ] 分析预处理 / 推理 / 后处理耗时
- [ ] 优化实时视觉 Pipeline
- [ ] 探索多线程推理
- [ ] 尝试 FP16 推理
- [ ] 尝试 INT8 量化
- [ ] 比较模型大小、速度与精度

---

### v0.5 — Runo Orchard

让 Runo 开始认识真正的果园。

- [ ] 调研并整理公开农业视觉数据集
- [ ] 构建小规模农业目标检测数据集
- [ ] 果实检测
- [ ] 果实状态 / 品质识别
- [ ] 叶片病害识别
- [ ] 模型训练与微调
- [ ] 模型评估与数据迭代
- [ ] 条件允许时采集真实果园数据

---

### v0.7 — Runo Mobile

把 Runo 装进口袋。

- [ ] 创建 Android 应用
- [ ] 图片选择与显示
- [ ] Android NDK / JNI 接入
- [ ] C++ 推理模块移植
- [ ] Android 端图片检测
- [ ] CameraX 摄像头接入
- [ ] 手机实时目标检测
- [ ] 移动端性能优化
- [ ] 完全离线运行

---

### v1.0 — Runo

做一个真正可以交给家人使用的版本。

- [ ] 简化移动端操作流程
- [ ] 拍照 / 相册 / 实时识别
- [ ] 清晰展示检测结果
- [ ] 模型与应用完全离线运行
- [ ] 异常处理与基础稳定性优化
- [ ] APK 打包与真实设备测试

---

### v1.x — Multimodal Runo

让 Runo 不只是“看见”，还能够“理解”。

- [ ] 探索轻量 VLM
- [ ] 图像 + 自然语言交互
- [ ] 检测模型与 VLM 联动
- [ ] 基于检测区域进行进一步视觉分析
- [ ] 探索农业知识与视觉结果结合
- [ ] 探索端侧多模态模型部署

---

## 🌳 Future

Runo 当前主要是一个 **AI Engineering Playground**。

如果未来能够获得稳定的真实果园数据和使用条件，将进一步探索：

- 固定摄像头长期监测
- 果实数量与状态统计
- 多时段视觉变化分析
- 病害异常提醒
- 果园视觉数据管理

更远期也可以探索机器人、视觉导航与具身智能，但这些并不是当前阶段的开发目标。

---

## 📍 Current Status

**当前阶段：v0.2 — Hello YOLO（实验阶段，并非正式发布版本）**

目前已经完成：

**C++**

`Camera → OpenCV → Real-time Display`

**Python**

`Camera → YOLO11n (COCO) → Detection → Visualization + FPS`

下一阶段：

**让 YOLO 离开 Python。**

`YOLO → ONNX → C++ → OpenCV → Real-time Inference`

---

> **Let me run!**
>
> Build it. Understand it. Make it run.