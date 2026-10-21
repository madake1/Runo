# Runo

Runo 是一个面向果园场景的离线视觉应用项目。希望用户拿起普通 Android 手机，就能在实时画面中看到识别目标的位置；定格画面后，还能进一步了解识别结果。项目会逐步探索果实检测与病害识别。

这也是一个贯穿数据整理、模型训练、模型导出和端侧推理的学习项目。识别能力需要在真实果园中检验；果实检测与病害识别是不同任务，需要分别准备数据和验证效果。

## 项目目录

| 目录 | 用途 |
| --- | --- |
| [`android/`](android/README.md) | Android 应用：相机画面、端侧推理与结果展示。 |
| [`cpp/`](cpp/README.md) | C++ 桌面视觉工程，用于学习 OpenCV 和原生程序开发。当前 Android 应用通过 ONNX Runtime 运行模型，因此 C++ 不是手机部署的前置条件。 |
| [`docs/`](docs/) | 各语言模块说明+数据流、数据与模型调研、开发日志。 |
| [`python/`](python/README.md) | 数据准备、模型训练、评估、预测和导出，是模型实验的主要工作区。训练得到的模型可交给 Android 应用使用。 |

## 根目录配置文件

### `.editorconfig`

供支持 EditorConfig 的编辑器读取。当前配置指定 UTF-8 编码；默认使用 CRLF 换行，Python 文件以及 `.gitignore`、`.gitattributes`、`.editorconfig` 使用 LF。这样在 VS Code、Android Studio 等编辑器之间切换时，文件编码和换行规则更明确。

### `.gitattributes`

供 Git 管理文件的文本与换行方式。当前配置自动识别文本文件，并为 Markdown、C++、JSON 等文件指定 CRLF，为 Python 文件以及上述三个点文件指定 LF；`.pt` 权重文件按二进制处理。
