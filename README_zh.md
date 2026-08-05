<!-- Modified by RISCY-SVT in 2026 to identify this derivative release. -->
> **Unofficial RISCY-SVT build.** This derivative is based on XSlim commit
> `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` and is not endorsed by SpacemiT.
> Provenance and changes are recorded in [UPSTREAM.md](UPSTREAM.md) and
> [MODIFICATIONS.md](MODIFICATIONS.md).

# XSlim

中文 | [English](README.md)

[![版本](https://img.shields.io/badge/版本-2.1.2%2Briscy.1-blue.svg)](https://github.com/RISCY-SVT/xslim/releases)
[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/)

**XSlim** 是 [SpacemiT](https://www.spacemit.com) 推出的离线（Post-Training）量化工具，集成了已经调整好的适配芯片的量化策略，使用 JSON 配置文件调用统一接口实现 ONNX 模型量化。

---

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [文档](#文档)
- [示例](#示例)
- [更新日志](#更新日志)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 特性

- **INT8 / FP16 / 动态量化** — 多种精度级别满足不同部署场景
- **JSON 驱动配置** — 简洁的声明式量化设定
- **Python API 与命令行** — 可作为库调用或通过命令行使用
- **自定义预处理** — 支持自定义预处理函数
- **更完整的 ONNX 算子覆盖** — 支持在包含常见算术、激活、比较、规约、Dropout 以及 opset-24 `Pad` 形态的模型上运行 Graphwise Analysis 与量化
- **自动 YOLO Decode 融合** — 将受支持的 YOLO 解码后处理子图融合为单个 `spacemit_functions.YoloDecode` 节点
- **感知 ONNX Function 的导出链路** — 自动保留内嵌 `FunctionProto` 定义并补齐所需的自定义域导入
- **可选输出语义检查** — 在不向核心量化流程加入特定模型规则的前提下检测分数塌缩
- **Q/DQ 边界审计** — 报告可表示范围、饱和、偏差、MAE、余弦相似度、直方图和哈希
- **基于 ONNX** — 构建在 ONNX 生态系统之上

## 安装

RISCY-SVT 非官方版本不发布到 PyPI。请安装
`v2.1.2-riscy.1` GitHub Release 中附带的 wheel：

```bash
python -m pip install ./xslim-2.1.2+riscy.1-py3-none-any.whl
```

或从源码安装：

```bash
git clone https://github.com/RISCY-SVT/xslim.git
cd xslim
git switch riscy/k1x-yolo26-hardening-release-001
python -m pip install .
```

本地开发建议使用可编辑安装：

```bash
python -m pip install -e .
```

构建元数据统一定义在 `pyproject.toml` 中，导入包位于标准的 `src/` 布局下。如需在本地构建源码包和 wheel：

```bash
python -m pip install --upgrade build
python -m build
```

## 快速开始

### Python API

```python
import xslim

# 使用 JSON 配置文件
xslim.quantize_onnx_model("config.json")

# 使用字典
config = {
    "model_parameters": {
        "onnx_model": "model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parameters": [{
            "mean_value": [123.675, 116.28, 103.53],
            "std_value": [58.395, 57.12, 57.375],
            "color_format": "rgb",
            "preprocess_file": "PT_IMAGENET",
            "data_list_path": "./calib_img_list.txt"
        }]
    }
}
xslim.quantize_onnx_model(config)

# 也可以直接传入模型路径和输出路径
xslim.quantize_onnx_model("config.json", "input.onnx", "output.onnx")
```

### 命令行

```bash
# 安装后的标准命令入口
xslim --config config.json

# 模块入口仍然可用
python -m xslim --config config.json

# 指定输入和输出模型路径
xslim -c config.json -i input.onnx -o output.onnx

# 动态量化（无需配置文件）
xslim -i input.onnx -o output.onnx --dynq

# FP16 转换（无需配置文件）
xslim -i input.onnx -o output.onnx --fp16

# 将默认 ai.onnx opset 转换到指定版本
xslim -i input.onnx -o output.onnx --opset 20

# 仅模型精简（无需配置文件）
xslim -i input.onnx -o output.onnx
```

在无配置文件的动态量化和 FP16 转换流程中，可用逗号分隔的算子类型或名称排除指定算子：

```bash
xslim -i input.onnx -o output.onnx --dynq --ignore_op_types Softmax,LayerNormalization
xslim -i input.onnx -o output.onnx --fp16 --ignore_op_names /model/head/MatMul
```

静态 INT8 量化要求输入模型仍为浮点模型。若模型中已经包含 `QuantizeLinear` 或 `DequantizeLinear`，XSlim 会明确报错并停止，避免对已量化图再次量化。

对于受支持的 YOLO 导出模型，无需额外开关：XSlim 会在模型精简阶段尝试把 decode 密集的后处理融合成 `spacemit_functions.YoloDecode`，并在导出模型时保留对应的 ONNX `FunctionProto`。

Stage64 的 YOLO26 direct-E2E 诊断图未匹配该融合，量化后的分数通道发生塌缩。
本分支不宣称该路径已修复。当前验证通过的路径是显式六输出切分：bbox 与
confidence 分支保持分离，后处理保留浮点。参见
[K1X/YOLO26 脱敏示例](samples/k1x_yolo26_split/README.md)。

两个检查器均为显式启用的工具，不会无条件改变量化行为：

```bash
xslim-yolo-output-check --help
xslim-qdq-boundary-audit --help
```

## 文档

- [配置参考](doc/configuration_zh.md) — 所有 JSON 配置选项的完整说明
- [使用示例](doc/examples_zh.md) — INT8、FP16、动态量化、自定义预处理等场景的分步指南
- [精度调优指南](doc/accuracy_tuning_zh.md) — 如何诊断并提升量化后的模型精度

## 示例

- [K1X YOLO26 六输出切分](samples/k1x_yolo26_split/README.md) —— 脱敏配置、
  精确 letterbox 预处理、语义检查和 Q/DQ 边界审计

请查看 [samples](samples/) 目录，包含 ResNet-18、MobileNet V3、BERT 等模型的可运行示例。YOLO 专项用法请参考示例文档与精度调优指南。

## 更新日志

下游构建请查阅 [RISCY-SVT Releases](https://github.com/RISCY-SVT/xslim/releases) 页面。
供应商版本仍由[上游项目](https://github.com/spacemit-com/xslim/releases)发布。

## 参与贡献

下游版本问题请提交到 [RISCY-SVT Issue 跟踪器](https://github.com/RISCY-SVT/xslim/issues)。
上游贡献请遵循 [SpacemiT 项目](https://github.com/spacemit-com/xslim) 的流程。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
