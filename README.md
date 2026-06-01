# FretHMM

单分子时间序列隐马尔可夫模型（HMM）状态分类工具。受 [HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy) 启发，使用 Python 从零重写，支持跨平台运行、批量处理和 GUI 交互。

## 功能概览

| 特性 | 说明 |
|------|------|
| HMM 引擎 | Baum-Welch 训练 + Viterbi 解码（基于 hmmlearn） |
| 数据模式 | 自动检测 / 单通道信号 / 双通道 Donor-Acceptor |
| 批量处理 | 多文件并行（`ProcessPoolExecutor`），支持目录扫描 |
| CLI | `run`、`tdp`、`gui` 三个子命令 |
| GUI | tkinter 界面，中英文切换，后台线程分析 |
| 输出格式 | `*_classified.csv`、`*_summary.json` 及 HaMMy 兼容格式 |
| TDP | 转换密度图（Transition Density Plot）可视化 |
| 打包 | PyInstaller 一键构建 Windows 可执行文件 |

## 与原版 HaMMy 的对比

| 特性 | 原版 HaMMy | FretHMM |
|------|-----------|---------|
| 平台 | 仅 Windows | 跨平台（Windows / Linux / macOS） |
| 源码 | 闭源（Numerical Recipes in C） | 完全开源 |
| 界面 | GUI（WinForms） | CLI + GUI（tkinter） |
| 并行 | 单线程 | 多进程并行批处理 |
| 状态数上限 | 10 | 无限制 |
| 数据格式 | Donor/Acceptor 对 | 自动检测，支持单通道和多通道 |

## 安装

```bash
git clone https://github.com/Caizhaohui/FretHMM.git
cd FretHMM
pip install -e ".[dev]"
```

**运行依赖：**

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- hmmlearn >= 0.3.0
- matplotlib >= 3.7（TDP 可视化需要）

**可选依赖：**

- `pip install -e ".[dev]"` — 安装 pytest 测试框架
- `pip install -e ".[gui]"` — 安装 PyInstaller 打包工具

## 使用方法

### CLI

```bash
# 单文件分析（2 态，自动检测数据格式）
frethmm run --files trace.csv --states 2 --output-dir ./results/

# 批量处理目录下所有轨迹文件（4 个并行进程）
frethmm run --input-dir ./traces/ --states 5 --workers 4 --output-dir ./results/

# 提供初始猜测值（适用于状态间距较小的情况）
frethmm run --files data.csv --states 2 --guesses "0.3,0.7"

# 指定单通道模式及信号列
frethmm run --files data.csv --states 2 --mode single_channel --signal-column 1

# 详细输出模式
frethmm run --files data.csv --states 3 -v

# 生成转换密度图
frethmm tdp --input-dir ./results/ --exposure 0.1 --output tdp.png

# 启动 GUI
frethmm gui
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--states` | 2 | HMM 状态数 |
| `--guesses` | 无 | 逗号分隔的初始信号猜测值 |
| `--max-iter` | 500 | Baum-Welch 最大迭代次数 |
| `--tol` | 1e-4 | 收敛容差 |
| `--workers` | 1 | 并行工作进程数 |
| `--mode` | auto | 数据模式：`auto` / `paired_channel` / `single_channel` |
| `--signal-column` | 1 | 单通道模式下的信号列索引（1-based） |
| `--input-dir` | — | 输入目录（与 `--files` 二选一） |
| `--files` | — | 指定文件列表（与 `--input-dir` 二选一） |
| `--output-dir` | — | 输出目录（默认与输入文件同目录） |

### GUI

```bash
frethmm gui
```

GUI 功能：

- **菜单栏**：文件管理、HMM 参数设置、中英文切换、关于信息
- **文件选择**：添加文件 / 文件夹，支持 `.csv`、`.dat`、`.txt`、`.tsv`
- **参数面板**：状态数、初始猜测值、迭代次数、容差、并行数、数据模式
- **进度条**：实时显示分析进度
- **结果表格**：每个文件的拟合结果，颜色标识状态（绿色=成功，橙色=警告，红色=错误）
- **日志面板**：彩色日志输出
- **状态栏**：显示当前状态和版本号

### 打包为可执行文件

```bash
python build_exe.py
```

构建产物位于 `dist/FretHMM/`，生成独立的 Windows GUI 可执行文件，无需 Python 环境。

## 输入格式

程序自动检测文件格式（有无表头、分隔符类型、列数），支持以下两种模式：

**单通道模式**（CSV，带表头）：

```csv
Time,channel1
0,2820
1,2884
2,2570
```

或包含多列信号时通过 `--signal-column` 选择：

```csv
Time,channel1,channel2
0,2884,-5096
1,2884,1289
```

**双通道 Donor/Acceptor 模式**（空格/Tab 分隔，3 列）：

```
<time>  <donor>  <acceptor>
```

此模式下自动计算 FRET 效率 `A/(D+A)` 作为 HMM 输入信号。

## 输出文件

每个输入文件生成以下输出：

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_classified.csv` | CSV | 主输出：`time, classified_mean` 两列理想化轨迹 |
| `*_summary.json` | JSON | 状态均值、占比、转移矩阵、驻留统计、警告信息 |
| `*report.dat` | 文本 | HaMMy 兼容：模型参数（状态数、均值、sigma、转移概率） |
| `*path.dat` | 文本 | HaMMy 兼容：原始信号 + 分类信号路径 |
| `*dwell.dat` | 文本 | HaMMy 兼容：驻留时间表 |

## 项目结构

```
FretHMM/
├── frethmm/
│   ├── __init__.py              # 版本信息
│   ├── app/
│   │   ├── cli.py               # CLI 入口（run / tdp / gui）
│   │   ├── gui.py               # tkinter GUI（1123 行）
│   │   └── i18n.py              # 国际化（英文/中文）
│   ├── core/
│   │   ├── io.py                # 文件读写（轨迹读取 + 报告输出）
│   │   ├── model.py             # HMM 引擎（hmmlearn 封装 + Baum-Welch + Viterbi）
│   │   ├── batch.py             # 多进程批处理器
│   │   └── postprocess.py       # 分类轨迹构建 + 驻留时间提取 + 转移统计
│   ├── domain/
│   │   └── models.py            # 数据模型（ClassificationConfig / SignalTrace / ClassificationResult）
│   ├── formats/
│   │   └── report_parser.py     # FretHMM 输出报告解析器
│   ├── legacy/
│   │   └── report_parser.py     # HaMMy 原版 report.dat 解析器
│   └── viz/
│       └── tdp.py               # 转换密度图可视化 + 高斯速率拟合
├── tests/
│   ├── fixtures/                # 回归测试基准数据
│   ├── test_io.py               # I/O 与报告解析测试
│   └── test_golden.py           # HaMMy 样例 golden tests + CLI 回归测试
├── docs/
│   └── FretHMM-refactor-plan.md # 重构路线与任务清单
├── pyproject.toml               # 项目配置（v0.3.0）
├── build_exe.py                 # PyInstaller 打包脚本
├── frethmm.spec                 # PyInstaller 规格文件
└── README.md
```

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

测试覆盖范围：

- **I/O 测试**（`test_io.py`）：FRET 比值计算、轨迹读取、报告写入/解析往返、文件发现与过滤
- **Golden 测试**（`test_golden.py`）：HaMMy 2 态 / 10 态样例报告解析、Viterbi 路径重映射、CLI 端到端回归（Values1.csv / Values2.csv 输出哈希校验）

## 开发路线

详见 [docs/FretHMM-refactor-plan.md](./docs/FretHMM-refactor-plan.md)。

### 当前已完成（v0.3.0）

- [x] 项目从 pyHaMMy 重命名为 FretHMM
- [x] 模块化架构拆分（core / domain / app / formats / legacy / viz）
- [x] CLI 批量处理（`--files` / `--input-dir` / `--workers`）
- [x] GUI 完整功能（菜单栏、参数设置、中英文切换、后台分析）
- [x] 主输出 `*_classified.csv` + `*_summary.json`
- [x] HaMMy 兼容输出 `report / path / dwell`
- [x] TDP 转换密度图可视化
- [x] PyInstaller Windows GUI 打包
- [x] Golden tests 回归测试覆盖

### 计划中

- [ ] 多起点拟合（multi-start）降低局部最优敏感性
- [ ] AIC/BIC 模型选择自动确定状态数
- [ ] 最小驻留时间合并与近邻状态合并
- [ ] GUI 内嵌轨迹预览与分类信号叠加显示
- [ ] 批量分析实验级汇总表
- [ ] 异常值检测与 NaN/Inf 预处理

## 致谢

本项目是对 [HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy) 的 Python 重写。原版 HaMMy 由 Sean McKinney（UIUC）开发，基于隐马尔可夫模型对单分子 FRET 时间轨迹进行概率分析。

## 许可证

MIT License
