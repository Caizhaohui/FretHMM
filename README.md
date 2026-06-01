# FretHMM

[HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy) 思路启发下开发的单分子时间序列隐马尔可夫模型分类工具。

重构路线与执行任务清单见 [docs/FretHMM-refactor-plan.md](./docs/FretHMM-refactor-plan.md)。

FretHMM 当前以两列 `time, signal` 单通道数据为主工作流，同时保留部分 HaMMy 风格输出用于兼容和对照。

## 与原版 HaMMy 的对比

| 特性 | 原版 HaMMy | FretHMM |
|------|-----------|---------|
| 平台 | 仅 Windows | 跨平台 (Windows / Linux / macOS) |
| 源码 | 闭源 (Numerical Recipes in C) | 完全开源 |
| 界面 | GUI (WinForms) | CLI 命令行 (批量处理 / HPC 友好) |
| 并行 | 单线程串行 | 多进程并行批处理 |
| 状态数上限 | 10 | 无限制 |
| 输出格式 | `*report.dat` / `*path.dat` / `*dwell.dat` | 兼容原版文件布局，持续补充兼容性验证 |

## 功能特性

- **Baum-Welch** 训练 (HMM 的 EM 算法)，使用 tied 协方差高斯发射模型
- **Viterbi** 解码，生成理想化状态轨迹
- **自动检测** 数据格式 (donor/acceptor 对 或 单通道信号)
- **多进程批处理** 支持大量文件的并行分析
- **TDP 可视化** (Transition Density Plot)，包含高斯速率拟合辅助函数
- 默认生成 `*_classified.csv` 与 `*_summary.json`
- 读取原版 HaMMy `report.dat` 样例并保留同类 `report/path/dwell` 兼容输出

## 安装

```bash
# 克隆仓库
git clone <your-repo-url> FretHMM
cd FretHMM

# 安装（开发模式，含测试与打包依赖）
pip install -e ".[dev,gui]"
```

## 使用方法

### 基本用法

```bash
# 单文件分析（2态）
frethmm run --files trace.csv --states 2 --output-dir ./results/

# 批量分析目录下所有文件（4个并行进程）
frethmm run --input-dir ./traces/ --states 5 --workers 4 --output-dir ./results/

# 提供初始猜测值（适用于状态间距较小的情况）
frethmm run --files data.csv --states 2 --guesses "0.3,0.7"

# 详细输出模式
frethmm run --files data.csv --states 3 -v

# 指定数据模式（单通道）
frethmm run --files data.csv --states 2 --mode single_channel

# 典型两列单分子信号数据
frethmm run --files ../Values1.csv --states 2 --mode single_channel
```

### TDP 可视化

```bash
# 从 report 文件生成转换密度图
frethmm tdp --input-dir ./results/ --exposure 0.1

# 保存为图片文件
frethmm tdp --input-dir ./results/ --exposure 0.1 --output tdp.png
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--states` | 2 | HMM 状态数 |
| `--guesses` | 无 | 逗号分隔的初始 FRET/信号猜测值 |
| `--max-iter` | 500 | Baum-Welch 最大迭代次数 |
| `--tol` | 1e-4 | 收敛容差 |
| `--workers` | 1 | 并行工作进程数 |
| `--mode` | auto | 数据模式：`auto` / `fret` / `donor_acceptor` / `single_channel` |
| `--signal-column` | 1 | 单通道模式下的信号列索引 |

## GUI 界面

FretHMM 提供了一个基于 `tkinter` 的图形用户界面，适合交互式分析和结果预览。

### 启动方式

```bash
frethmm gui
```

### 界面说明

- **菜单栏**：
  - **文件 (File)**：添加文件、添加文件夹、清除所有、退出
  - **设置 (Settings)**：HMM 参数设置对话框、语言切换（English / 中文）
  - **帮助 (Help)**：关于对话框
- **参数设置对话框**：通过菜单 Settings → HMM Parameters 打开独立参数设置窗口
- **文件选择**：通过按钮或菜单选择 `.csv` / `.dat` 轨迹文件，或指定输入目录进行批量处理
- **参数面板**：在主界面中直接设置状态数、初始猜测值、最大迭代次数等 HMM 参数
- **进度条**：实时显示当前分析任务的完成进度
- **结果表格**：分析完成后展示每个文件的拟合结果，带颜色状态标识（绿色=成功，橙色=有警告，红色=错误）
- **日志面板**：彩色日志输出（蓝色=标题、橙色=警告、红色=错误、绿色=完成）
- **状态栏**：底部显示当前状态和版本号
- **多语言支持**：支持英文（默认）和中文界面切换，所有 UI 元素实时刷新

### 打包为可执行文件

如需分发 GUI 给无 Python 环境的用户，可使用 PyInstaller 构建 Windows bundle：

 ```bash
python build_exe.py
```

生成内容位于 `dist/pyHaMMy/`。当前打包目标是 GUI 分析流程，不包含 `matplotlib`，因此不覆盖 `frethmm tdp` 命令。

## 输入格式

支持自动检测的 ASCII 文本文件：

**Donor/Acceptor 模式**（3列，空格/Tab分隔）：
```
<time>  <donor_intensity>  <acceptor_intensity>
```

**单通道模式**（CSV 格式，带表头）：
```csv
Time,channel1,channel2
0,2820,-5096
1,2820,1342
```

> **说明**：单通道模式下，程序会自动将指定的信号列作为观测数据输入 HMM，无需进行 FRET = A/(D+A) 的预计算。

## 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_classified.csv` | 默认输出 | 两列：`time, classified_mean` |
| `*_summary.json` | 默认输出 | 状态均值、状态占比、转移矩阵、驻留统计、警告信息 |
| `*report.dat` | 兼容输出 | 模型参数：状态数、峰值、sigma、转移概率矩阵 |
| `*path.dat` | 兼容输出 | HaMMy 风格路径输出 |
| `*dwell.dat` | 兼容输出 | 驻留时间表：`<start_mean> <stop_mean> <frames_lasted>` |

## 项目结构

```
FretHMM/
├── frethmm/
│   ├── __init__.py         # 版本信息
│   ├── app/
│   │   ├── cli.py          # CLI 入口
│   │   ├── gui.py          # GUI 界面 (tkinter)
│   │   └── i18n.py         # 国际化 (英文/中文翻译)
│   ├── core/
│   │   ├── io.py           # 文件读写 (输入轨迹 + 输出报告)
│   │   ├── model.py        # HMM 引擎 (hmmlearn 封装)
│   │   ├── batch.py        # 多进程批处理器
│   │   └── postprocess.py  # 分类轨迹 + 驻留时间提取
│   ├── domain/
│   │   └── models.py       # 配置、输入轨迹、分类结果数据模型
│   └── viz/
│       └── tdp.py          # TDP 可视化 + 高斯拟合
├── tests/
│   ├── test_io.py          # I/O 与原版 report 解析测试
│   └── test_golden.py      # HaMMy 示例输出 golden tests
├── pyproject.toml          # 项目配置
├── build_exe.py            # PyInstaller 打包脚本
├── README.md
└── .gitignore
```

## 依赖

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- hmmlearn >= 0.3.0
- matplotlib >= 3.7

## 当前验证范围

- 已有 golden tests 校验 `HaMMy-main` 自带 `report.dat` 样例的解析兼容性
- 已验证 pyHaMMy 可生成与原版相同命名约定的 `report/path/dwell` 文件
- 尚未建立“相同输入轨迹下，数值结果与原版 HaMMy 完全一致”的系统性回归基准

## 致谢

本项目是对 [HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy) 的 Python 重写。原版 HaMMy 由 Sean McKinney (UIUC) 开发，基于隐马尔可夫模型对单分子 FRET 时间轨迹进行概率分析。

## 许可证

MIT License

## 更新日志

### v0.2.0 (2026-06-01)

**GUI 重大更新**

- **菜单栏**：新增 File / Settings / Help 菜单，支持通过菜单栏操作文件、打开参数设置对话框、切换语言、查看关于信息
- **参数设置对话框**：Settings → HMM Parameters 打开独立参数设置窗口，可统一配置所有分析参数
- **多语言支持 (i18n)**：新增 `i18n.py` 模块，支持英文（默认）和中文界面切换，所有 UI 元素（菜单、按钮、标签、表格表头、状态栏、对话框、日志消息）实时刷新
- **现代化 UI 样式**：
  - 平台自适应字体（Windows: Segoe UI / Consolas, macOS: Helvetica Neue / Menlo, Linux: Helvetica / DejaVu Sans Mono）
  - 自定义 ttk 样式主题（clam 基础 + 蓝色强调色 `#1565C0`）
  - 结果表格行高 28px，带颜色状态标签（绿色/橙色/红色）
  - 日志面板彩色输出（蓝色标题、橙色警告、红色错误、绿色完成）
  - 状态栏显示当前状态和版本号
- **启动速度优化**：
  - GUI 模块不再在顶层导入 `hmmlearn` / `sklearn` / `numpy` 等重型库
  - HMM 分析在后台线程中按需导入依赖
  - 配置对象通过 `pickle` 序列化传递给工作线程，避免主线程加载 `numpy`
  - 后台预热线程预加载 `numpy`，不阻塞 UI
  - PyInstaller 打包排除 `matplotlib` / `unittest` / `pydoc` 等不需要的模块

**警告处理优化**

- `model.py`：使用 `warnings.catch_warnings(record=True)` 捕获 HMM 拟合过程中的所有警告（收敛失败、数值问题等），过滤 `DeprecationWarning` / `FutureWarning`，存入 `ClassificationResult.warnings` 字段
- `models.py`：`ClassificationResult` 包含 `warnings: list[str]` 字段
- `gui.py`：警告以橙色显示在日志面板中，结果表格中有警告的条目标记为 "OK (warnings)"（橙色）
- `batch.py` / `cli.py`：分析完成后打印每个文件的警告信息
- 修复 GUI 后台线程的 `_Msg` 日志消息封装错误
- 修复按均值排序状态后的 Viterbi 路径重映射错误
- 新增基于 `HaMMy-main` 官方样例输出的 golden tests

### v0.1.0 (2026-05-30)

- 初始版本：完整的 HMM 分析流程（Baum-Welch 训练 + Viterbi 解码）
- CLI 工具（`run` / `tdp` / `gui` 子命令）
- tkinter GUI（文件选择、参数面板、进度条、结果表格、日志面板）
- 多进程批处理支持
- TDP 可视化
- 生成 `report/path/dwell` 输出
- 提供 PyInstaller GUI 打包脚本
