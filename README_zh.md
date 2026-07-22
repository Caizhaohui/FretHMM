# FretHMM

单分子时间序列隐马尔可夫模型（HMM）状态分类工具。受 [HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy) 启发，使用 Python 从零重写，支持跨平台运行、批量处理和 GUI 交互。

**[English](README.md)**

## 功能概览

| 特性 | 说明 |
|------|------|
| HMM 引擎 | Baum-Welch 训练 + Viterbi 解码（基于 hmmlearn），支持自定义初始猜测值 |
| 数据模式 | 自动检测 / 单通道信号 / 双通道 Donor-Acceptor（自动计算 FRET 效率） |
| 批量处理 | 多文件并行（`ProcessPoolExecutor`），支持目录扫描与多进程 |
| Review Grid | 批量分类 + 分页多面板 PNG 可视化审查，快速筛查分类质量 |
| 低态尾部裁剪 | 两遍 HMM 拟合，自动识别并裁剪持续低信号尾部（如光漂白态） |
| CLI | `run`、`tdp`、`review-grid`、`events`、`dwell-stats`、`gui` 六个子命令 |
| GUI | CustomTkinter 界面，深色/浅色主题，中英文切换，后台线程分析，支持批量 review grid 导出 |
| 输出格式 | `*_classified.csv`、`*_summary.json`、`*report.dat`、`*path.dat`、`*dwell.dat`（GUI 可勾选） |
| TDP | 转换密度图（Transition Density Plot）可视化 + 高斯速率拟合 |
| 打包 | PyInstaller 一键构建 Windows 可执行文件（支持目录模式 / `--onefile` 单文件模式） |

## 安装

```bash
git clone https://github.com/Caizhaohui/FretHMM.git
cd FretHMM
pip install -e .
```

**运行依赖：**

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- hmmlearn >= 0.3.0
- matplotlib >= 3.7（TDP 和 Review Grid 可视化需要）
- customtkinter >= 5.2.0（GUI 需要）

**可选依赖：**

```bash
pip install -e ".[dev]"    # 安装 pytest 测试框架
pip install -e ".[gui]"    # 安装 PyInstaller 打包工具
```

## 使用方法

### CLI

FretHMM 提供六个子命令：`run`（HMM 分析）、`review-grid`（可视化审查）、`tdp`（转换密度图）、`events`（ON/OFF 事件分析）、`dwell-stats`（停留时间统计 + 速率拟合）、`gui`（图形界面）。

#### run — HMM 状态分类

```bash
# 单文件分析（2 态，自动检测数据格式）
frethmm run --files trace.csv --states 2 --output-dir ./results/

# 批量处理目录下所有轨迹文件（4 个并行进程）
frethmm run --input-dir ./traces/ --states 5 --workers 4 --output-dir ./results/

# 同时指定多个文件
frethmm run --files trace1.csv trace2.csv trace3.csv --states 3 --output-dir ./results/

# 提供初始猜测值（适用于状态间距较小的情况）
frethmm run --files data.csv --states 2 --guesses "0.3,0.7"

# 指定单通道模式及信号列
frethmm run --files data.csv --states 2 --mode single_channel --signal-column 1

# 使用低态尾部裁剪（裁剪持续 ≥ 5 秒的低信号尾部后重新分类）
frethmm run --files trace.csv --states 2 --low-state-tail-trim-seconds 5.0

# 只输出主结果 classified.csv
frethmm run --files data.csv --states 2 --classified-only

# 详细输出模式（显示所有警告信息）
frethmm run --files data.csv --states 3 -v
```

**`run` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--files` | — | 指定一个或多个轨迹文件路径（与 `--input-dir` 二选一，必填） |
| `--input-dir` | — | 指定输入目录，自动扫描其中所有轨迹文件（与 `--files` 二选一，必填） |
| `--output-dir` | — | 输出目录（默认与输入文件同目录） |
| `--states` | 2 | HMM 状态数，或填 `auto` 由 BIC 自动选择（详见[算法加固](#算法加固多次启动拟合--bic-模型选择)） |
| `--guesses` | 无 | 逗号分隔的初始信号猜测值，数量须与 `--states` 一致（`--states auto` 时忽略） |
| `--max-iter` | 500 | Baum-Welch 最大迭代次数 |
| `--tol` | 1e-4 | 收敛容差 |
| `--workers` | 1 | 并行工作进程数（>1 时启用多进程批处理） |
| `--mode` | auto | 数据模式：`auto`（自动检测）/ `paired_channel`（双通道）/ `single_channel`（单通道） |
| `--signal-column` | 1 | 单通道模式下选择的信号列索引（1-based，第 1 列为 Time 之后的列） |
| `--low-state-tail-trim-seconds` | 无 | 低态尾部裁剪阈值（秒），启用两遍拟合（详见[数据过滤](#数据过滤低态尾部裁剪)） |
| `--n-init` | 10 | 确定性多次启动 Baum-Welch 的次数，取对数似然最高的结果（填 `1` 可复现旧版单次拟合） |
| `--min-states` | 2 | BIC 选择的最小状态数（仅 `--states auto` 时生效） |
| `--max-states` | 6 | BIC 选择的最大状态数（仅 `--states auto` 时生效） |
| `--classified-only` | 关闭 | 仅输出 `*_classified.csv`，不写出 `summary/report/path/dwell` |
| `-v` / `--verbose` | 关闭 | 详细输出模式，显示所有警告 |

**批量处理说明：**

- `--input-dir` 会扫描目录下所有 `.csv`、`.dat`、`.txt`、`.tsv` 文件，自动跳过 `*report.dat`、`*path.dat`、`*dwell.dat`、`*_classified.csv`、`*_summary.json` 等输出文件
- `--workers N` 启用多进程并行，N 为进程数，建议不超过 CPU 核心数
- 批量过程中单个文件出错不会中断整体流程，错误信息会打印到终端

#### review-grid — 批量可视化审查

```bash
# 基本用法：生成 4×4 的 2 态审查图
frethmm review-grid --input-dir ./traces/ --output review.png --states 2

# 自定义网格布局
frethmm review-grid --input-dir ./traces/ --output review.png --states 3 --rows 5 --cols 6

# 指定初始猜测值，同时输出 classified CSV 到指定目录
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --guesses "0.2,0.8" --output-dir ./classified/

# 结合低态尾部裁剪
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --low-state-tail-trim-seconds 5.0

# 4 个并行进程加速批量分类
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --workers 4 --rows 4 --cols 8
```

**`review-grid` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | — | 输入轨迹文件目录（必填） |
| `--output` | — | 输出 PNG 路径（必填，如 `review.png`） |
| `--output-dir` | 无 | 可选，用于存放 classified CSV 侧输出 |
| `--states` | 2 | HMM 状态数，或填 `auto` 由 BIC 自动选择 |
| `--guesses` | 无 | 逗号分隔的初始信号猜测值（`--states auto` 时忽略） |
| `--max-iter` | 500 | Baum-Welch 最大迭代次数 |
| `--tol` | 1e-4 | 收敛容差 |
| `--workers` | 1 | 并行工作进程数 |
| `--mode` | auto | 数据模式：auto / paired_channel / single_channel |
| `--signal-column` | 1 | 单通道模式下的信号列索引 |
| `--low-state-tail-trim-seconds` | 无 | 低态尾部裁剪阈值（秒） |
| `--n-init` | 10 | 确定性多次启动拟合次数（填 `1` 复现旧版单次拟合） |
| `--min-states` | 2 | BIC 选择的最小状态数（仅 `--states auto` 时生效） |
| `--max-states` | 6 | BIC 选择的最大状态数（仅 `--states auto` 时生效） |
| `--rows` | 4 | 每页面板行数 |
| `--cols` | 4 | 每行面板数 |

**分页说明：** 当轨迹数量超过 `rows × cols` 时，自动生成多页图片，文件名格式为 `review_page_01.png`、`review_page_02.png` 等。每个面板上方叠加显示原始信号（灰色）与 HMM 分类信号（红色），标题标注文件名、log-likelihood 和状态均值。拟合有警告的轨迹会以橙色边框高亮标记。

#### tdp — 转换密度图

```bash
# 从输出目录中的 report 文件生成转换密度图（交互窗口）
frethmm tdp --input-dir ./results/ --exposure 0.1

# 保存为图片文件
frethmm tdp --input-dir ./results/ --exposure 0.1 --output tdp.png

# 只显示前 N 个状态（按转移频次排序）
frethmm tdp --input-dir ./results/ --exposure 0.1 --states 3 --output tdp.png
```

**`tdp` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | — | 包含 `*report.dat` 文件的目录（必填） |
| `--exposure` | 0.1 | 每帧曝光时间（秒），用于速率计算 |
| `--states` | 无 | 仅显示前 N 个状态（按转移频次排序） |
| `--output` | 无 | 输出图片路径（如 `tdp.png`），不指定则弹出交互窗口 |

#### events — ON/OFF 事件分析

从 `*_classified.csv`（`run` 的主输出）中提取离散的 ON/OFF 事件。**最高均值态视为 ON，其余所有态视为 OFF** —— 对 2 态轨迹即自然的"高值 = ON"规则，对 3 态及以上则推广为"仅最高态算 ON"。末尾较长的 OFF 段（如光漂白尾部）会被标记为 `excluded`，不纳入停留时间统计，但仍会列在输出中以便审计。

```bash
# 批量：扫描目录中的所有 *_classified.csv
frethmm events --input-dir ./results/ --output-dir ./events/

# 处理指定文件
frethmm events --files trace1_classified.csv trace2_classified.csv --output-dir ./events/

# 调整末尾 OFF 排除阈值（默认 100 秒）
frethmm events --input-dir ./results/ --tail-off-threshold-seconds 250 --output-dir ./events/
```

**`events` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | — | 含 `*_classified.csv` 的目录（与 `--files` 二选一，必填） |
| `--files` | — | 单个或多个 `*_classified.csv` 路径（与 `--input-dir` 二选一，必填） |
| `--output-dir` | — | 输出目录（必填） |
| `--tail-off-threshold-seconds` | 100.0 | 末尾事件为 OFF 且持续至少该秒数时排除 |

每次运行写出三张 CSV 表：

| 文件 | 说明 |
|------|------|
| `event_details.csv` | 每个事件一行：源文件、类型（ON/OFF）、序号、状态值、起止时间与帧、时长、是否排除 |
| `event_summary.csv` | 每个源文件一行：ON/OFF 计数、总时长与平均停留时间、末尾 OFF 排除状态 |
| `event_stats_overall.csv` | 跨文件汇总：事件计数、总/平均 ON 与 OFF 时长 |

#### dwell-stats — 停留时间统计 + 速率常数拟合

消费 `events` 产出的 `event_details.csv`，计算单分子分析所需的更深入描述性统计：ON/OFF 停留时间的中位数、标准差、min/max、25/75 百分位（跨所有分子汇总）。可选地对每个停留时间分布拟合单指数 `A·exp(-k·t)`（直方图 + `scipy.optimize.curve_fit`，约束 `k ≥ 0`），报告速率常数 `k` 及其对应的平均停留时间 `1/k`。

**速率常数的物理含义：** `on_rate_constant` 是**离开 ON 态**的速率（≈ 结合/解离动力学中的 `k_off`），`off_rate_constant` 是**离开 OFF 态**的速率（≈ `k_on`）。

```bash
# 默认：描述性统计 + 指数拟合，消费 events 输出
frethmm dwell-stats --input ./events/event_details.csv --output-dir ./stats/

# 只要描述性统计（跳过拟合）
frethmm dwell-stats --input ./events/event_details.csv --output-dir ./stats/ --no-fit

# 自定义拟合直方图 bin 数
frethmm dwell-stats --input ./events/event_details.csv --output-dir ./stats/ --bins 30
```

**`dwell-stats` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | — | `event_details.csv` 路径（`frethmm events` 的输出，必填） |
| `--output-dir` | — | 输出目录（必填） |
| `--bins` | 无 | 指数拟合的直方图 bin 数（默认 `max(10, n_events // 3)`） |
| `--no-fit` | 关闭 | 跳过指数拟合，只输出描述性统计 |

写出两张 CSV 表：

| 文件 | 说明 |
|------|------|
| `dwell_stats_summary.csv` | 单行：汇总的 ON/OFF 计数、mean/median/std/min/max/p25/p75/total，以及拟合列（速率常数、标准差、平均时间、振幅、bin 数、是否收敛）—— `--no-fit` 或拟合失败时为空 |
| `dwell_stats_per_file.csv` | 每个源文件一行：每个分子的扩展描述性统计（便于查看单分子差异） |

> **拟合为空的情况：** 指数拟合要求每类至少 5 个停留样本且直方图呈衰减形态。停留时间恒定、事件过少或分布不衰减时速率列为空 —— 描述性统计仍然有效。

#### gui — 图形界面

```bash
frethmm gui
```

GUI 界面截图（v1.0.0，含批量 review grid 审查区块）：

![FretHMM GUI v1.0.0](docs/images/gui-v1.0.0-review-grid.png)

### GUI 使用说明

```bash
frethmm gui
```

- **菜单栏**：
  - **文件 (File)**：添加文件、添加文件夹、清除所有、退出
  - **设置 (Settings)**：HMM 参数设置对话框、语言切换（English / 中文）、界面风格（明亮 / 暗黑 / 跟随系统）
  - **帮助 (Help)**：关于对话框
- **文件选择**：通过按钮或菜单选择 `.csv` / `.dat` 轨迹文件，或指定输入目录批量处理
- **状态文件夹批处理**：新增"按状态分组的文件夹批处理"面板，可同时添加多个文件夹并为每个文件夹指定不同的状态数、数据模式和信号列
- **参数面板**：状态数、初始猜测值、最大迭代次数、容差、并行数、数据模式、信号列（与输出面板并排显示）
- **输出选项**：GUI 新增输出文件勾选框，可自由选择输出 classified.csv / summary.json / report.dat / path.dat / dwell.dat
- **Review Grid 区块**：专用区域设置行数、列数和输出文件名，点击"Generate Review Grid"按钮一键生成可视化审查图
- **运行面板**：可折叠的右侧运行面板（Show/Hide Runtime），实时显示分析状态、进度、运行汇总和最近输出路径
- **结果详情**：选中结果表格中的文件后，右侧面板展示完整拟合指标（状态数、log_prob、状态均值、sigma）和警告信息
- **进度条**：实时显示分析任务完成进度
- **结果表格**：分析完成后展示每个文件的拟合结果，颜色标识（绿色=成功，橙色=警告，红色=错误）
- **主题切换**：通过 Settings 菜单或标题栏 🌓 按钮切换 Light / Dark / System 主题
- **双语支持**：Settings → Language 实时切换 English / 中文界面
- **后台线程处理**：所有分析任务在后台线程执行，支持随时取消（Cancel 按钮）
- **日志面板**：彩色日志输出（蓝色标题、橙色警告、红色错误、绿色完成）
- **状态栏**：底部显示当前状态和版本号

## 可视化功能

### Review Grid 审查图

Review Grid 是面向人工审查的批量可视化工具，将目录中所有单分子轨迹的 HMM 分类结果渲染为一张（或多张）分页拼图。

**工作原理：**

1. 扫描输入目录中的所有轨迹文件
2. 对每个文件执行 HMM 状态分类
3. 将分类结果排列为 `rows × cols` 的网格面板
4. 每个面板叠加显示原始信号（灰色细线）和 HMM 分类信号（红色粗线）
5. 面板标题显示文件名、log-likelihood 和各状态均值
6. 拟合产生警告的轨迹以橙色边框标记，便于快速定位问题文件

**输出示例：**

```
review.png                     # 单页（轨迹数 ≤ rows × cols）
review_page_01.png             # 多页时自动编号
review_page_02.png
```

**典型工作流：**

```bash
# 1. 先用 review-grid 快速审查所有轨迹的分类质量
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 --rows 4 --cols 8

# 2. 发现问题文件后，单独处理
frethmm run --files traces/bad_trace.csv --states 3 --guesses "0.1,0.5,0.9" -v

# 3. 审查通过后，批量输出完整结果
frethmm run --input-dir ./traces/ --states 2 --workers 4 --output-dir ./results/
```

### TDP 转换密度图

TDP（Transition Density Plot）从 HMM 分类生成的 `*report.dat` 文件中聚合所有分子的状态转移信息，绘制为散点密度图。

**图表构成：**

- **X 轴**：起始状态均值（Start state mean）
- **Y 轴**：终止状态均值（Stop state mean）
- **点大小和颜色**：编码转移次数（使用 `hot` 色谱，暖色 = 高频转移）
- **对角虚线**：自转移参考线

**`--states N` 过滤**：当混合不同状态数的数据集时，可通过此参数仅保留每个分子中转移频次最高的 N 个状态，便于跨数据集对比。

**速率分析**：除了可视化，FretHMM 还提供 `fit_gaussian_to_rates()` 编程接口，可对特定状态对之间的转移速率分布进行高斯拟合，提取平均速率和标准差。

## 算法加固（多次启动拟合 + BIC 模型选择）

Baum-Welch 对初始状态均值敏感，单次拟合容易陷入较差的局部最优。FretHMM 提供两项算法加固，让结果更稳定、更少依赖手动调参。

### 多次启动拟合（`--n-init`）

每次拟合时，FretHMM 会从**确定性**的初始均值出发，运行 `--n-init` 次（默认 10）Baum-Welch，并保留对数似然最高的结果。

- 第 0 次启动始终使用旧版等距默认均值，因此 `--n-init 1` 可逐字节复现历史单次拟合结果。
- 第 1..n-1 次启动在默认均值基础上叠加固定种子抖动（种子仅由配置决定，与墙钟时间无关），因此对同一输入的重复运行完全可复现。
- 填 `--n-init 1` 可完全关闭多次启动（最快，等价于旧行为）。

```bash
# 默认 10 次启动（推荐，提升稳定性）
frethmm run --files trace.csv --states 3

# 复现旧版单次拟合
frethmm run --files trace.csv --states 3 --n-init 1
```

### BIC 模型选择（`--states auto`）

当你不知道状态数时，传入 `--states auto`，FretHMM 会在 `[--min-states, --max-states]`（默认 `2`..`6`）范围内扫描，对每个候选状态数执行完整的多次启动拟合，并选取 **贝叶斯信息准则（BIC）** 最小的一个（BIC = `k·ln(n) − 2·log_prob`，其中 `k` 是 tied 协方差高斯 HMM 的自由参数数，`n` 是帧数）。

```bash
# 在 2..5 个状态范围内用 BIC 自动选择状态数
frethmm run --input-dir ./traces/ --states auto --min-states 2 --max-states 5 --workers 4
```

启用自动选择时，`*_summary.json` 会记录所选的 BIC、AIC，以及一个 `model_candidates` 表，列出每个候选的 `n_states` / `log_prob` / `bic` / `aic`，便于审计决策过程。

> **关于 `--guesses` 的说明：** `--states auto` 下初始猜测被忽略，因为每个候选的状态数不同；改由多次启动提供初始化多样性。

## 数据过滤

### 低态尾部裁剪（Low-State Tail Trimming）

**问题背景：** 在单分子荧光实验中，轨迹末尾常出现持续的低信号段（如光漂白态、荧光分子失活）。这些尾部数据不属于感兴趣的构象状态，但会被 HMM 当作一个额外的低均值状态，干扰对真实状态的正确分类。

**两遍拟合工作流：**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  第一遍 HMM  │ ──→ │  定位最低态   │ ──→ │  数据截断    │
│  完整数据拟合 │     │  持续 ≥ 阈值  │     │  移除尾部    │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  第二遍 HMM  │
                                        │  截断数据拟合 │
                                        └─────────────┘
```

1. **第一遍分类**：对完整轨迹进行 HMM 拟合，得到 Viterbi 状态路径
2. **定位最低态**：找到均值最低的状态
3. **检测持续段**：沿时间轴扫描，寻找最低态首次连续出现超过 `--low-state-tail-trim-seconds` 秒的位置
4. **截断数据**：在该时间点截断，丢弃尾部数据
5. **第二遍分类**：对截断后的数据重新运行 HMM 拟合，获得更干净的分类结果

> **注意：** 如果最低态从未连续出现超过阈值时间，则不执行截断，保留第一遍的分类结果。

**CLI 示例：**

```bash
# 单文件：裁剪持续 ≥ 5 秒的低信号尾部
frethmm run --files trace.csv --states 2 --low-state-tail-trim-seconds 5.0

# 批量处理：3 秒阈值，4 个并行进程
frethmm run --input-dir ./traces/ --states 3 --low-state-tail-trim-seconds 3.0 --workers 4

# 结合 Review Grid：先裁剪后审查
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --low-state-tail-trim-seconds 5.0 --rows 4 --cols 8
```

**输出元数据：** 启用裁剪后，`*_summary.json` 会记录以下额外字段：

```json
{
  "low_state_tail_trim_seconds": 5.0,
  "low_state_tail_cutoff_time": 47.3,
  "low_state_tail_kept_frames": 473
}
```

- `low_state_tail_trim_seconds`：设定的裁剪阈值
- `low_state_tail_cutoff_time`：实际截断时间点（`null` 表示未触发裁剪）
- `low_state_tail_kept_frames`：裁剪后保留的帧数

**GUI 使用：** 在 GUI 的输出面板中找到"低态尾部裁剪（秒）"输入框，输入阈值后点击 Run Analysis 即可。所有文件和文件夹批处理任务都会应用该裁剪设置。

## 输入格式

程序自动检测文件格式（有无表头、分隔符类型、列数），支持以下两种模式：

**单通道模式**（CSV，带表头）：

```csv
Time,channel1
0,2820
1,2884
2,2570
```

多列信号时通过 `--signal-column` 选择指定列：

```csv
Time,channel1,channel2
0,2884,-5096
1,2884,1289
```

`--signal-column 1` 使用 `channel1` 列，`--signal-column 2` 使用 `channel2` 列。

**双通道 Donor/Acceptor 模式**（空格/Tab 分隔，3 列，无表头）：

```
<time>  <donor>  <acceptor>
```

此模式下自动计算 FRET 效率 `A/(D+A)` 作为 HMM 输入信号。

## 输出文件

每个输入文件生成以下输出：

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_classified.csv` | CSV | 主输出：`time, classified_mean` 两列理想化轨迹 |
| `*_summary.json` | JSON | 状态均值、占比、转移矩阵、驻留统计、裁剪元数据、警告信息 |
| `*report.dat` | 文本 | 模型参数（状态数、均值、sigma、转移概率矩阵） |
| `*path.dat` | TSV | 原始信号通道 + FRET 信号 + 分类信号路径（每帧一行） |
| `*dwell.dat` | TSV | 驻留时间表：`<start_mean> <stop_mean> <frames_lasted>`（每个驻留段一行） |

## 项目结构

```
FretHMM/
├── frethmm/
│   ├── __init__.py              # 版本信息
│   ├── app/
│   │   ├── cli.py               # CLI 入口（run / tdp / review-grid / events / gui）
│   │   ├── gui.py               # CustomTkinter GUI
│   │   └── i18n.py              # 国际化（英文 / 中文，138 个翻译键）
│   ├── assets/
│   │   ├── frethmm.ico          # 应用图标
│   │   └── frethmm_logo.png     # 应用 Logo
│   ├── core/
│   │   ├── io.py                # 文件读写（轨迹读取 + 报告输出）
│   │   ├── model.py             # HMM 引擎（Baum-Welch + Viterbi + 低态裁剪）
│   │   ├── batch.py             # 多进程批处理器
│   │   └── postprocess.py       # 分类轨迹构建 + 驻留时间提取 + 转移统计
│   ├── domain/
│   │   └── models.py            # 数据模型（Config / Trace / Result / ExportOptions）
│   ├── formats/
│   │   └── report_parser.py     # report.dat 解析器
│   ├── legacy/
│   │   └── report_parser.py     # 旧版报告格式解析器
│   └── viz/
│       ├── review_grid.py       # Review Grid 批量可视化审查（分页拼图）
│       └── tdp.py               # 转换密度图可视化 + 高斯速率拟合
├── tests/
│   ├── fixtures/                # 回归测试基准数据
│   ├── test_io.py               # I/O 与报告解析测试
│   ├── test_review_grid.py      # Review Grid 可视化测试
│   └── test_golden.py           # CLI 回归测试
├── docs/
│   ├── images/                  # 截图
│   └── FretHMM-refactor-plan.md # 开发路线
├── pyproject.toml               # 项目配置
├── build_exe.py                 # PyInstaller 打包脚本
├── frethmm.spec                 # PyInstaller 规格文件
├── LICENSE                      # MIT License
└── README.md
```

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 可复现性

每次成功的 CLI 分析都会在主输出旁写入带时间戳的
`frethmm_run_manifest_*.json`。该清单记录命令、拟合参数、输入/输出文件元数据、
FretHMM 版本、Python 版本和运行时依赖版本，但不会复制实验数据。即使使用
`--classified-only`，也会保留运行清单；该选项仅跳过分类的辅助输出。

`tests/data/` 中提交的是小型合成或脱敏的 CSV 与兼容报告样本，可完整运行核心
回归测试，不包含原始图像或 ND2 文件。FretHMM 当前从已导出的轨迹文件
（`.csv`、`.dat`、`.txt`、`.tsv`）开始分析；ND2 图像到轨迹的处理仍属于上游流程。

## 打包为可执行文件

```bash
# 先安装明确的构建依赖
pip install -e ".[gui]"

# 目录模式（默认，生成 dist/FretHMM/ 目录）
python build_exe.py

# 单文件模式（生成 dist/FretHMM.exe，便于分发）
python build_exe.py --onefile
```

构建产物为独立的 Windows GUI 可执行文件，无需 Python 环境。目录模式还会生成
`dist/FretHMM.zip`、SHA-256 校验文件和 JSON 发布清单。可使用
`dist/FretHMM/FretHMM.exe --version` 验证 EXE 启动路径而不打开 GUI。

## 更新日志

### v1.4.0（候选发布）

停留时间统计与速率常数拟合：

- **新增 `dwell-stats` 子命令**：消费 `event_details.csv`（`events` 的输出），写出 `dwell_stats_summary.csv` + `dwell_stats_per_file.csv`，含 ON/OFF 停留时间的扩展描述性统计（中位数、标准差、min/max、p25/p75）。
- **指数速率常数拟合**：对汇总的停留时间直方图做单指数 `A·exp(-k·t)` 拟合（直方图 + `scipy.optimize.curve_fit`，约束 `k ≥ 0`），报告 ON/OFF 的速率常数及对应平均停留时间。可用 `--no-fit` 跳过。
- **新增模块**：`frethmm/core/dwell_stats.py`（`describe_durations`、`fit_exponential_dwell`、扩展汇总）与 `frethmm/formats/event_details_parser.py`（反解析 `event_details.csv`）。
- **无回归**：`events` 命令与 `events.py` 不变；`dwell-stats` 是纯下游消费者。
- **测试**：新增 `test_dwell_stats.py`（12 项）与 `test_dwell_stats_cli.py`（3 项端到端），均自包含。
- **发布可复现性**：CLI 分析命令现在写入带时间戳的运行清单；提交脱敏夹具替代依赖工作区的跳过回归测试；Windows CI 构建 GUI 包并进行版本启动检查。
- **尾部裁剪修正**：仅当最低态持续到轨迹末尾时才裁剪，避免错误删除中间的有效低态片段。

### v1.3.0 (2026-06-15)

ON/OFF 事件分析收编进包：

- **新增 `events` 子命令**：从 `*_classified.csv` 提取 ON/OFF 事件，写出 `event_details.csv`、`event_summary.csv`、`event_stats_overall.csv` 三张表。
- **N 态推广**：最高均值态 = ON，其余态 = OFF（2 态时与旧版"高值 = ON"规则完全一致）。
- **末尾 OFF 排除**：末尾较长的 OFF 段（默认 ≥ 100 秒）标记为 `excluded`，不纳入停留时间统计但仍列出。
- **新增模块**：`frethmm/core/events.py`（事件检测 + 汇总）、`frethmm/formats/classified_parser.py`（反解析 `*_classified.csv`）、`frethmm/core/io.py` 中的 `find_classified_files`。
- **测试**：新增 `test_events.py`、`test_classified_parser.py`、`test_events_cli.py`，均使用自包含合成轨迹（不依赖外部样本）。

### v1.2.0 (2026-06-15)

算法加固 —— 多次启动拟合与基于 BIC 的状态数选择：

- **多次启动拟合**（`--n-init`，默认 10）：确定性的多次启动 Baum-Welch，保留对数似然最高的结果。第 0 次启动复现旧版单次拟合，因此 `--n-init 1` 与 v1.1 逐字节兼容。
- **BIC 模型选择**（`--states auto` 配合 `--min-states`/`--max-states`）：扫描状态数范围，对每个候选用多次启动拟合，选取 BIC 最小者。
- **新增 metrics 模块**（`frethmm.core.metrics`）：`compute_aic`、`compute_bic`、`count_gaussian_hmm_params`。
- **summary JSON**：当多次启动或自动选择启用时记录 `n_init`、`best_start_index`、`bic`、`aic`、`model_candidates`（旧版单次拟合不写入这些字段，保持字节兼容）。
- **GUI**：参数区新增 "自动选择状态数 (BIC)" 复选框及最小/最大状态范围，并新增 `n_init` 输入框；参数对话框与文件夹批处理任务均支持自动选择。
- **测试**：新增 `test_multistart.py` 与 `test_model_selection.py`（含合成轨迹夹具）；golden 测试固定 `--n-init 1` 以保持字节级回归。

### v1.1.0 (2026-06-09)

文档与发布基础设施更新：

- **README 增强**：新增可视化功能（Review Grid + TDP）详细说明、数据过滤（低态尾部裁剪）完整工作流文档、GUI 功能详细描述
- **LICENSE**：新增 MIT License 文件
- **版本同步**：`__init__.py` 与 `pyproject.toml` 版本号统一为 1.1.0
- **`.gitignore` 更新**：补充日志目录等排除规则

### v1.0.0 (2026-06-04)

面向人工审查的批量可视化发布版本：

- **批量 review grid CLI**：新增 `review-grid` 子命令，可对目录中的单分子轨迹批量分类并导出分页拼图总览
- **GUI/EXE review grid**：GUI 新增 `Batch Review Grid` 区块，支持从文件或文件夹直接生成分页审查图
- **分页拼图布局**：支持自定义 `rows x cols`，适合 `2-state`、`3-state` 等批量样本的人眼快速筛查
- **可视化审查增强**：每个子图叠加 raw signal 和 classified trace，并显示文件名、`log_prob`、`state means`

### v0.6.0 (2026-06-01)

GUI 界面布局优化与打包瘦身：

- **布局重构**：移除 ScrollableFrame，改用扁平布局；参数面板与输出面板并排显示，节省纵向空间
- **可折叠运行面板**：右侧运行面板默认隐藏，通过 "Show/Hide Runtime" 按钮切换显示，最大化主工作区
- **应用图标**：新增 `frethmm.ico` 和 `frethmm_logo.png` 资源文件，窗口标题栏和任务栏显示自定义图标
- **窗口尺寸调整**：默认尺寸从 1150×750 增至 1280×720，最小尺寸 1180×660
- **空态安全**：`_tree`、`_log_text` 等控件初始化为 `None`，所有访问前增加空检查，防止构建阶段异常
- **PyInstaller 打包瘦身**：精简 spec 文件，使用 `collect_data_files` + `collect_dynamic_libs` 替代 `collect_all`；排除 PyQt5 / matplotlib / pandas / pytest / torch 等未使用的包，显著减小 EXE 体积
- **`--onefile` 模式**：`build_exe.py` 新增 `--onefile` 参数，通过 `FRETHMM_ONEFILE` 环境变量控制生成单文件 EXE

### v0.5.0 (2026-06-01)

GUI 稳定性修复与导出选项增强：

- **`ExportOptions` 数据类**：新增 `ExportOptions` 域模型，支持精细控制每种输出文件的生成（classified_csv / summary_json / state_report / state_path / dwell_report）
- **GUI 输出文件勾选框**：在输出面板新增复选框，用户可自由选择需要输出的文件类型（classified.csv 始终输出）
- **Worker 错误处理增强**：后台线程异常现在输出完整 traceback 到日志面板和调试日志文件（`%LOCALAPPDATA%/FretHMM/frethmm-gui.log`）
- **全局异常钩子**：`sys.excepthook` 捕获主线程未处理异常，在 `console=False` 的 EXE 中也能弹出错误对话框
- **`_poll_queue` 修复**：消息处理异常不再导致轮询终止；移除 `is_alive()` 检查避免队列消息丢失
- **`_on_mode_changed` 修复**：CTkComboBox 的 `command` 回调签名从 `Event` 改为 `str`
- **PyInstaller 打包优化**：使用 `collect_all` + `copy_metadata` 完整收集 hmmlearn / sklearn / scipy / numpy / matplotlib 资源
- **测试**：新增 `TestProcessTraceFileExports` 单元测试

### v0.4.0 (2026-06-01)

GUI 现代化重构与 CLI 功能增强：

- **CustomTkinter 迁移**：GUI 从 tkinter/ttk 全面迁移到 CustomTkinter，支持明亮/暗黑/跟随系统三种界面风格
- **`--classified-only` 参数**：CLI 新增 `--classified-only` 开关，仅输出 `*_classified.csv`，跳过 summary/report/path/dwell 辅助文件
- **文件夹批处理面板**：GUI 新增"按状态分组的文件夹批处理"面板，可同时添加多个文件夹并为每个文件夹独立指定状态数、数据模式等参数
- **运行面板与结果详情**：GUI 新增右侧运行面板，实时展示分析状态、进度、运行汇总；选中结果行后展示完整拟合指标和警告信息
- **完成通知对话框**：分析完成后弹出汇总对话框，显示成功/警告/错误计数和最后输出路径
- **i18n 扩展**：新增约 30 条翻译条目，覆盖主题切换、文件夹批处理、运行面板、结果详情等全部新功能
- **测试**：新增 `test_cli_run_classified_only_writes_only_primary_csv` 回归测试
- **依赖**：新增 `customtkinter>=5.2.0`

### v0.3.0 (2026-06-01)

项目重构为 FretHMM，建立模块化架构：

- 模块化拆分为 `core` / `domain` / `app` / `formats` / `legacy` / `viz` 六个子包
- CLI 支持单文件（`--files`）和目录批量（`--input-dir`）两种处理模式，支持多进程并行（`--workers`）
- GUI 完整功能：菜单栏、参数设置对话框、中英文切换、后台线程分析、彩色日志
- 默认生成 `*_classified.csv`（`time, classified_mean`）和 `*_summary.json` 主输出
- 同时输出 `report / path / dwell` 格式文件
- TDP 转换密度图可视化 + 高斯速率拟合
- PyInstaller 一键构建 Windows GUI 可执行文件
- 回归测试覆盖（I/O、报告解析、CLI 端到端）

### v0.2.0 (2026-06-01)

GUI 重大更新：

- 新增菜单栏（File / Settings / Help）和独立参数设置对话框
- 新增多语言支持（i18n），英文和中文界面实时切换
- 现代化 UI 样式：平台自适应字体、自定义 ttk 主题、彩色日志、状态栏
- 启动速度优化：GUI 延迟导入重型库，后台预热
- 警告处理优化：捕获 HMM 拟合警告，GUI 中以橙色标识

### v0.1.0 (2026-05-30)

初始版本：

- 完整 HMM 分析流程（Baum-Welch 训练 + Viterbi 解码）
- CLI 工具（`run` / `tdp` / `gui` 子命令）
- tkinter GUI（文件选择、参数面板、进度条、结果表格、日志面板）
- 多进程批处理支持
- TDP 可视化
- PyInstaller GUI 打包脚本

## 许可证

[MIT License](LICENSE)
