# Transformer 文本摘要训练系统

一个可复用、可扩展的 Transformer 文本摘要训练与实验框架。支持标准 Encoder-Decoder 结构、相对位置编码、线性注意力、梯度检查点、完整的训练日志与可视化，以及一键批量运行消融实验。

本说明覆盖两类使用场景：
- 通过 `main.py` 进行常规单次训练
- 通过 `run_ablation.py` 进行批量消融实验

----------------------------------------

## 目录
- 安装与环境
- 数据格式
- 快速开始（main）
- 参数说明（main）
- 训练日志与可视化
- 消融实验（run_ablation）
- 结果汇总与分析
- 性能与稳定性建议
- 常见问题排查

----------------------------------------

## 安装与环境
- 操作系统：Linux / Windows / macOS（建议 Linux + CUDA）
- Python：>= 3.10
- 依赖：`torch`, `pandas`, `matplotlib`, `tqdm`, `numpy`

示例安装：
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch pandas matplotlib tqdm numpy
```

可选：启用 TF32 与 cudnn benchmark（已在 `train.py` 中自动启用，提升吞吐）。

----------------------------------------

## 数据格式
训练/验证数据使用 CSV，至少包含以下列：
- `article`: 源文本（待摘要的文章）
- `highlights`: 目标摘要文本

示例：
```csv
id,article,highlights
1,"这是一篇很长的文章内容...","这是对应的摘要"
2,"另一篇文章内容...","另一个摘要"
```

默认路径：
- 训练集：`data/train.csv`
- 验证集：`data/val.csv`

注意：词表由 `SimpleTokenizer` 使用训练集与验证集构建（包含 <SOS>/<EOS>）。首次加载会打印列名以便检查。

----------------------------------------

## 快速开始（main）
`main.py` 用于常规训练与评估：

```bash
python main.py \
  --train_path data/train.csv \
  --val_path data/val.csv \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --warmup_steps 1000 \
  --max_lr 2e-3 \
  --min_lr 1e-6 \
  --use_relative \
  --use_linear \
  --save_training_log \
  --save_plot
```

训练流程概览：
- 解析参数并打印训练配置
- 构建分词器与词表、加载 DataLoader
- 实例化 `Transformer`（支持相对位置、线性注意力、绝对位置编码）
- 初始化 `TrainingLogger`，记录配置与每轮指标
- 调用 `train()` 进行训练，保存最佳模型到 `--save_path`
- 训练结束计算验证集困惑度（perplexity），输出日志与图表

最佳模型默认保存至 `checkpoints/best_model.pt`（可通过 `--save_path` 修改）。

----------------------------------------

## 参数说明（main）
以下为关键参数及默认值（详见 `args_parser.py`）：

- 数据参数
  - `--train_path` 默认 `data/train.csv`
  - `--val_path` 默认 `data/val.csv`
- 训练参数
  - `--batch_size` 默认 `128`
  - `--epochs` 默认 `20`
  - `--lr` 默认 `1e-3`
  - `--gradient_accumulation_steps` 默认 `4`
  - `--patience` 默认 `5`（早停）
  - `--weight_decay` 默认 `1e-2`
  - `--dropout` 默认 `0.1`
  - `--clip_grad_norm` 默认 `1.0`
- 学习率调度（Warmup + Cosine）
  - `--warmup_steps` 默认 `1000`
  - `--max_lr` 默认 `2e-3`
  - `--min_lr` 默认 `1e-6`
- 优化器
  - `--beta1` 默认 `0.9`
  - `--beta2` 默认 `0.98`
  - `--eps` 默认 `1e-9`
- 模型结构
  - `--d_model` 默认 `256`
  - `--num_heads` 默认 `8`（需满足 `d_model % num_heads == 0`）
  - `--num_layers` 默认 `6`
  - `--d_ff` 默认 `1024`
  - `--max_src_len` 默认 `512`
  - `--max_tgt_len` 默认 `128`
  - `--use_relative` 启用相对位置编码（Ablation）
  - `--use_linear` 启用线性注意力（Ablation）
  - `--use_absolute_pe` 启用绝对位置编码（Ablation）
  - `--use_checkpointing` 启用梯度检查点（Ablation）
- 设备与日志
  - `--device` 默认自动选择（cuda/CPU）
  - `--save_path` 默认 `checkpoints/best_model.pt`
  - `--seed` 默认 `42`
  - `--log_dir` 默认 `logs`
  - `--experiment_name` 自定义实验名（日志目录名）
  - `--save_training_log` 保存 JSON/CSV 训练日志
  - `--log_interval` 默认 `100`
  - `--save_plot` 保存训练曲线图

调度器说明：`train.py` 中实现 Warmup + Cosine Annealing。学习率先预热到 `max_lr`、随后按余弦退火逐步下降到 `min_lr`。你也可以仅使用固定初始学习率 `lr`（优化器初值），但建议结合调度器获得更平滑的训练。

----------------------------------------

## 训练日志与可视化
`TrainingLogger` 会在 `logs/<experiment_name>/` 下保存：
- `training_log.json`：包含每轮 `train_loss`、`val_loss`、`learning_rate`、`epochs`、最佳轮与最佳验证损失、起止时间、总时长等
- `training_metrics.csv`：每轮指标的 CSV 便于后续分析
- `training_curves.png`：训练与验证曲线图
- `summary.json`：消融使用的训练总结（在 `run_ablation.py` 中写入）

日志示例字段（见仓库现有 `logs/*/training_log.json`）：
- `experiment_info`：名称、时间、状态（`completed`/`early_stopped`）
- `metrics`：`train_losses`、`val_losses`、`learning_rates`、`epochs`、`best_val_loss`、`best_epoch`
- `step_logs`：逐轮详细记录（含时间、单轮耗时）

----------------------------------------

## 消融实验（run_ablation）
`run_ablation.py` 提供多组可复现的配置：
- `baseline`：绝对位置编码 + 梯度检查点
- `baseline_plus_rel_pe`：在 baseline 基础上启用相对位置
- `relative_pe_only`：仅相对位置编码，不使用绝对位置
- `baseline_plus_linear_attn`：启用线性注意力
- `full_model`：相对位置 + 线性注意力 + 绝对位置编码 + 检查点
- `full_model_no_checkpoint`：与上类似但关闭梯度检查点

运行全部实验：
```bash
python run_ablation.py
```

仅运行指定实验：
```bash
python run_ablation.py --only relative_pe_only
```

使用完整数据集（默认会采样子集以便快速演示）：
```bash
python run_ablation.py --use_full_data
```

实验输出：每个实验会在 `logs/<exp_name>/` 下生成 `summary.json`、训练曲线与指标，并将最佳模型保存到 `models/<exp_name>_best_model.pt`。

学习率与轮数：
- 消融脚本内会对部分参数做上限限制（如 `epochs`、`d_model`、`num_layers`），以保证快速运行与资源可控
- 如需更长训练，可在 `run_ablation.py` 中提高上限或传参覆盖

----------------------------------------

## 结果汇总与分析
汇总消融结果：
```bash
python collect_results.py
```
该脚本会扫描 `logs/` 下各实验目录的 `summary.json` 并汇总为表格。

高级可视化：
- `visualizer.py` 与 `professional_ablation_results*` 目录提供更丰富的可视化与对比分析（热图、曲线、效率比较等）。

----------------------------------------

## 性能与稳定性建议
- 显存优化：降低 `--batch_size`，提高 `--gradient_accumulation_steps`，开启 `--use_linear`
- 训练效率：启用 `--warmup_steps` 与余弦退火；适度的 `--clip_grad_norm`；合理设置 `--patience`
- 模型质量：启用 `--use_relative`；逐步提高 `--d_model` / `--num_layers`；监控过拟合
- 截断长度：根据数据调整 `--max_src_len` 与 `--max_tgt_len`，降低无用计算

----------------------------------------

## 常见问题排查（FAQ）
- 报错“数据文件未找到”：检查 `--train_path` 与 `--val_path`；默认在 `data/` 目录
- CUDA 内存不足：减少 `--batch_size`、增加 `--gradient_accumulation_steps`、或启用 `--use_linear`
- 学习率曲线太平缓或太陡：调整 `--warmup_steps`、`--max_lr`、`--min_lr`；或使用更小初始 `--lr`
- 训练不收敛：降低学习率、增加预热步数、增大 `--patience`；检查数据清洗与列名是否正确
- 词表大小不一致：`main.py` 会用实际构建的词表覆盖 `args.vocab_size`；确保数据列名正确

----------------------------------------

## 参考与文件定位
- 主训练入口：`main.py`
- 消融入口：`run_ablation.py`
- 参数与默认值：`args_parser.py`
- 数据与分词：`data_utils.py`
- 模型结构：`transformer_enhanced.py`
- 训练与调度：`train.py`
- 日志记录：`training_logger.py`
- 汇总结果：`collect_results.py`
- 可视化：`visualizer.py`, `plot_training.py`

如需自定义实验或扩展模块，请从 `args_parser.py` 添加参数开始，并在 `main.py` 或 `run_ablation.py` 中接入。日志与模型保存路径均可通过命令行灵活配置。