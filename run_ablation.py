#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py

以最少改动的方式，运行增强版 Transformer 的几组消融实验：
- baseline
- baseline_plus_rel_pe
- relative_pe_only
- baseline_plus_linear_attn
- full_model
- full_model_no_checkpoint

复用现有的数据加载、模型、训练与日志模块；为每个实验写入 logs/<exp>/summary.json，
并将最佳模型分别保存到 models/<exp>_best_model.pt。
"""

import os
import json
import time
import traceback
import torch
import pandas as pd

from args_parser import get_default_args
from data_utils import get_dataloaders, SimpleTokenizer
# 假设 transformer_enhanced.py 和 train.py 已经是我们修复过的版本
from transformer_enhanced import Transformer
from train import train, compute_perplexity
from training_logger import TrainingLogger


# 定义要运行的消融实验配置（尽量保持简单）
EXPERIMENTS = [
    {
        'name': 'baseline',
        'flags': {
            'use_relative': False,
            'use_linear': False,
            'use_absolute_pe': True,
            'use_checkpointing': True,
        }
    },
    {
        'name': 'baseline_plus_rel_pe',
        'flags': {
            'use_relative': True,
            'use_linear': False,
            'use_absolute_pe': True,
            'use_checkpointing': True,
        }
    },
    {
        'name': 'relative_pe_only',
        'flags': {
            'use_relative': True,
            'use_linear': False,
            'use_absolute_pe': False,  # 仅相对位置编码，关闭绝对位置
            'use_checkpointing': True,
        }
    },
    {
        'name': 'baseline_plus_linear_attn',
        'flags': {
            'use_relative': False,
            'use_linear': True,
            'use_absolute_pe': True,
            'use_checkpointing': True,
        }
    },
    {
        'name': 'full_model',
        'flags': {
            'use_relative': True,
            'use_linear': True,
            'use_absolute_pe': True,
            'use_checkpointing': True,
        }
    },
    {
        'name': 'full_model_no_checkpoint',
        'flags': {
            'use_relative': True,
            'use_linear': True,
            'use_absolute_pe': True,
            'use_checkpointing': False,  # 关闭梯度检查点
        }
    },
]


def run_one_experiment(exp_name: str, flags: dict, use_full_data: bool = False) -> dict:
    """运行单个实验，并返回摘要信息字典。"""
    # 获取默认参数（避免修改原有解析流程）
    args = get_default_args()

    # 应用实验开关
    args.use_relative = bool(flags.get('use_relative', False))
    args.use_linear = bool(flags.get('use_linear', False))
    args.use_absolute_pe = bool(flags.get('use_absolute_pe', True))
    args.use_checkpointing = bool(flags.get('use_checkpointing', True))

    # 为每个实验使用独立的日志目录与模型保存路径
    args.experiment_name = exp_name
    os.makedirs('models', exist_ok=True)
    args.save_path = os.path.join('models', f'{exp_name}_best_model.pt')

    # 保存训练记录与图表
    args.save_training_log = True
    args.save_plot = True

    # 显存友好配置
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if isinstance(args.device, torch.device) and args.device.type == 'cuda':
        args.use_amp = True
    else:
        args.use_amp = False
    
    # 你的模型大小设置
    args.batch_size = min(getattr(args, 'batch_size', 64), 32)
    args.max_src_len = getattr(args, 'max_src_len', 768)
    args.max_tgt_len = getattr(args, 'max_tgt_len', 256)
    args.gradient_accumulation_steps = max(1, getattr(args, 'gradient_accumulation_steps', 1))
    args.epochs = min(getattr(args, 'epochs', 20), 20)
    args.vocab_size = getattr(args, 'vocab_size', 32000)
    
    # 你的 200M 模型结构
    args.d_model = 1024
    args.num_heads = 8 
    if args.d_model % args.num_heads != 0:
        for h in [8, 4, 2, 1]:
            if args.d_model % h == 0:
                args.num_heads = h
                break
    args.num_layers = 8
    args.d_ff = 1024
    
    
    # --- [FIX] 关键修复：更积极的 Warmup 策略 ---
    
    # 1. 保持安全的最大学习率
    args.max_lr = 1e-4  
    
    # 2. 保持最小学习率
    args.min_lr = 1e-5
    
    # 3. [修改] 设置一个更积极、标准的预热步数
    # 之前的 17946 步 (10%) 太长了，导致模型在低LR下"饿死"。
    # 我们将其缩短到 8000 步 (约 1 个 epoch)，让 LR 更快地爬升。
    args.warmup_steps = 8000
    
    # 4. 强制开启梯度裁剪 (必须)
    args.clip_grad_norm = 1.0 
    
    # --- [修复结束] ---
    
    
    # 采样小数据子集（可选）；如需使用完整数据集，传入 use_full_data=True
    if not use_full_data:
        try:
            train_df = pd.read_csv(args.train_path)
            val_df = pd.read_csv(args.val_path)
            # 保持列一致性
            train_small = train_df.sample(n=min(5000, len(train_df)), random_state=42)
            val_small = val_df.sample(n=min(1000, len(val_df)), random_state=42)
            os.makedirs(os.path.dirname(args.train_path), exist_ok=True)
            small_train_path = os.path.join(os.path.dirname(args.train_path), 'train_small.csv')
            small_val_path = os.path.join(os.path.dirname(args.val_path), 'val_small.csv')
            train_small.to_csv(small_train_path, index=False)
            val_small.to_csv(small_val_path, index=False)
            args.train_path = small_train_path
            args.val_path = small_val_path
            print("[INFO] 使用采样数据: train_small.csv / val_small.csv")
        except Exception as e:
            print(f"[WARN] 构建小数据集失败，使用原始数据集: {e}")
    else:
        print("[INFO] 使用完整数据集: data/train.csv / data.val.csv")

    # 构建数据与词表
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, pad_idx=args.pad_idx)
    train_loader, val_loader = get_dataloaders(args, tokenizer)

    # --- [FIX] 删除了自动调整 warmup_steps 的逻辑 ---
    # (确保这里没有那段检查 loader_size 和 total_steps 的代码)

    # 初始化模型（使用构建后的实际词表大小更稳妥）
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        dropout=args.dropout,
        use_relative=args.use_relative,
        use_linear=args.use_linear,
        use_absolute_pe=args.use_absolute_pe,
        use_checkpointing=args.use_checkpointing,
        pad_idx=args.pad_idx  # <--- [关键] 传入 pad_idx
    ).to(args.device)

    # 日志记录器（使用实验名）
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)

    # 训练
    # 清理显存碎片
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trained_model = train(model, train_loader, val_loader, args, logger)

    # 加载最佳权重（如有），计算验证困惑度
    try:
        # 使用安全加载，避免未来版本的反序列化安全警告
        try:
            trained_model.load_state_dict(torch.load(args.save_path, map_location=args.device, weights_only=True))
        except TypeError:
            # 兼容旧版 PyTorch 无 weights_only 参数
            trained_model.load_state_dict(torch.load(args.save_path, map_location=args.device))
    except Exception:
        # 未能加载最佳模型时，继续使用当前权重计算困惑度
        pass

    final_ppl = compute_perplexity(trained_model, val_loader, pad_idx=args.pad_idx)

    # 汇总信息并写入 summary.json（collect_results.py 会读取）
    summary = logger.get_summary()
    summary['final_perplexity'] = final_ppl
    summary['flags'] = flags
    # 兼容 collect_results.py 期望的字段名
    if 'training_time' in summary and summary['training_time'] is not None:
        summary['total_training_time_s'] = summary['training_time']
    summary_path = os.path.join(logger.experiment_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def run_all_experiments() -> None:
    os.makedirs('logs', exist_ok=True)
    results_status = {}

    for cfg in EXPERIMENTS:
        name = cfg['name']
        flags = cfg['flags']
        print(f"\n[RUN] Experiment: {name} | Flags: {flags}")
        try:
            summary = run_one_experiment(name, flags)
            # 标记为 COMPLETED（与已有 ablation_summary 文件风格保持一致）
            results_status[name] = 'COMPLETED'
            print(f"[DONE] {name} | Best Val Loss: {summary.get('best_val_loss')} | PPL: {summary.get('final_perplexity')}")
        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            traceback.print_exc()
            results_status[name] = 'FAILED'

    # 写入一次总览 JSON，便于快速查看整体结果状态
    ts = time.strftime('%Y%m%d_%H%M%S')
    overview_path = os.path.join('logs', f'ablation_summary_{ts}.json')
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(results_status, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] Ablation overview saved to: {overview_path}")
    print("如需更详细的汇总，请运行: python collect_results.py")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation experiments or a single one')
    parser.add_argument('--only', type=str, default=None, help='Only run the named experiment')
    
    # [FIX] 修复了拼写错误: parser.add.argument -> parser.add_argument
    parser.add_argument('--use_full_data', action='store_true', help='Use full data without sampling')
    
    args_cli = parser.parse_args()

    if args_cli.only is None:
        run_all_experiments()
    else:
        # 查找匹配的实验名并仅运行该实验
        match = None
        for cfg in EXPERIMENTS:
            if cfg['name'] == args_cli.only:
                match = cfg
                break
        if match is None:
            print(f"[ERROR] 未找到实验: {args_cli.only}. 可选: {[c['name'] for c in EXPERIMENTS]}")
        else:
            try:
                summary = run_one_experiment(match['name'], match['flags'], use_full_data=args_cli.use_full_data)
                print(f"[DONE] {match['name']} | Best Val Loss: {summary.get('best_val_loss')} | PPL: {summary.get('final_perplexity')}")
            except Exception as e:
                print(f"[FAILED] {match['name']}: {e}")
                traceback.print_exc()