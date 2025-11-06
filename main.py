# main.py
import os
import torch
import numpy as np

# --- 导入你的自定义模块 ---
from args_parser import parse_args
from data_utils import get_dataloaders, SimpleTokenizer
from transformer_enhanced import Transformer
from train import train, compute_perplexity
from training_logger import TrainingLogger
# --------------------------

def main():
    # 解析命令行参数 (使用你的 args_parser.py)
    args = parse_args()
    print("=" * 50)
    print("训练配置参数:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # 针对多GPU

    # -----------------------
    # Tokenizer (使用你的 data_utils.py)
    # -----------------------
    print("[INFO] 初始化 Tokenizer...")
    # 注意：你的 SimpleTokenizer 需要先 build_vocab
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, pad_idx=args.pad_idx)

    # -----------------------
    # DataLoader (使用你的 data_utils.py)
    # -----------------------
    print("[INFO] 加载数据并构建词表...")
    try:
        train_loader, val_loader = get_dataloaders(args, tokenizer)
        print(f"[INFO] 词汇表构建完毕，大小: {len(tokenizer.word2idx)}")
        # 更新 args 中的 vocab_size，以防万一
        args.vocab_size = len(tokenizer.word2idx)
        print(f"[INFO] 真实词汇表大小更新为: {args.vocab_size}")

    except FileNotFoundError as e:
        print(f"[ERROR] 数据文件未找到: {e}")
        print("请确保 'data/train.csv' 和 'data/val.csv' 存在，")
        print("或者通过 --train_path 和 --val_path 指定正确路径。")
        return
    except Exception as e:
        print(f"[ERROR] 加载数据时出错: {e}")
        return

    # -----------------------
    # Model (使用 transformer_enhanced.py)
    # -----------------------
    print("[INFO] 初始化 Transformer 模型...")
    model = Transformer(
        vocab_size=args.vocab_size, # 使用构建后的真实词表大小
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        dropout=args.dropout,
        use_relative=args.use_relative,
        use_linear=args.use_linear,
        use_absolute_pe=args.use_absolute_pe,       # (来自修改后的 args_parser)
        use_checkpointing=args.use_checkpointing   # (来自修改后的 args_parser)
    ).to(args.device)

    # -----------------------
    # Logger (使用你的 training_logger.py)
    # -----------------------
    print("[INFO] 初始化 Logger...")
    # TrainingLogger 会自动使用时间戳（如果 experiment_name 为 None）
    # 在 run_ablation.py 中，我们会指定 experiment_name
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)
    
    # 记录配置
    # 将 Namespace 转换为字典，并处理 torch.device
    config_dict = vars(args)
    config_dict['device'] = str(args.device)
    logger.log_config(config_dict)

    # -----------------------
    # 训练模型 (使用修改后的 train.py)
    # -----------------------
    model = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        args=args, 
        logger=logger  # <--- 传入你的 logger 实例
    )

    # -----------------------
    # Save best model
    # -----------------------
    # (train.py 内部已经保存了最佳模型到 args.save_path)
    print(f"[INFO] 最佳模型已在训练期间保存到 {args.save_path}")

    # -----------------------
    # Compute Perplexity
    # -----------------------
    print("[INFO] 加载最佳模型计算最终困惑度...")
    try:
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        compute_perplexity(model, val_loader, args.pad_idx) # <--- 传入 pad_idx
    except FileNotFoundError:
        print(f"[ERROR] 无法加载最佳模型: {args.save_path} 未找到。")
    except Exception as e:
        print(f"[ERROR] 加载模型或计算 PPL 时出错: {e}")

    # -----------------------
    # Plot training curve
    # -----------------------
    # (train.py 内部已调用 logger.save_plots() 和 logger.save_csv())
    print(f"[INFO] 训练日志和图表已保存到: {logger.experiment_dir}")

if __name__ == "__main__":
    main()