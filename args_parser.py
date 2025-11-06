# args_parser.py
import argparse
import torch
import os

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='Transformer训练参数配置')
    
    # 数据路径参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--train_path', type=str, default='data/train.csv',
                            help='训练数据路径')
    data_group.add_argument('--val_path', type=str, default='data/val.csv',
                            help='验证数据路径')
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--vocab_size', type=int, default=32000,
                             help='词汇表大小')
    train_group.add_argument('--pad_idx', type=int, default=0,
                             help='填充token索引')
    train_group.add_argument('--batch_size', type=int, default=128,
                             help='批次大小')
    train_group.add_argument('--epochs', type=int, default=20,
                             help='训练轮数')
    train_group.add_argument('--lr', type=float, default=1e-3,
                             help='初始学习率')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=4,
                             help='梯度累积步数')
    train_group.add_argument('--patience', type=int, default=5,
                             help='早停耐心值')
    train_group.add_argument('--weight_decay', type=float, default=1e-2,
                             help='权重衰减')
    train_group.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout率')
    train_group.add_argument('--clip_grad_norm', type=float, default=1.0,
                             help='梯度裁剪阈值')
    
    # 学习率调度参数
    lr_group = parser.add_argument_group('学习率调度参数')
    lr_group.add_argument('--warmup_steps', type=int, default=1000,
                          help='预热步数')
    lr_group.add_argument('--max_lr', type=float, default=2e-3,
                          help='最大学习率')
    lr_group.add_argument('--min_lr', type=float, default=1e-6,
                          help='最小学习率')
    
    # 优化器参数
    opt_group = parser.add_argument_group('优化器参数')
    opt_group.add_argument('--beta1', type=float, default=0.9,
                           help='Adam优化器beta1参数')
    opt_group.add_argument('--beta2', type=float, default=0.98,
                           help='Adam优化器beta2参数')
    opt_group.add_argument('--eps', type=float, default=1e-9,
                           help='Adam优化器epsilon参数')
    
    # Transformer模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--d_model', type=int, default=256,
                             help='模型维度')
    model_group.add_argument('--num_heads', type=int, default=8,
                             help='注意力头数')
    model_group.add_argument('--num_layers', type=int, default=6,
                             help='Transformer层数')
    model_group.add_argument('--d_ff', type=int, default=1024,
                             help='前馈网络维度')
    model_group.add_argument('--max_src_len', type=int, default=512,
                             help='源序列最大长度')
    model_group.add_argument('--max_tgt_len', type=int, default=128,
                             help='目标序列最大长度')
    model_group.add_argument('--use_relative', action='store_true',
                             help='[Ablation] 是否使用相对位置编码')
    model_group.add_argument('--use_linear', action='store_true',
                             help='[Ablation] 是否使用线性注意力')
    # --- [新增] 消融实验参数 ---
    model_group.add_argument('--use_absolute_pe', action='store_true',
                             help='[Ablation] 是否使用绝对位置编码')
    model_group.add_argument('--use_checkpointing', action='store_true',
                             help='[Ablation] 是否使用梯度检查点')
    # --------------------------

    # 设备和保存参数
    misc_group = parser.add_argument_group('其他参数')
    misc_group.add_argument('--device', type=str, 
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='训练设备')
    misc_group.add_argument('--save_path', type=str, default='checkpoints/best_model.pt',
                            help='模型保存路径')
    misc_group.add_argument('--seed', type=int, default=42,
                            help='随机种子')
    
    # 训练记录参数
    misc_group.add_argument('--log_dir', type=str, default='logs', help='训练日志保存目录')
    # --- [新增] 实验名称 ---
    misc_group.add_argument('--experiment_name', type=str, default=None,
                            help='实验名称 (用于日志记录)')
    # -----------------------
    misc_group.add_argument('--save_training_log', action='store_true', help='是否保存训练记录')
    misc_group.add_argument('--log_interval', type=int, default=100, help='日志记录间隔（步数）')
    misc_group.add_argument('--save_plot', action='store_true', help='是否保存训练曲线图')
    
    return parser

def validate_args(args):
    """验证参数的合理性"""
    errors = []
    
    # (移除了 os.path.exists 检查，让 pd.read_csv 在 get_dataloaders 中处理)
    
    # 验证数值参数范围
    if args.vocab_size <= 0:
        errors.append("词汇表大小必须大于0")
    # ... (你其他的验证逻辑都很好，保留它们) ...
    if args.d_model % args.num_heads != 0:
        errors.append("模型维度必须能被注意力头数整除")
    
    # 验证设备
    if args.device not in ['cpu', 'cuda']:
        if not args.device.startswith('cuda:'):
            errors.append("设备必须是 'cpu', 'cuda' 或 'cuda:N'")
    
    # 创建保存目录
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"无法创建保存目录 {save_dir}: {e}")
    
    # 创建日志目录 (TrainingLogger 已经处理了，但双重检查无害)
    if hasattr(args, 'save_training_log') and args.save_training_log and not os.path.exists(args.log_dir):
        try:
            os.makedirs(args.log_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"无法创建日志目录 {args.log_dir}: {e}")
    
    if errors:
        raise ValueError("参数验证失败:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# args_parser.py

def parse_args():
    """解析并验证命令行参数"""
    parser = create_parser()
    args = parser.parse_args() # <--- 此时 args.device 还是字符串
    
    # 验证 (先用字符串进行验证)
    validate_args(args) 
    
    # --- [修改] ---
    # 验证通过后，再转换为 torch.device 对象
    args.device = torch.device(args.device) 
    
    return args

def get_default_args():
    """获取默认参数（用于测试或程序内部调用）"""
    parser = create_parser()
    args = parser.parse_args([])  # 空参数列表，使用所有默认值
    args.device = torch.device(args.device)
    # 注意：validate_args 可能会失败，如果默认的 'data/train.csv' 不存在
    # validate_args(args) # 在 get_default_args 中最好跳过验证
    return args

if __name__ == "__main__":
    # 测试参数解析
    try:
        args = parse_args()
        print("参数解析成功!")
        print(f"训练数据路径: {args.train_path}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.lr}")
        print(f"模型维度: {args.d_model}")
        print(f"设备: {args.device}")
        print(f"实验名称: {args.experiment_name}")
        print(f"使用相对位置: {args.use_relative}")
        print(f"使用绝对位置: {args.use_absolute_pe}")

    except ValueError as e:
        print(e)
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")