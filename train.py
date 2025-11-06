# train.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
import time
import argparse  # (来自之前的修复)
# 统一使用新的 AMP API，避免弃用警告
# torch>=2.0 推荐使用 torch.amp.autocast 与 torch.amp.GradScaler

# 允许TF32与cudnn benchmark以提升吞吐、降低精度要求下的显存压力
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

from training_logger import TrainingLogger # 导入你的 Logger
from torch.utils.data import DataLoader

class WarmupCosineScheduler:
    """学习率预热 + 余弦退火调度器"""
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps) # 避免除零
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            if self.warmup_steps == 0:
                 lr = self.max_lr
            else:
                lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr = max(lr, self.min_lr) # 确保不低于 min_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          args: argparse.Namespace, 
          logger: TrainingLogger):
    """
    模型训练函数
    """

    device = args.device
    model = model.to(device)

    # ---------- 参数统计 ----------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params}")
    print(f"[INFO] Trainable parameters: {trainable_params}")

    # ---------- 优化器与损失 ----------
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.min_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)

    # 计算总训练步数
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    if total_steps <= args.warmup_steps:
        print(f"[WARNING] Total steps ({total_steps}) <= warmup steps ({args.warmup_steps}).")
        new_warmup = max(1, int(total_steps * 0.1))
        print(f"[WARNING] Warmup steps 自动调整为: {new_warmup}")
        args.warmup_steps = new_warmup
    
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_steps, total_steps, args.max_lr, args.min_lr
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
    # 在 CUDA 上启用 AMP；CPU 上禁用
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[INFO] Training started on {device} for {args.epochs} epochs (AMP={'on' if use_amp else 'off'})")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        steps = 0
        optimizer.zero_grad()

        # tqdm 现在会正确地写入 stderr
        pbar = tqdm(train_loader, 
                    desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            # 新版 autocast 需要指定设备类型
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs, targets[:, :-1]) # [B, T-1, V]
                # --- [FIX] 这里是 -1 (数字1) ---
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
                # --- [FIX END] ---

            loss = loss / args.gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                current_lr = scheduler.step()
                optimizer.zero_grad()
                steps += 1
                total_loss += loss.item() * args.gradient_accumulation_steps

                pbar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        avg_train_loss = total_loss / max(1, steps)
        
        # ---------- Validation ----------
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, 
                                        desc="Validation", 
                                        leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(inputs, targets[:, :-1])
                    # --- [FIX] 这里是 -1 (数字1) ---
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
                    # --- [FIX END] ---

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # 这个 print 会被 run_experiment.py 捕获并打印到 stdout
        print(f"[INFO] Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LR = {current_lr:.2e}, Time = {epoch_time:.2f}s")

        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time
        )

        # ---------- 模型保存 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.save_path)
            epochs_no_improve = 0
            print(f"  [CHECKPOINT] New best model saved: {args.save_path} (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [INFO] No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.patience:
                print(f"  [EARLY STOP] Stopping training at epoch {epoch + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    print(f"[INFO] Training completed. Best Val Loss: {best_val_loss:.4f}")

    status = 'early_stopped' if epochs_no_improve >= args.patience else 'completed'
    logger.finish_training(status)
    
    if args.save_training_log:
        logger.save_csv()
    if args.save_plot:
        logger.save_plots()
        
    logger.print_summary()
    
    return model


def compute_perplexity(model: nn.Module, val_loader: DataLoader, pad_idx: int):
    device = next(model.parameters()).device
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)
    total_loss = 0
    total_tokens = 0
    use_amp_eval = device.type == 'cuda'

    with torch.no_grad():
        pbar_desc = "Computing Perplexity"
        for inputs, targets in tqdm(val_loader, desc=pbar_desc):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp_eval):
                outputs = model(inputs, targets[:, :-1])
                # --- [FIX] 这里是 -1 (数字1) ---
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
                # --- [FIX END] ---
            
            total_loss += loss.item()
            
            non_pad_tokens = (targets[:, 1:] != pad_idx).sum().item()
            total_tokens += non_pad_tokens

    if total_tokens == 0:
        print("[ERROR] No non-pad tokens found. Cannot compute perplexity.")
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"[INFO] Validation Perplexity = {perplexity:.4f}")
    return perplexity