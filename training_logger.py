#!/usr/bin/env python3
"""
训练记录保存模块
用于记录和保存训练过程中的各种指标和信息
"""

import json
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

class TrainingLogger:
    """训练记录器"""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        初始化训练记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        
        # 创建实验目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化记录
        self.training_log = {
            'experiment_info': {
                'name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'total_training_time': None,
                'status': 'running'
            },
            'config': {},
            'metrics': {
                'train_losses': [],
                'val_losses': [],
                'learning_rates': [],
                'epochs': [],
                'best_val_loss': float('inf'),
                'best_epoch': -1
            },
            'step_logs': []
        }
        
        self.start_time = time.time()
        
    def log_config(self, config: Dict[str, Any]):
        """记录训练配置"""
        self.training_log['config'] = config
        self._save_log()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, epoch_time: float = None):
        """记录每个epoch的指标"""
        self.training_log['metrics']['epochs'].append(epoch)
        self.training_log['metrics']['train_losses'].append(train_loss)
        self.training_log['metrics']['val_losses'].append(val_loss)
        self.training_log['metrics']['learning_rates'].append(learning_rate)
        
        # 更新最佳模型记录
        if val_loss < self.training_log['metrics']['best_val_loss']:
            self.training_log['metrics']['best_val_loss'] = val_loss
            self.training_log['metrics']['best_epoch'] = epoch
            
        # 记录详细信息
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        if epoch_time is not None:
            epoch_info['epoch_time'] = epoch_time
            
        self.training_log['step_logs'].append(epoch_info)
        self._save_log()
        
    def log_step(self, step: int, loss: float, learning_rate: float):
        """记录训练步骤（可选，用于详细记录）"""
        step_info = {
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        # 这里可以选择是否保存每步的详细信息
        pass
        
    def finish_training(self, status: str = 'completed'):
        """完成训练记录"""
        end_time = time.time()
        self.training_log['experiment_info']['end_time'] = datetime.now().isoformat()
        self.training_log['experiment_info']['total_training_time'] = end_time - self.start_time
        self.training_log['experiment_info']['status'] = status
        self._save_log()
        
    def save_plots(self):
        """保存训练曲线图"""
        if not self.training_log['metrics']['train_losses']:
            print("[WARNING] 没有训练数据可以绘图")
            return
            
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.training_log['metrics']['epochs']
        train_losses = self.training_log['metrics']['train_losses']
        val_losses = self.training_log['metrics']['val_losses']
        learning_rates = self.training_log['metrics']['learning_rates']
        
        # 损失曲线
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
        ax1.plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2.plot(epochs, learning_rates, label='Learning Rate', color='green', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.experiment_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] 训练曲线已保存到: {plot_path}")
        
    def save_csv(self):
        """保存训练数据为CSV格式"""
        if not self.training_log['metrics']['train_losses']:
            return
            
        # 创建DataFrame
        df = pd.DataFrame({
            'epoch': self.training_log['metrics']['epochs'],
            'train_loss': self.training_log['metrics']['train_losses'],
            'val_loss': self.training_log['metrics']['val_losses'],
            'learning_rate': self.training_log['metrics']['learning_rates']
        })
        
        # 保存CSV
        csv_path = os.path.join(self.experiment_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"[INFO] 训练指标已保存到: {csv_path}")
        
    def _save_log(self):
        """保存日志到JSON文件"""
        log_path = os.path.join(self.experiment_dir, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
            
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        metrics = self.training_log['metrics']
        return {
            'experiment_name': self.experiment_name,
            'total_epochs': len(metrics['epochs']),
            'best_val_loss': metrics['best_val_loss'],
            'best_epoch': metrics['best_epoch'],
            'final_train_loss': metrics['train_losses'][-1] if metrics['train_losses'] else None,
            'final_val_loss': metrics['val_losses'][-1] if metrics['val_losses'] else None,
            'training_time': self.training_log['experiment_info'].get('total_training_time'),
            'status': self.training_log['experiment_info']['status']
        }
        
    def print_summary(self):
        """打印训练摘要"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("训练摘要")
        print("="*50)
        print(f"实验名称: {summary['experiment_name']}")
        print(f"总训练轮数: {summary['total_epochs']}")
        print(f"最佳验证损失: {summary['best_val_loss']:.4f}")
        print(f"最佳轮数: {summary['best_epoch']}")
        if summary['final_train_loss']:
            print(f"最终训练损失: {summary['final_train_loss']:.4f}")
        if summary['final_val_loss']:
            print(f"最终验证损失: {summary['final_val_loss']:.4f}")
        if summary['training_time']:
            print(f"训练时间: {summary['training_time']:.2f}秒")
        print(f"状态: {summary['status']}")
        print("="*50)