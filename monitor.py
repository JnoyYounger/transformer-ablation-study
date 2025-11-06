#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控和资源使用统计模块
实现实时监控GPU使用率、显存占用、训练损失等指标
"""

import os
import time
import json
import threading
import subprocess
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class GPUMonitor:
    """GPU资源监控器"""
    
    def __init__(self, gpu_ids: List[int], log_interval: int = 10):
        """
        初始化GPU监控器
        
        Args:
            gpu_ids: 要监控的GPU设备ID列表
            log_interval: 监控间隔（秒）
        """
        self.gpu_ids = gpu_ids
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_stats = defaultdict(lambda: {
            'timestamps': deque(maxlen=1000),
            'memory_used': deque(maxlen=1000),
            'memory_total': deque(maxlen=1000),
            'utilization': deque(maxlen=1000),
            'temperature': deque(maxlen=1000),
            'power_draw': deque(maxlen=1000)
        })
    
    def get_gpu_info(self, gpu_id: int) -> Dict[str, Any]:
        """获取指定GPU的实时信息"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits',
                f'--id={gpu_id}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'memory_used': float(values[0]),
                    'memory_total': float(values[1]),
                    'utilization': float(values[2]),
                    'temperature': float(values[3]),
                    'power_draw': float(values[4]) if values[4] != '[Not Supported]' else 0.0,
                    'memory_utilization': (float(values[0]) / float(values[1])) * 100
                }
        except Exception as e:
            print(f"获取GPU {gpu_id} 信息失败: {e}")
        
        return {
            'memory_used': 0.0,
            'memory_total': 0.0,
            'utilization': 0.0,
            'temperature': 0.0,
            'power_draw': 0.0,
            'memory_utilization': 0.0
        }
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            timestamp = datetime.now()
            
            for gpu_id in self.gpu_ids:
                gpu_info = self.get_gpu_info(gpu_id)
                stats = self.gpu_stats[gpu_id]
                
                stats['timestamps'].append(timestamp)
                stats['memory_used'].append(gpu_info['memory_used'])
                stats['memory_total'].append(gpu_info['memory_total'])
                stats['utilization'].append(gpu_info['utilization'])
                stats['temperature'].append(gpu_info['temperature'])
                stats['power_draw'].append(gpu_info['power_draw'])
            
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print(f"开始监控GPU: {self.gpu_ids}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("GPU监控已停止")
    
    def get_current_stats(self) -> Dict[int, Dict[str, float]]:
        """获取当前GPU统计信息"""
        current_stats = {}
        for gpu_id in self.gpu_ids:
            current_stats[gpu_id] = self.get_gpu_info(gpu_id)
        return current_stats
    
    def get_historical_stats(self, gpu_id: int, minutes: int = 10) -> Dict[str, List]:
        """获取历史统计数据"""
        if gpu_id not in self.gpu_stats:
            return {}
        
        stats = self.gpu_stats[gpu_id]
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        # 过滤指定时间范围内的数据
        filtered_data = {
            'timestamps': [],
            'memory_used': [],
            'memory_total': [],
            'utilization': [],
            'temperature': [],
            'power_draw': []
        }
        
        for i, timestamp in enumerate(stats['timestamps']):
            if timestamp.timestamp() >= cutoff_time:
                filtered_data['timestamps'].append(timestamp)
                filtered_data['memory_used'].append(stats['memory_used'][i])
                filtered_data['memory_total'].append(stats['memory_total'][i])
                filtered_data['utilization'].append(stats['utilization'][i])
                filtered_data['temperature'].append(stats['temperature'][i])
                filtered_data['power_draw'].append(stats['power_draw'][i])
        
        return filtered_data

class ProcessMonitor:
    """进程资源监控器"""
    
    def __init__(self, log_interval: int = 10):
        """
        初始化进程监控器
        
        Args:
            log_interval: 监控间隔（秒）
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.process_stats = defaultdict(lambda: {
            'timestamps': deque(maxlen=1000),
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'memory_rss': deque(maxlen=1000),
            'io_read': deque(maxlen=1000),
            'io_write': deque(maxlen=1000)
        })
        self.tracked_pids = set()
    
    def add_process(self, pid: int):
        """添加要监控的进程"""
        self.tracked_pids.add(pid)
        print(f"开始监控进程 PID: {pid}")
    
    def remove_process(self, pid: int):
        """移除监控的进程"""
        self.tracked_pids.discard(pid)
        print(f"停止监控进程 PID: {pid}")
    
    def get_process_info(self, pid: int) -> Dict[str, Any]:
        """获取进程信息"""
        try:
            process = psutil.Process(pid)
            
            # 获取IO统计（如果支持）
            try:
                io_counters = process.io_counters()
                io_read = io_counters.read_bytes
                io_write = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                io_read = 0
                io_write = 0
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_rss': process.memory_info().rss / 1024 / 1024,  # MB
                'io_read': io_read,
                'io_write': io_write,
                'status': process.status(),
                'create_time': process.create_time()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            timestamp = datetime.now()
            
            # 清理已结束的进程
            dead_pids = set()
            for pid in self.tracked_pids:
                if not psutil.pid_exists(pid):
                    dead_pids.add(pid)
            
            for pid in dead_pids:
                self.tracked_pids.remove(pid)
            
            # 监控活跃进程
            for pid in self.tracked_pids:
                process_info = self.get_process_info(pid)
                if process_info:
                    stats = self.process_stats[pid]
                    
                    stats['timestamps'].append(timestamp)
                    stats['cpu_percent'].append(process_info['cpu_percent'])
                    stats['memory_percent'].append(process_info['memory_percent'])
                    stats['memory_rss'].append(process_info['memory_rss'])
                    stats['io_read'].append(process_info['io_read'])
                    stats['io_write'].append(process_info['io_write'])
            
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("开始进程监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("进程监控已停止")

class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, experiment_id: str, log_dir: str):
        """
        初始化训练监控器
        
        Args:
            experiment_id: 实验ID
            log_dir: 日志目录
        """
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"{experiment_id}_training.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_id}_metrics.json")
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 训练指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'steps': [],
            'timestamps': [],
            'epoch': []
        }
        
        # 资源使用记录
        self.resource_usage = {
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_percent': [],
            'memory_rss': [],
            'timestamps': []
        }
    
    def log_metrics(self, step: int, epoch: int, train_loss: float, 
                   val_loss: Optional[float] = None, val_accuracy: Optional[float] = None,
                   learning_rate: Optional[float] = None):
        """记录训练指标"""
        timestamp = datetime.now()
        
        self.metrics['steps'].append(step)
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['timestamps'].append(timestamp.isoformat())
        
        # 写入日志文件
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'step': step,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': learning_rate
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_resource_usage(self, gpu_memory: float, gpu_utilization: float,
                          cpu_percent: float, memory_rss: float):
        """记录资源使用情况"""
        timestamp = datetime.now()
        
        self.resource_usage['gpu_memory'].append(gpu_memory)
        self.resource_usage['gpu_utilization'].append(gpu_utilization)
        self.resource_usage['cpu_percent'].append(cpu_percent)
        self.resource_usage['memory_rss'].append(memory_rss)
        self.resource_usage['timestamps'].append(timestamp.isoformat())
    
    def save_metrics(self):
        """保存指标到文件"""
        combined_data = {
            'experiment_id': self.experiment_id,
            'metrics': self.metrics,
            'resource_usage': self.resource_usage,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    def get_latest_metrics(self, n: int = 10) -> Dict[str, List]:
        """获取最新的n个指标"""
        latest_metrics = {}
        for key, values in self.metrics.items():
            if values:
                latest_metrics[key] = values[-n:]
            else:
                latest_metrics[key] = []
        return latest_metrics

class MonitoringSystem:
    """综合监控系统"""
    
    def __init__(self, gpu_ids: List[int], log_dir: str = "monitoring_logs"):
        """
        初始化监控系统
        
        Args:
            gpu_ids: 要监控的GPU设备ID列表
            log_dir: 监控日志目录
        """
        self.gpu_ids = gpu_ids
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化各个监控器
        self.gpu_monitor = GPUMonitor(gpu_ids)
        self.process_monitor = ProcessMonitor()
        
        # 训练监控器字典
        self.training_monitors = {}
        
        # 系统状态
        self.monitoring_active = False
    
    def start_monitoring(self):
        """启动所有监控"""
        if not self.monitoring_active:
            self.gpu_monitor.start_monitoring()
            self.process_monitor.start_monitoring()
            self.monitoring_active = True
            print("监控系统已启动")
    
    def stop_monitoring(self):
        """停止所有监控"""
        if self.monitoring_active:
            self.gpu_monitor.stop_monitoring()
            self.process_monitor.stop_monitoring()
            
            # 保存所有训练监控器的数据
            for monitor in self.training_monitors.values():
                monitor.save_metrics()
            
            self.monitoring_active = False
            print("监控系统已停止")
    
    def add_training_experiment(self, experiment_id: str, process_pid: int):
        """添加训练实验监控"""
        # 创建训练监控器
        training_monitor = TrainingMonitor(experiment_id, self.log_dir)
        self.training_monitors[experiment_id] = training_monitor
        
        # 添加进程监控
        self.process_monitor.add_process(process_pid)
        
        print(f"添加实验监控: {experiment_id} (PID: {process_pid})")
    
    def remove_training_experiment(self, experiment_id: str, process_pid: int):
        """移除训练实验监控"""
        if experiment_id in self.training_monitors:
            # 保存并移除训练监控器
            self.training_monitors[experiment_id].save_metrics()
            del self.training_monitors[experiment_id]
        
        # 移除进程监控
        self.process_monitor.remove_process(process_pid)
        
        print(f"移除实验监控: {experiment_id} (PID: {process_pid})")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'gpu_stats': self.gpu_monitor.get_current_stats(),
            'active_experiments': list(self.training_monitors.keys()),
            'tracked_processes': len(self.process_monitor.tracked_pids)
        }
        
        return status
    
    def generate_status_report(self) -> str:
        """生成状态报告"""
        status = self.get_system_status()
        
        report = []
        report.append("=" * 60)
        report.append("监控系统状态报告")
        report.append("=" * 60)
        report.append(f"时间: {status['timestamp']}")
        report.append(f"监控状态: {'活跃' if status['monitoring_active'] else '停止'}")
        report.append(f"活跃实验: {len(status['active_experiments'])}")
        report.append(f"监控进程: {status['tracked_processes']}")
        report.append("")
        
        # GPU状态
        report.append("GPU状态:")
        for gpu_id, gpu_info in status['gpu_stats'].items():
            report.append(f"  GPU {gpu_id}:")
            report.append(f"    显存使用: {gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f} MB ({gpu_info['memory_utilization']:.1f}%)")
            report.append(f"    GPU使用率: {gpu_info['utilization']:.1f}%")
            report.append(f"    温度: {gpu_info['temperature']:.1f}°C")
            if gpu_info['power_draw'] > 0:
                report.append(f"    功耗: {gpu_info['power_draw']:.1f}W")
        
        # 活跃实验
        if status['active_experiments']:
            report.append("")
            report.append("活跃实验:")
            for exp_id in status['active_experiments']:
                report.append(f"  - {exp_id}")
        
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """测试监控系统"""
    # 测试GPU监控
    gpu_monitor = GPUMonitor([0])
    gpu_monitor.start_monitoring()
    
    print("监控5秒...")
    time.sleep(5)
    
    # 获取当前状态
    current_stats = gpu_monitor.get_current_stats()
    print("当前GPU状态:")
    for gpu_id, stats in current_stats.items():
        print(f"GPU {gpu_id}: {stats}")
    
    gpu_monitor.stop_monitoring()
    
    # 测试综合监控系统
    print("\n测试综合监控系统...")
    monitoring_system = MonitoringSystem([0])
    monitoring_system.start_monitoring()
    
    # 模拟添加实验
    monitoring_system.add_training_experiment("test_exp_1", os.getpid())
    
    time.sleep(3)
    
    # 生成状态报告
    print("\n" + monitoring_system.generate_status_report())
    
    monitoring_system.stop_monitoring()

if __name__ == "__main__":
    main()