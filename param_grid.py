#!/usr/bin/env python3
"""
参数网格搜索配置
定义所有测试参数组合和实验配置
"""

import itertools
import json
import os
from typing import List, Dict, Any
import hashlib

class ParameterGrid:
    """参数网格生成器"""
    
    def __init__(self):
        """初始化参数网格"""
        # 基础固定参数
        self.base_params = {
            'train_path': 'data/train.csv',
            'val_path': 'data/val.csv',
            'vocab_size': 10000,
            'max_src_len': 128,
            'max_tgt_len': 128,
            'pad_idx': 0,
            'seed': 42,
            'device': 'cuda',
            'save_training_log': True,
            'save_plot': True,
            'log_interval': 100,
            'gradient_accumulation_steps': 4,
            'clip_grad_norm': 1.0,
            'use_relative': True,
            'use_linear': True,
            'beta1': 0.9,
            'beta2': 0.98,
            'eps': 1e-9,
            'patience': 10,
            'epochs': 100,  # 足够大的值，通过steps控制
            'max_steps': 10000  # 每组参数运行10000步
        }
        
        # 可变参数网格
        self.param_grid = {
            # 模型结构参数
            'num_heads': [4, 8, 16],
            'd_model': [512, 768, 1024],
            'num_layers': [6, 12, 24],
            
            # 训练超参数
            'lr': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
            'batch_size': [32, 64, 128],
            'warmup_steps': [1000, 4000],
            
            # 正则化参数
            'dropout': [0.1, 0.2, 0.3],
            'weight_decay': [0.001, 0.01]
        }
        
        # 预定义的优化参数组合（减少搜索空间）
        self.optimized_combinations = [
            # 小模型配置
            {
                'num_heads': 4,
                'd_model': 512,
                'num_layers': 6,
                'batch_size': 128,
                'lr': 3e-4,
                'warmup_steps': 1000,
                'dropout': 0.1,
                'weight_decay': 0.01
            },
            {
                'num_heads': 8,
                'd_model': 512,
                'num_layers': 6,
                'batch_size': 64,
                'lr': 2e-4,
                'warmup_steps': 1000,
                'dropout': 0.2,
                'weight_decay': 0.001
            },
            # 中等模型配置
            {
                'num_heads': 8,
                'd_model': 768,
                'num_layers': 12,
                'batch_size': 64,
                'lr': 2e-4,
                'warmup_steps': 4000,
                'dropout': 0.1,
                'weight_decay': 0.01
            },
            {
                'num_heads': 12,
                'd_model': 768,
                'num_layers': 12,
                'batch_size': 32,
                'lr': 1e-4,
                'warmup_steps': 4000,
                'dropout': 0.2,
                'weight_decay': 0.001
            },
            # 大模型配置
            {
                'num_heads': 16,
                'd_model': 1024,
                'num_layers': 24,
                'batch_size': 32,
                'lr': 1e-4,
                'warmup_steps': 4000,
                'dropout': 0.3,
                'weight_decay': 0.01
            },
            {
                'num_heads': 16,
                'd_model': 1024,
                'num_layers': 12,
                'batch_size': 64,
                'lr': 2e-4,
                'warmup_steps': 4000,
                'dropout': 0.2,
                'weight_decay': 0.001
            },
            # 额外的测试配置
            {
                'num_heads': 8,
                'd_model': 1024,
                'num_layers': 6,
                'batch_size': 128,
                'lr': 4e-4,
                'warmup_steps': 1000,
                'dropout': 0.1,
                'weight_decay': 0.001
            },
            {
                'num_heads': 4,
                'd_model': 768,
                'num_layers': 24,
                'batch_size': 32,
                'lr': 5e-4,
                'warmup_steps': 4000,
                'dropout': 0.3,
                'weight_decay': 0.01
            }
        ]
    
    def generate_full_grid(self) -> List[Dict[str, Any]]:
        """生成完整的参数网格（所有组合）"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # 添加基础参数
            full_params = {**self.base_params, **param_dict}
            
            # 计算d_ff (通常是d_model的4倍)
            full_params['d_ff'] = full_params['d_model'] * 4
            
            # 设置学习率调度参数
            full_params['max_lr'] = full_params['lr']
            full_params['min_lr'] = full_params['lr'] * 0.01
            
            combinations.append(full_params)
        
        return combinations
    
    def generate_optimized_grid(self) -> List[Dict[str, Any]]:
        """生成优化的参数组合（预定义的高质量组合）"""
        combinations = []
        for combo in self.optimized_combinations:
            # 添加基础参数
            full_params = {**self.base_params, **combo}
            
            # 计算d_ff
            full_params['d_ff'] = full_params['d_model'] * 4
            
            # 设置学习率调度参数
            full_params['max_lr'] = full_params['lr']
            full_params['min_lr'] = full_params['lr'] * 0.01
            
            combinations.append(full_params)
        
        return combinations
    
    def filter_by_memory_constraint(self, combinations: List[Dict], max_memory_mb: int = 10000) -> List[Dict]:
        """根据显存限制过滤参数组合"""
        from gpu_manager import GPUManager
        
        gpu_manager = GPUManager()
        filtered_combinations = []
        
        for params in combinations:
            estimated_memory = gpu_manager.estimate_memory_usage(params)
            if estimated_memory <= max_memory_mb:
                params['estimated_memory_mb'] = estimated_memory
                filtered_combinations.append(params)
            else:
                print(f"跳过参数组合（显存需求过大: {estimated_memory}MB）: "
                      f"d_model={params['d_model']}, layers={params['num_layers']}, "
                      f"batch_size={params['batch_size']}")
        
        return filtered_combinations
    
    def generate_experiment_id(self, params: Dict) -> str:
        """为参数组合生成唯一的实验ID"""
        # 选择关键参数生成ID
        key_params = {
            'num_heads': params['num_heads'],
            'd_model': params['d_model'],
            'num_layers': params['num_layers'],
            'lr': params['lr'],
            'batch_size': params['batch_size'],
            'dropout': params['dropout'],
            'weight_decay': params['weight_decay'],
            'warmup_steps': params['warmup_steps']
        }
        
        # 生成参数字符串
        param_str = '_'.join([f"{k}={v}" for k, v in sorted(key_params.items())])
        
        # 生成短哈希
        hash_obj = hashlib.md5(param_str.encode())
        short_hash = hash_obj.hexdigest()[:8]
        
        # 生成可读的ID
        exp_id = f"exp_{params['d_model']}d_{params['num_layers']}l_{params['num_heads']}h_lr{params['lr']:.0e}_bs{params['batch_size']}_{short_hash}"
        
        return exp_id
    
    def save_experiment_configs(self, combinations: List[Dict], output_dir: str = "experiments"):
        """保存实验配置到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个组合生成实验ID并保存配置
        experiment_configs = []
        for i, params in enumerate(combinations):
            exp_id = self.generate_experiment_id(params)
            params['experiment_id'] = exp_id
            params['experiment_index'] = i
            
            # 保存单个实验配置
            config_file = os.path.join(output_dir, f"{exp_id}.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            experiment_configs.append({
                'experiment_id': exp_id,
                'config_file': config_file,
                'estimated_memory_mb': params.get('estimated_memory_mb', 0),
                'key_params': {
                    'd_model': params['d_model'],
                    'num_layers': params['num_layers'],
                    'num_heads': params['num_heads'],
                    'lr': params['lr'],
                    'batch_size': params['batch_size']
                }
            })
        
        # 保存实验列表
        experiment_list_file = os.path.join(output_dir, "experiment_list.json")
        with open(experiment_list_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_configs, f, indent=2, ensure_ascii=False)
        
        print(f"已生成 {len(combinations)} 个实验配置，保存到 {output_dir}/")
        return experiment_configs
    
    def save_config(self, params: Dict, config_file: str):
        """保存单个配置到文件"""
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    
    def print_grid_summary(self, combinations: List[Dict]):
        """打印参数网格摘要"""
        print("\n" + "="*60)
        print("参数网格摘要")
        print("="*60)
        print(f"总实验数量: {len(combinations)}")
        
        # 统计各参数的分布
        param_stats = {}
        for param in ['d_model', 'num_layers', 'num_heads', 'lr', 'batch_size', 'dropout', 'weight_decay']:
            values = [combo[param] for combo in combinations]
            unique_values = list(set(values))
            param_stats[param] = {
                'unique_values': sorted(unique_values),
                'count': len(unique_values)
            }
        
        for param, stats in param_stats.items():
            print(f"{param}: {stats['unique_values']} ({stats['count']} 种取值)")
        
        # 显存使用统计
        if 'estimated_memory_mb' in combinations[0]:
            memory_usage = [combo['estimated_memory_mb'] for combo in combinations]
            print(f"\n显存使用范围: {min(memory_usage)}-{max(memory_usage)} MB")
            print(f"平均显存使用: {sum(memory_usage)/len(memory_usage):.0f} MB")
        
        print("="*60)

def main():
    """主函数 - 生成和保存实验配置"""
    grid = ParameterGrid()
    
    print("生成参数网格...")
    
    # 选择生成方式
    use_optimized = True  # 设置为False使用完整网格
    
    if use_optimized:
        print("使用优化的参数组合...")
        combinations = grid.generate_optimized_grid()
    else:
        print("使用完整参数网格...")
        combinations = grid.generate_full_grid()
    
    print(f"初始组合数量: {len(combinations)}")
    
    # 根据显存限制过滤
    print("根据显存限制过滤参数组合...")
    filtered_combinations = grid.filter_by_memory_constraint(combinations, max_memory_mb=12000)
    print(f"过滤后组合数量: {len(filtered_combinations)}")
    
    # 打印摘要
    grid.print_grid_summary(filtered_combinations)
    
    # 保存配置
    experiment_configs = grid.save_experiment_configs(filtered_combinations)
    
    print(f"\n实验配置已保存，共 {len(experiment_configs)} 个实验")
    print("可以使用以下命令开始并行测试:")
    print("python parallel_trainer.py --config_dir experiments")

if __name__ == "__main__":
    main()