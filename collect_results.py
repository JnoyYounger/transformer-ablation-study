# collect_results.py
import os
import json
import pandas as pd # 需要安装 pandas: pip install pandas

def collect_ablation_results(log_dir="logs"):
    """
    扫描 log_dir，读取每个实验的 summary.json，并汇总
    """
    results = []
    
    # 遍历 logs 目录下的所有文件夹
    try:
        experiment_names = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    except FileNotFoundError:
        print(f"[ERROR] 日志目录未找到: {log_dir}")
        print("请确保已至少运行过一个实验。")
        return

    for exp_name in experiment_names:
        summary_path = os.path.join(log_dir, exp_name, "summary.json")
        
        if not os.path.exists(summary_path):
            results.append({
                "Experiment": exp_name,
                "Status": "Not Found"
            })
            continue

        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # (从你的 TrainingLogger 的 summary.json 中提取)
            results.append({
                "Experiment": exp_name,
                "Status": data.get('status', 'N/A'),
                "Best Val Loss": data.get('best_val_loss', float('inf')),
                "Best Epoch": data.get('best_epoch', -1),
                "Total Time (s)": data.get('total_training_time_s', 0)
            })
        except Exception as e:
            results.append({
                "Experiment": exp_name,
                "Status": f"Error loading data: {e}",
                "Best Val Loss": float('inf'),
            })

    if not results:
        print(f"在 {log_dir} 中未找到任何实验结果。")
        return

    # 使用 Pandas 创建一个漂亮的表格
    df = pd.DataFrame(results)
    
    # 按最佳验证损失排序
    df = df.sort_values(by="Best Val Loss", ascending=True)

    print("=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)
    print(df.to_string(index=False)) # to_string() 格式化输出
    print("=" * 80)

    # 保存为 CSV
    save_path = os.path.join(log_dir, "ablation_summary_collected.csv")
    df.to_csv(save_path, index=False)
    print(f"汇总结果已保存到: {save_path}")

if __name__ == "__main__":
    collect_ablation_results()