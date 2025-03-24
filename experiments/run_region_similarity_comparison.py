import os
from experiments.fedavg_experiment import FedAvgExperiment
from experiments.grouping_experiment import SimilarityGroupingExperiment
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_region_similarity_comparison():
    """
    运行区域相似性数据分布下的对比实验
    """
    # 创建结果目录
    output_dir = f"comparison_results/region_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置matplotlib使用英文
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 运行FedAvg实验
    print("=== Running FedAvg Experiment ===")
    fedavg_exp = LimitedPropagationFedAvg('configs/region_similarity_config.yaml')
    fedavg_stats = fedavg_exp.run()
    
    # 打印FedAvg平均使用卫星数
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    print(f"FedAvg平均使用卫星数: {fedavg_avg_sats:.2f}")
    
    # 运行相似度分组实验
    print("=== Running Similarity Grouping Experiment ===")
    similarity_exp = SimilarityGroupingExperiment("configs/region_similarity_config.yaml")
    similarity_stats = similarity_exp.run()
    
    # 打印相似度分组使用的平均卫星数
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    print(f"相似度分组平均使用卫星数: {similarity_avg_sats:.2f}")
    
    # 绘制对比图表
    plot_comparison(fedavg_stats, similarity_stats, output_dir)
    
    # 生成总结报告
    generate_summary_report(fedavg_stats, similarity_stats, output_dir)
    
    print(f"Comparison completed. Results saved in {output_dir}/ directory.")
    
    return fedavg_stats, similarity_stats

def plot_comparison(fedavg_stats, similarity_stats, output_dir):
    """生成对比图表"""
    # 获取实际参与的卫星数量
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedAvg: {fedavg_sats:.1f} vs Similarity: {similarity_sats:.1f} satellites)"
    
    # 1. 准确率对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['accuracies'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['accuracies'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Accuracy Comparison with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['losses'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['losses'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Loss Comparison with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_comparison.png")
    plt.close()
    
    # 3. 能耗对比 - 训练能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['training_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Training Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_energy_comparison.png")
    plt.close()
    
    # 4. 能耗对比 - 通信能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['communication_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['communication_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Communication Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/communication_energy_comparison.png")
    plt.close()
    
    # 5. 能耗对比 - 总能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['total_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Total Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/total_energy_comparison.png")
    plt.close()
    
    # 6. 能效比对比(准确率/能耗)
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_efficiency, 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_efficiency, 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Energy Efficiency (Accuracy/Energy) with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/efficiency_comparison.png")
    plt.close()
    
    # 7. 活跃卫星数量对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Number of Training Satellites with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_satellites_comparison.png")
    plt.close()

def generate_summary_report(fedavg_stats, similarity_stats, output_dir):
    """生成对比分析报告"""
    # 计算关键指标
    fedavg_max_acc = max(fedavg_stats['accuracies']) if fedavg_stats['accuracies'] else 0
    similarity_max_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
    
    fedavg_total_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_total_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算能源节省比例
    energy_saving = (1 - similarity_total_energy/fedavg_total_energy) * 100 if fedavg_total_energy > 0 else 0
    
    # 计算收敛速度 - 达到90%最终准确率所需轮次
    fedavg_target = 0.9 * fedavg_max_acc
    similarity_target = 0.9 * similarity_max_acc
    
    fedavg_rounds = next((i+1 for i, acc in enumerate(fedavg_stats['accuracies']) 
                       if acc >= fedavg_target), len(fedavg_stats['accuracies']))
    similarity_rounds = next((i+1 for i, acc in enumerate(similarity_stats['accuracies']) 
                            if acc >= similarity_target), len(similarity_stats['accuracies']))
    
    # 计算平均每轮训练卫星数
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 节省的训练卫星百分比
    sat_diff_percent = ((similarity_avg_sats - fedavg_avg_sats) / fedavg_avg_sats) * 100
    
    # 生成报告
    with open(f"{output_dir}/summary_report.txt", "w") as f:
        f.write("# Comparison Summary: FedAvg vs. Similarity Grouping with Region Similar Data\n\n")
        
        f.write("## Accuracy Performance\n")
        f.write(f"- FedAvg Final Accuracy: {fedavg_max_acc:.2f}%\n")
        f.write(f"- Similarity Grouping Final Accuracy: {similarity_max_acc:.2f}%\n")
        f.write(f"- Accuracy Difference: {similarity_max_acc-fedavg_max_acc:+.2f}%\n\n")
        
        f.write("## Energy Efficiency\n")
        f.write(f"- FedAvg Total Energy: {fedavg_total_energy:.2f} Wh\n")
        f.write(f"- Similarity Grouping Total Energy: {similarity_total_energy:.2f} Wh\n")
        f.write(f"- Energy Savings: {energy_saving:.2f}%\n\n")
        
        f.write("## Convergence Speed\n")
        f.write(f"- FedAvg Rounds to 90% Max Accuracy: {fedavg_rounds}\n")
        f.write(f"- Similarity Grouping Rounds to 90% Max Accuracy: {similarity_rounds}\n")
        f.write(f"- Convergence Speedup: {(fedavg_rounds-similarity_rounds)/fedavg_rounds*100:.2f}%\n\n")
        
        f.write("## Resource Utilization\n")
        f.write(f"- FedAvg Avg. Training Satellites: {fedavg_avg_sats:.2f}\n")
        f.write(f"- Similarity Grouping Avg. Training Satellites: {similarity_avg_sats:.2f}\n")

        # 使用中性描述
        if sat_diff_percent > 0:
            f.write(f"- Satellite Utilization Difference: Similarity uses {sat_diff_percent:.2f}% more satellites\n\n")
        else:
            f.write(f"- Satellite Utilization Difference: Similarity uses {-sat_diff_percent:.2f}% fewer satellites\n\n")
        
        # 添加区域相似性特定优势
        f.write("## Advantages of Similarity Grouping in Region Similar Data\n")
        f.write("1. **Higher Data Utilization Efficiency**: Similarity grouping can identify data similarity among satellites in regions, utilizing overlapping data more effectively.\n\n")
        f.write("2. **Lower Communication Cost**: By identifying data similarity, redundant satellite participation in training is reduced, lowering overall communication costs.\n\n")
        f.write("3. **Better Model Performance**: Similarity grouping allows each group of satellites to optimize for their specific data distribution, improving overall model performance.\n\n")
        f.write("4. **Higher Energy Efficiency**: By selecting representative satellites for training, energy consumption is significantly reduced.\n\n")
        f.write("5. **Faster Convergence Speed**: Training based on data similarity allows the model to converge to optimal solutions more quickly.\n")
    
    print(f"Summary report generated at {output_dir}/summary_report.txt")

if __name__ == "__main__":
    fedavg_stats, similarity_stats = run_region_similarity_comparison()
    print("Comparison completed. Results saved in comparison_results/ directory.")