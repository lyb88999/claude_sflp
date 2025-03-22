import os
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.fedavg_experiment import FedAvgExperiment
from experiments.grouping_experiment import SimilarityGroupingExperiment
import warnings

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_comparison():
    # 创建结果目录
    os.makedirs("comparison_results", exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置matplotlib使用英文
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 首先运行相似度分组实验
    print("=== Running Similarity Grouping Experiment ===")
    similarity_exp = SimilarityGroupingExperiment("configs/similarity_grouping_config.yaml")
    similarity_stats = similarity_exp.run()
    
    # 计算相似度分组使用的平均卫星数
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    print(f"相似度分组平均使用卫星数: {similarity_avg_sats:.2f}")
    
    # 创建有限传播FedAvg配置，目标使用类似数量的卫星
    target_sats = int(np.ceil(similarity_avg_sats))
    print(f"配置有限传播FedAvg目标卫星数: {target_sats}")
    
    # 运行有限传播FedAvg
    print("=== Running Limited Propagation FedAvg Experiment ===")
    # 使用正确的配置文件
    fedavg_exp = LimitedPropagationFedAvg("configs/propagation_fedavg_config.yaml")
    
    # 打印配置信息
    print(f"FedAvg参数传播配置:")
    print(f"- 跳数: {fedavg_exp.propagation_hops}")
    print(f"- 最大卫星数: {fedavg_exp.max_propagation_satellites}")
    
    # 运行实验
    fedavg_stats = fedavg_exp.run()
    
    # 打印实际使用的卫星数
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    print(f"有限传播FedAvg平均使用卫星数: {fedavg_avg_sats:.2f}")
    
    # 绘制对比图表
    plot_comparison(fedavg_stats, similarity_stats)
    
    # 生成总结报告
    generate_summary_report(fedavg_stats, similarity_stats)
    
    return fedavg_stats, similarity_stats

def plot_comparison(fedavg_stats, similarity_stats):
    """生成对比图表"""
    # 1. 准确率对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['accuracies'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['accuracies'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/accuracy_comparison.png")
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['losses'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['losses'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Loss Comparison')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/loss_comparison.png")
    plt.close()
    
    # 3. 能耗对比 - 训练能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['training_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Training Energy Consumption')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/training_energy_comparison.png")
    plt.close()
    
    # 4. 能耗对比 - 通信能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['communication_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['communication_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Communication Energy Consumption')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/communication_energy_comparison.png")
    plt.close()
    
    # 5. 能耗对比 - 总能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['total_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Total Energy Consumption')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/total_energy_comparison.png")
    plt.close()
    
    # 6. 能效比对比(准确率/能耗)
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_efficiency, 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_efficiency, 'r-', label='Similarity Grouping', marker='o')
    plt.title('Energy Efficiency (Accuracy/Energy)')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/efficiency_comparison.png")
    plt.close()
    
    # 7. 活跃卫星数量对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 'r-', label='Similarity Grouping', marker='o')
    plt.title('Number of Training Satellites')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_results/training_satellites_comparison.png")
    plt.close()

def generate_summary_report(fedavg_stats, similarity_stats):
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
    sat_saving = (1 - similarity_avg_sats/fedavg_avg_sats) * 100 if fedavg_avg_sats > 0 else 0
    
    # 生成报告
    with open("comparison_results/summary_report.txt", "w") as f:
        f.write("# Comparison Summary: FedAvg vs. Similarity Grouping\n\n")
        
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

        sat_diff_percent = ((similarity_avg_sats - fedavg_avg_sats) / fedavg_avg_sats) * 100

        # 修改这一行，使用中性描述
        if sat_diff_percent > 0:
            f.write(f"- Satellite Utilization Difference: Similarity uses {sat_diff_percent:.2f}% more satellites\n\n")
        else:
            f.write(f"- Satellite Utilization Difference: Similarity uses {-sat_diff_percent:.2f}% fewer satellites\n\n")
    
    print("Summary report generated at comparison_results/summary_report.txt")

if __name__ == "__main__":
    fedavg_stats, similarity_stats = run_comparison()
    print("Comparison completed. Results saved in comparison_results/ directory.")