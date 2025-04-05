#!/usr/bin/env python3
"""
对比实验 - 比较 FedProx、FedAvg 与相似度分组算法
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from experiments.fedavg_experiment import FedAvgExperiment
from experiments.fedprox_experiment import FedProxExperiment
from experiments.grouping_experiment import SimilarityGroupingExperiment
from visualization.visualization import Visualization

# 设置 matplotlib 不使用中文
plt.rcParams['font.sans-serif'] = ['Arial']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fedprox_comparison.log")
    ]
)
logger = logging.getLogger('fedprox_comparison')

def run_experiment(config_path, experiment_class):
    """运行单个实验"""
    logger.info(f"使用配置 {config_path} 运行 {experiment_class.__name__} 实验")
    
    try:
        experiment = experiment_class(config_path)
        
        # 准备数据
        experiment.prepare_data()
        
        # 设置客户端
        experiment.setup_clients()
        
        # 执行训练并获取统计信息
        stats = experiment.train()
        
        # 记录一些关键指标
        if 'accuracies' in stats and stats['accuracies']:
            max_acc = max(stats['accuracies'])
            logger.info(f"实验最高准确率: {max_acc:.2f}%")
        
        if 'satellite_stats' in stats and 'training_satellites' in stats['satellite_stats']:
            avg_sats = np.mean(stats['satellite_stats']['training_satellites'])
            logger.info(f"平均参与卫星数: {avg_sats:.2f}")
        
        return stats, experiment
    
    except Exception as e:
        logger.error(f"运行实验 {experiment_class.__name__} 出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def create_comparison_plots(fedprox_stats, fedavg_stats, similarity_stats, output_dir):
    """创建对比可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取实际参与的卫星数量
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedProx: {fedprox_sats:.1f}, FedAvg: {fedavg_sats:.1f}, Similarity: {similarity_sats:.1f} satellites)"
    
    # 1. 准确率对比
    plt.figure(figsize=(10, 6))
    rounds = range(1, min(len(fedprox_stats['accuracies']), 
                         len(fedavg_stats['accuracies']), 
                         len(similarity_stats['accuracies'])) + 1)
    
    plt.plot(rounds, fedprox_stats['accuracies'][:len(rounds)], 'g-', label='FedProx', marker='o')
    plt.plot(rounds, fedavg_stats['accuracies'][:len(rounds)], 'b-', label='FedAvg', marker='s')
    plt.plot(rounds, similarity_stats['accuracies'][:len(rounds)], 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Accuracy Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. 损失对比
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, fedprox_stats['losses'][:len(rounds)], 'g-', label='FedProx', marker='o')
    plt.plot(rounds, fedavg_stats['losses'][:len(rounds)], 'b-', label='FedAvg', marker='s')
    plt.plot(rounds, similarity_stats['losses'][:len(rounds)], 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Loss Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_comparison.png")
    plt.close()
    
    # 3. 能耗对比
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, fedprox_stats['energy_stats']['total_energy'][:len(rounds)], 'g-', label='FedProx', marker='o')
    plt.plot(rounds, fedavg_stats['energy_stats']['total_energy'][:len(rounds)], 'b-', label='FedAvg', marker='s')
    plt.plot(rounds, similarity_stats['energy_stats']['total_energy'][:len(rounds)], 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Energy Consumption Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/energy_comparison.png")
    plt.close()
    
    # 4. 活跃卫星数对比
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, fedprox_stats['satellite_stats']['training_satellites'][:len(rounds)], 'g-', label='FedProx', marker='o')
    plt.plot(rounds, fedavg_stats['satellite_stats']['training_satellites'][:len(rounds)], 'b-', label='FedAvg', marker='s')
    plt.plot(rounds, similarity_stats['satellite_stats']['training_satellites'][:len(rounds)], 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Training Satellites Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/satellites_comparison.png")
    plt.close()
    
    # 5. 能效比对比 (每颗卫星获得的准确率)
    plt.figure(figsize=(10, 6))
    
    fedprox_efficiency = [acc / sat_count if sat_count > 0 else 0 
                         for acc, sat_count in zip(
                             fedprox_stats['accuracies'][:len(rounds)], 
                             fedprox_stats['satellite_stats']['training_satellites'][:len(rounds)])]
    
    fedavg_efficiency = [acc / sat_count if sat_count > 0 else 0 
                         for acc, sat_count in zip(
                             fedavg_stats['accuracies'][:len(rounds)], 
                             fedavg_stats['satellite_stats']['training_satellites'][:len(rounds)])]
    
    similarity_efficiency = [acc / sat_count if sat_count > 0 else 0 
                            for acc, sat_count in zip(
                                similarity_stats['accuracies'][:len(rounds)], 
                                similarity_stats['satellite_stats']['training_satellites'][:len(rounds)])]
    
    plt.plot(rounds, fedprox_efficiency, 'g-', label='FedProx', marker='o')
    plt.plot(rounds, fedavg_efficiency, 'b-', label='FedAvg', marker='s')
    plt.plot(rounds, similarity_efficiency, 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Efficiency (Accuracy per Satellite) {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy per Satellite (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/efficiency_comparison.png")
    plt.close()
    
    # 6. 累积能耗与准确率对比
    plt.figure(figsize=(10, 6))
    
    fedprox_cum_energy = np.cumsum(fedprox_stats['energy_stats']['total_energy'][:len(rounds)])
    fedavg_cum_energy = np.cumsum(fedavg_stats['energy_stats']['total_energy'][:len(rounds)])
    similarity_cum_energy = np.cumsum(similarity_stats['energy_stats']['total_energy'][:len(rounds)])
    
    plt.plot(fedprox_cum_energy, fedprox_stats['accuracies'][:len(rounds)], 'g-', label='FedProx', marker='o')
    plt.plot(fedavg_cum_energy, fedavg_stats['accuracies'][:len(rounds)], 'b-', label='FedAvg', marker='s')
    plt.plot(similarity_cum_energy, similarity_stats['accuracies'][:len(rounds)], 'r-', label='Similarity Grouping', marker='^')
    
    plt.title(f'Accuracy vs. Cumulative Energy {title_suffix}')
    plt.xlabel('Cumulative Energy (Wh)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_vs_energy.png")
    plt.close()

def generate_comparison_report(fedprox_stats, fedavg_stats, similarity_stats, output_path):
    """生成对比报告"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 计算各种指标
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    
    fedprox_avg_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算各种效率指标
    fedprox_efficiency = fedprox_max_acc / fedprox_avg_sats if fedprox_avg_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_avg_sats if fedavg_avg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_avg_sats if similarity_avg_sats > 0 else 0
    
    fedprox_energy_efficiency = fedprox_max_acc / fedprox_energy if fedprox_energy > 0 else 0
    fedavg_energy_efficiency = fedavg_max_acc / fedavg_energy if fedavg_energy > 0 else 0
    similarity_energy_efficiency = similarity_max_acc / similarity_energy if similarity_energy > 0 else 0
    
    # 计算收敛速度 (达到90%最终准确率的轮次)
    def calculate_convergence(accuracies, target_percentage=0.9):
        target = max(accuracies) * target_percentage
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i + 1
        return len(accuracies)  # 如果没有达到，返回总轮次
    
    fedprox_convergence = calculate_convergence(fedprox_stats['accuracies'])
    fedavg_convergence = calculate_convergence(fedavg_stats['accuracies'])
    similarity_convergence = calculate_convergence(similarity_stats['accuracies'])
    
    # 生成报告
    with open(output_path, 'w') as f:
        f.write("# 对比报告: FedProx vs FedAvg vs 相似度分组\n\n")
        
        f.write("## 准确率性能\n")
        f.write(f"- FedProx 最高准确率: {fedprox_max_acc:.2f}%\n")
        f.write(f"- FedAvg 最高准确率: {fedavg_max_acc:.2f}%\n")
        f.write(f"- 相似度分组最高准确率: {similarity_max_acc:.2f}%\n\n")
        
        f.write(f"- FedProx vs FedAvg: {fedprox_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- FedProx vs 相似度分组: {fedprox_max_acc - similarity_max_acc:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedAvg: {similarity_max_acc - fedavg_max_acc:+.2f}%\n\n")
        
        f.write("## 资源利用\n")
        f.write(f"- FedProx 平均训练卫星数: {fedprox_avg_sats:.2f}\n")
        f.write(f"- FedAvg 平均训练卫星数: {fedavg_avg_sats:.2f}\n")
        f.write(f"- 相似度分组平均训练卫星数: {similarity_avg_sats:.2f}\n\n")
        
        f.write("## 能耗\n")
        f.write(f"- FedProx 总能耗: {fedprox_energy:.2f} Wh\n")
        f.write(f"- FedAvg 总能耗: {fedavg_energy:.2f} Wh\n")
        f.write(f"- 相似度分组总能耗: {similarity_energy:.2f} Wh\n\n")
        
        f.write("## 效率指标\n")
        f.write(f"- FedProx 每卫星准确率: {fedprox_efficiency:.2f}%\n")
        f.write(f"- FedAvg 每卫星准确率: {fedavg_efficiency:.2f}%\n")
        f.write(f"- 相似度分组每卫星准确率: {similarity_efficiency:.2f}%\n\n")
        
        f.write(f"- FedProx 能源效率: {fedprox_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- FedAvg 能源效率: {fedavg_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- 相似度分组能源效率: {similarity_energy_efficiency:.4f}%/Wh\n\n")
        
        f.write("## 收敛速度\n")
        f.write(f"- FedProx 达到90%最高准确率轮次: {fedprox_convergence}\n")
        f.write(f"- FedAvg 达到90%最高准确率轮次: {fedavg_convergence}\n")
        f.write(f"- 相似度分组达到90%最高准确率轮次: {similarity_convergence}\n\n")
        
        f.write("## 总结\n")
        # 添加各个方法的优缺点和总结
        f.write("### FedProx\n")
        f.write("- **优势**: 在非IID数据分布下更稳定的收敛性，改进的性能，对数据异质性更鲁棒。\n")
        f.write("- **劣势**: 额外的计算开销，需要调整接近性参数μ，通信成本较高。\n\n")
        
        f.write("### FedAvg\n")
        f.write("- **优势**: 简单，计算开销低。\n")
        f.write("- **劣势**: 在非IID数据上可能发散，资源利用率较低，对异质性数据敏感。\n\n")
        
        f.write("### 相似度分组\n")
        f.write("- **优势**: 高效的资源利用，适应数据分布特点，每卫星性能更好，能耗效率高。\n")
        f.write("- **劣势**: 需要计算数据相似度的开销，实现更复杂。\n\n")
        
        if similarity_max_acc > fedprox_max_acc and similarity_efficiency > fedprox_efficiency:
            f.write("### 结论\n")
            f.write("相似度分组方法在准确率和资源效率方面都优于FedProx和FedAvg，特别适合资源受限的卫星环境。\n")
        elif fedprox_max_acc > similarity_max_acc:
            f.write("### 结论\n")
            f.write("FedProx在准确率方面表现最好，适合对模型性能要求高的场景，但资源效率不如相似度分组方法。\n")
        else:
            f.write("### 结论\n")
            f.write("根据不同的优化目标，可以选择不同的方法：对准确率要求高选FedProx，对资源效率要求高选相似度分组。\n")

def run_comparison():
    """运行三种算法的比较实验"""
    logger.info("=== 开始比较实验: FedProx vs FedAvg vs 相似度分组 ===")
    
    # 1. 运行 FedProx 实验
    logger.info("\n=== 运行 FedProx 实验 ===")
    fedprox_config_path = "configs/fedprox_config.yaml"
    fedprox_stats, fedprox_exp = run_experiment(fedprox_config_path, FedProxExperiment)
    
    # 2. 运行 FedAvg 实验
    logger.info("\n=== 运行 FedAvg 实验 ===")
    fedavg_config_path = "configs/fedavg_config.yaml"
    fedavg_stats, fedavg_exp = run_experiment(fedavg_config_path, FedAvgExperiment)
    
    # 3. 运行相似度分组实验
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_config_path = "configs/similarity_grouping_config.yaml"
    similarity_stats, similarity_exp = run_experiment(similarity_config_path, SimilarityGroupingExperiment)
    
    # 检查所有实验是否完成
    if not fedprox_stats or not fedavg_stats or not similarity_stats:
        logger.error("一个或多个实验失败，无法进行比较")
        return
    
    # 创建输出目录
    output_dir = f"comparison_results/fedprox_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建比较图表
    create_comparison_plots(fedprox_stats, fedavg_stats, similarity_stats, output_dir)
    
    # 生成比较报告
    generate_comparison_report(fedprox_stats, fedavg_stats, similarity_stats, f"{output_dir}/comparison_report.md")
    
    logger.info(f"比较完成，结果保存在 {output_dir}/ 目录下")

if __name__ == "__main__":
    run_comparison()