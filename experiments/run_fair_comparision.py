#!/usr/bin/env python3
"""
公平对比实验 - 比较有限传播FedAvg与相似度分组算法
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
from experiments.grouping_experiment import SimilarityGroupingExperiment
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
from visualization.visualization import Visualization

# 设置matplotlib不使用中文
plt.rcParams['font.sans-serif'] = ['Arial']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fair_comparison.log")
    ]
)
logger = logging.getLogger('fair_comparison')

def create_comparison_config(base_config_path, target_satellite_count=8):
    """根据基础配置创建针对特定卫星数量的配置"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 备份原配置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "configs/fair_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建新配置文件名
    new_config_path = f"{output_dir}/{os.path.basename(base_config_path).replace('.yaml', f'_{target_satellite_count}sats.yaml')}"
    
    return config, new_config_path

def create_fedavg_limited_config(base_config_path="configs/fedavg_config.yaml", target_satellite_count=8):
    """创建限制卫星数量的FedAvg配置"""
    config, new_config_path = create_comparison_config(base_config_path, target_satellite_count)
    
    # 计算合适的参与率
    # 假设总卫星数为66颗 (6个轨道 x 11颗/轨道)
    total_satellites = config['fl'].get('num_satellites', 66)
    
    # 目标参与率 = 目标卫星数 / 总卫星数
    participation_rate = target_satellite_count / total_satellites
    
    # 限制参与率的范围
    participation_rate = max(0.01, min(1.0, participation_rate))  # 确保至少有1%参与率
    
    # 更新FedAvg参与率
    if 'fedavg' not in config:
        config['fedavg'] = {}
    config['fedavg']['participation_rate'] = participation_rate
    
    logger.info(f"设置FedAvg参与率为 {participation_rate:.4f} (目标卫星数: {target_satellite_count})")
    
    # 保存修改后的配置
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    return new_config_path

def create_propagation_fedavg_config(base_config_path="configs/propagation_fedavg_config.yaml", target_satellite_count=22, hops=1):
    """创建有限传播FedAvg配置"""
    config, new_config_path = create_comparison_config(base_config_path, target_satellite_count)
    
    # 确保传播配置存在
    if 'propagation' not in config:
        config['propagation'] = {}
    
    # 设置传播参数
    config['propagation']['max_satellites'] = target_satellite_count  # 关键参数，限制参与的卫星数量
    config['propagation']['hops'] = hops  # 传播跳数
    
    # 启用轨道内和跨轨道链接
    config['propagation']['intra_orbit_links'] = True  # 轨道内链接
    config['propagation']['inter_orbit_links'] = True  # 跨轨道链接
    
    # 配置其他参数
    config['propagation']['link_reliability'] = 0.95  # 链接可靠性
    config['propagation']['energy_per_hop'] = 0.05  # 每跳传播能耗(Wh)
    
    # 设置FedAvg参与率（仅用于初始可见卫星的选择，不影响总卫星数）
    # 估计每个轨道至少需要可见1-2颗卫星
    total_satellites = config['fl'].get('num_satellites', 66)
    initial_visible = min(12, max(6, target_satellite_count // 4))  # 估计初始可见卫星，大约是目标数的1/4
    initial_participation_rate = initial_visible / total_satellites
    
    if 'fedavg' not in config:
        config['fedavg'] = {}
    config['fedavg']['participation_rate'] = initial_participation_rate
    
    # 确保每个轨道至少有一颗可见卫星
    min_per_orbit = max(1, target_satellite_count // (2 * config['fl'].get('num_orbits', 6)))
    config['fedavg']['min_satellite_per_orbit'] = min_per_orbit
    
    logger.info(f"设置有限传播FedAvg配置: 目标卫星数={target_satellite_count}, 传播跳数={hops}")
    logger.info(f"初始参与率: {initial_participation_rate:.4f} (约 {initial_visible} 颗可见卫星)")
    logger.info(f"每轨道最小卫星数: {min_per_orbit}")
    logger.info(f"传播链接: 轨道内={config['propagation']['intra_orbit_links']}, 跨轨道={config['propagation']['inter_orbit_links']}")
    
    # 保存修改后的配置
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    return new_config_path

def create_similarity_limited_config(base_config_path="configs/similarity_grouping_config.yaml", target_satellite_count=None):
    """
    创建相似度分组配置 - 不做任何卫星数量限制
    (让相似度分组算法按照原有设计运行)
    
    Args:
        base_config_path: 基础配置文件路径
        target_satellite_count: 忽略此参数，保留仅为接口兼容
        
    Returns:
        str: 配置文件路径
    """
    # 直接返回原始配置路径，不做任何修改
    logger.info(f"使用原始相似度分组配置: {base_config_path}")
    return base_config_path

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

def calculate_efficiency_metrics(fedavg_stats, similarity_stats):
    """计算效率指标"""
    metrics = {}
    
    # 获取平均参与卫星数
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 网络资源使用
    metrics['resource_utilization'] = {
        'fedavg_satellites': fedavg_sats,
        'similarity_satellites': similarity_sats,
        'satellite_saving': ((fedavg_sats - similarity_sats) / fedavg_sats) * 100 if fedavg_sats > 0 else 0
    }
    
    # 1. 准确率
    fedavg_acc = max(fedavg_stats['accuracies']) if fedavg_stats['accuracies'] else 0
    similarity_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
    
    metrics['accuracy'] = {
        'fedavg': fedavg_acc,
        'similarity': similarity_acc,
        'improvement': (similarity_acc - fedavg_acc)
    }
    
    # 2. 每卫星准确率
    metrics['accuracy_per_satellite'] = {
        'fedavg': fedavg_acc / fedavg_sats if fedavg_sats > 0 else 0,
        'similarity': similarity_acc / similarity_sats if similarity_sats > 0 else 0
    }
    
    # 3. 能源效率
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    metrics['energy'] = {
        'fedavg': fedavg_energy,
        'similarity': similarity_energy,
        'saving': (fedavg_energy - similarity_energy) / fedavg_energy * 100 if fedavg_energy > 0 else 0
    }
    
    metrics['energy_per_satellite'] = {
        'fedavg': fedavg_energy / fedavg_sats if fedavg_sats > 0 else 0,
        'similarity': similarity_energy / similarity_sats if similarity_sats > 0 else 0
    }
    
    # 4. 收敛速度
    fedavg_target = 0.9 * fedavg_acc
    similarity_target = 0.9 * similarity_acc
    
    fedavg_rounds = next((i+1 for i, acc in enumerate(fedavg_stats['accuracies']) 
                      if acc >= fedavg_target), len(fedavg_stats['accuracies']))
    similarity_rounds = next((i+1 for i, acc in enumerate(similarity_stats['accuracies']) 
                          if acc >= similarity_target), len(similarity_stats['accuracies']))
    
    metrics['convergence'] = {
        'fedavg': fedavg_rounds,
        'similarity': similarity_rounds,
        'speedup': (fedavg_rounds - similarity_rounds) / fedavg_rounds * 100 if fedavg_rounds > 0 else 0
    }
    
    return metrics

def create_fair_comparison_plots(fedavg_stats, similarity_stats, output_dir="comparison_results/fair"):
    """创建公平比较的可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取实际参与的卫星数量
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedAvg: {fedavg_sats:.1f} satellites vs Similarity: {similarity_sats:.1f} satellites)"
    
    # 1. 准确率对比
    plt.figure(figsize=(10, 6))
    rounds = range(1, min(len(fedavg_stats['accuracies']), len(similarity_stats['accuracies'])) + 1)
    
    if fedavg_stats['accuracies']:
        plt.plot(rounds, fedavg_stats['accuracies'][:len(rounds)], 'b-', label='FedAvg', marker='o')
    if similarity_stats['accuracies']:
        plt.plot(rounds, similarity_stats['accuracies'][:len(rounds)], 'r-', label='Similarity Grouping', marker='o')
    
    plt.title(f'Accuracy Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. 每轮能耗对比
    plt.figure(figsize=(10, 6))
    
    if fedavg_stats['energy_stats']['total_energy']:
        plt.plot(rounds, fedavg_stats['energy_stats']['total_energy'][:len(rounds)], 'b-', label='FedAvg', marker='o')
    if similarity_stats['energy_stats']['total_energy']:
        plt.plot(rounds, similarity_stats['energy_stats']['total_energy'][:len(rounds)], 'r-', label='Similarity Grouping', marker='o')
    
    plt.title(f'Energy Usage Per Round {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/energy_comparison.png")
    plt.close()
    
    # 3. 每颗卫星能耗
    plt.figure(figsize=(10, 6))
    
    fedavg_per_sat = [e/s if s > 0 else 0 for e, s in zip(
        fedavg_stats['energy_stats']['total_energy'][:len(rounds)],
        fedavg_stats['satellite_stats']['training_satellites'][:len(rounds)]
    )]
    
    similarity_per_sat = [e/s if s > 0 else 0 for e, s in zip(
        similarity_stats['energy_stats']['total_energy'][:len(rounds)],
        similarity_stats['satellite_stats']['training_satellites'][:len(rounds)]
    )]
    
    plt.plot(rounds, fedavg_per_sat, 'b-', label='FedAvg', marker='o')
    plt.plot(rounds, similarity_per_sat, 'r-', label='Similarity Grouping', marker='o')
    
    plt.title(f'Energy Per Satellite {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy Per Satellite (Wh/satellite)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/energy_per_satellite.png")
    plt.close()
    
    # 4. 能源效率比较 (准确率/总能耗)
    if fedavg_stats['accuracies'] and similarity_stats['accuracies']:
        fedavg_efficiency = [acc / (sum(fedavg_stats['energy_stats']['total_energy'][:i+1]) + 1e-10) 
                          for i, acc in enumerate(fedavg_stats['accuracies'][:len(rounds)])]
        similarity_efficiency = [acc / (sum(similarity_stats['energy_stats']['total_energy'][:i+1]) + 1e-10) 
                              for i, acc in enumerate(similarity_stats['accuracies'][:len(rounds)])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, fedavg_efficiency, 'b-', label='FedAvg', marker='o')
        plt.plot(rounds, similarity_efficiency, 'r-', label='Similarity Grouping', marker='o')
        
        plt.title(f'Energy Efficiency (Accuracy/Total Energy) {title_suffix}')
        plt.xlabel('Round')
        plt.ylabel('Energy Efficiency (%/Wh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/energy_efficiency.png")
        plt.close()
    
    # 5. 每轮参与卫星数量
    plt.figure(figsize=(10, 6))
    
    fedavg_training_sats = fedavg_stats['satellite_stats']['training_satellites'][:len(rounds)]
    similarity_training_sats = similarity_stats['satellite_stats']['training_satellites'][:len(rounds)]
    
    plt.plot(rounds, fedavg_training_sats, 'b-', label='FedAvg', marker='o')
    plt.plot(rounds, similarity_training_sats, 'r-', label='Similarity Grouping', marker='o')
    
    plt.title(f'Training Satellites Per Round {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Training Satellites')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/satellites_per_round.png")
    plt.close()

def generate_fair_comparison_report(metrics, fedavg_stats, similarity_stats, output_path="comparison_results/fair/report.txt"):
    """生成公平比较报告"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取实际参与的卫星数量
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    with open(output_path, "w") as f:
        f.write("# Fair Comparison: FedAvg vs Similarity Grouping\n\n")
        
        f.write(f"## Resource Utilization\n")
        f.write(f"- FedAvg Avg. Training Satellites: {fedavg_sats:.2f}\n")
        f.write(f"- Similarity Grouping Avg. Training Satellites: {similarity_sats:.2f}\n")
        f.write(f"- Satellite Utilization Saving: {metrics['resource_utilization']['satellite_saving']:.2f}%\n\n")
        
        f.write("## Accuracy Performance\n")
        f.write(f"- FedAvg Final Accuracy: {metrics['accuracy']['fedavg']:.2f}%\n")
        f.write(f"- Similarity Grouping Final Accuracy: {metrics['accuracy']['similarity']:.2f}%\n")
        f.write(f"- Accuracy Difference: {metrics['accuracy']['improvement']:+.2f}%\n\n")
        
        f.write("## Accuracy Per Satellite\n")
        f.write(f"- FedAvg Accuracy Per Satellite: {metrics['accuracy_per_satellite']['fedavg']:.2f}%/satellite\n")
        f.write(f"- Similarity Grouping Accuracy Per Satellite: {metrics['accuracy_per_satellite']['similarity']:.2f}%/satellite\n")
        ratio = metrics['accuracy_per_satellite']['similarity'] / metrics['accuracy_per_satellite']['fedavg'] if metrics['accuracy_per_satellite']['fedavg'] > 0 else 0
        f.write(f"- Efficiency Ratio: {ratio:.2f}x\n\n")
        
        f.write("## Energy Efficiency\n")
        f.write(f"- FedAvg Total Energy: {metrics['energy']['fedavg']:.2f} Wh\n")
        f.write(f"- Similarity Grouping Total Energy: {metrics['energy']['similarity']:.2f} Wh\n")
        f.write(f"- Energy Saving: {metrics['energy']['saving']:+.2f}%\n\n")
        
        f.write("## Energy Per Satellite\n")
        f.write(f"- FedAvg Energy Per Satellite: {metrics['energy_per_satellite']['fedavg']:.2f} Wh/satellite\n")
        f.write(f"- Similarity Grouping Energy Per Satellite: {metrics['energy_per_satellite']['similarity']:.2f} Wh/satellite\n")
        ratio = metrics['energy_per_satellite']['fedavg'] / metrics['energy_per_satellite']['similarity'] if metrics['energy_per_satellite']['similarity'] > 0 else 0
        f.write(f"- Efficiency Ratio: {ratio:.2f}x\n\n")
        
        f.write("## Convergence Speed\n")
        f.write(f"- FedAvg Rounds to 90% Final Accuracy: {metrics['convergence']['fedavg']}\n")
        f.write(f"- Similarity Grouping Rounds to 90% Final Accuracy: {metrics['convergence']['similarity']}\n")
        f.write(f"- Convergence Speedup: {metrics['convergence']['speedup']:+.2f}%\n\n")

# 在run_fair_comparison.py中，修改run_fair_comparison函数

def run_fair_comparison():
    """
    运行公平比较实验 - 使用有限传播FedAvg与相似度分组进行比较
    两种算法都利用卫星间通信扩展模型传播范围
    """
    logger.info(f"=== 开始公平比较实验: 有限传播FedAvg vs 相似度分组 ===")
    
    # 1. 首先运行相似度分组实验，使用原始配置
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_config_path = "configs/similarity_grouping_config.yaml"  # 原始配置
    similarity_stats, similarity_exp = run_experiment(similarity_config_path, SimilarityGroupingExperiment)
    
    if similarity_stats is None:
        logger.error("相似度分组实验失败，无法继续比较")
        return None, None, None
    
    # 获取相似度分组使用的卫星数量
    similarity_sats = similarity_stats['satellite_stats']['training_satellites']
    avg_similarity_sats = np.mean(similarity_sats)
    logger.info(f"相似度分组平均使用卫星数: {avg_similarity_sats:.2f}")
    
    # 2. 配置有限传播FedAvg实验，目标使用相似数量的卫星
    target_sats = int(np.ceil(avg_similarity_sats))
    
    # 创建FedAvg实验
    logger.info(f"\n=== 运行有限传播FedAvg实验 ===")
    logger.info(f"目标卫星数: {target_sats}")
    
    # 创建配置文件
    modified_config_path = create_propagation_fedavg_config(
        "configs/propagation_fedavg_config.yaml", 
        target_satellite_count=target_sats,
        hops=3  # 增加跳数，确保能够达到目标卫星数
    )
    
    # 3. 运行有限传播FedAvg实验，使用实际结果
    prop_fedavg_stats, prop_fedavg_exp = run_experiment(modified_config_path, LimitedPropagationFedAvg)
    
    if prop_fedavg_stats is None:
        logger.error("有限传播FedAvg实验失败，无法继续比较")
        return None, similarity_stats, None
    
    # 获取有限传播FedAvg实际使用的卫星数量
    fedavg_sats = prop_fedavg_stats['satellite_stats']['training_satellites']
    avg_fedavg_sats = np.mean(fedavg_sats)
    logger.info(f"有限传播FedAvg平均使用卫星数: {avg_fedavg_sats:.2f}")
    
    # 检查两种算法的卫星数量差异
    sat_diff = abs(avg_similarity_sats - avg_fedavg_sats)
    if sat_diff > 5:
        logger.warning(f"两种算法使用的卫星数量差异仍然较大: {sat_diff:.2f}颗")
        logger.warning(f"相似度分组: {avg_similarity_sats:.2f}颗, 有限传播FedAvg: {avg_fedavg_sats:.2f}颗")
    else:
        logger.info(f"两种算法使用的卫星数量接近，差异: {sat_diff:.2f}颗")
    
    # 创建描述性目录名
    output_dir = f"comparison_results/fair_propagation_sim{avg_similarity_sats:.1f}_fedavg{avg_fedavg_sats:.1f}"
    
    # 4. 计算效率指标
    metrics = calculate_efficiency_metrics(prop_fedavg_stats, similarity_stats)
    
    # 5. 创建可视化图表
    create_fair_comparison_plots(prop_fedavg_stats, similarity_stats, output_dir=output_dir)
    
    # 6. 生成比较报告
    generate_fair_comparison_report(metrics, prop_fedavg_stats, similarity_stats, 
                                output_path=f"{output_dir}/report.txt")
    
    logger.info("\n=== 公平比较实验完成 ===")
    logger.info(f"结果保存在 {output_dir}/ 目录下")
    
    # 打印关键指标对比
    logger.info("\n=== 关键指标对比 ===")
    logger.info(f"平均卫星数量: 相似度分组 {avg_similarity_sats:.2f} vs 有限传播FedAvg {avg_fedavg_sats:.2f}")
    
    if 'accuracies' in prop_fedavg_stats and 'accuracies' in similarity_stats:
        fedavg_acc = max(prop_fedavg_stats['accuracies']) if prop_fedavg_stats['accuracies'] else 0
        similarity_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
        logger.info(f"最终准确率: 相似度分组 {similarity_acc:.2f}% vs 有限传播FedAvg {fedavg_acc:.2f}%")
        logger.info(f"准确率差异: {similarity_acc - fedavg_acc:+.2f}%")
    
    return prop_fedavg_stats, similarity_stats, metrics

def run_fair_comparison_adaptive():
    """
    运行自适应公平比较实验 - 自动调整参数直到两种算法使用相近数量的卫星
    """
    logger.info(f"=== 开始自适应公平比较实验: 有限传播FedAvg vs 相似度分组 ===")
    
    # 1. 首先运行相似度分组实验，使用原始配置
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_config_path = "configs/similarity_grouping_config.yaml"  # 原始配置
    similarity_stats, similarity_exp = run_experiment(similarity_config_path, SimilarityGroupingExperiment)
    
    if similarity_stats is None:
        logger.error("相似度分组实验失败，无法继续比较")
        return None, None, None
    
    # 获取相似度分组使用的卫星数量
    similarity_sats = similarity_stats['satellite_stats']['training_satellites']
    avg_similarity_sats = np.mean(similarity_sats)
    logger.info(f"相似度分组平均使用卫星数: {avg_similarity_sats:.2f}")
    
    # 2. 自适应调整参数运行有限传播FedAvg实验
    target_sats = int(np.ceil(avg_similarity_sats))
    max_attempts = 3  # 最大尝试次数
    tolerance = 5.0   # 可接受的卫星数量差异
    
    prop_fedavg_stats = None
    
    # 尝试不同的参数组合，直到找到合适的卫星数量
    for attempt in range(1, max_attempts + 1):
        # 根据之前的结果调整跳数和其他参数
        if attempt == 1:
            hops = 2
            target_increase = 0
        elif attempt == 2:
            hops = 3
            target_increase = int(target_sats * 0.1)  # 目标增加10%
        else:
            hops = 4
            target_increase = int(target_sats * 0.2)  # 目标增加20%
        
        adjusted_target = target_sats + target_increase
        
        logger.info(f"\n=== 尝试 {attempt}/{max_attempts}: 运行有限传播FedAvg实验 ===")
        logger.info(f"参数: 目标卫星数={adjusted_target}, 传播跳数={hops}")
        
        # 创建配置文件
        modified_config_path = create_propagation_fedavg_config(
            "configs/propagation_fedavg_config.yaml", 
            target_satellite_count=adjusted_target,
            hops=hops
        )
        
        # 运行实验
        current_stats, current_exp = run_experiment(modified_config_path, LimitedPropagationFedAvg)
        
        if current_stats is None:
            logger.error(f"尝试 {attempt} 失败，跳过")
            continue
        
        # 获取实际使用的卫星数量
        current_sats = current_stats['satellite_stats']['training_satellites']
        avg_current_sats = np.mean(current_sats)
        
        logger.info(f"尝试 {attempt} 结果: 平均使用 {avg_current_sats:.2f} 颗卫星")
        
        # 检查差异是否在可接受范围内
        sat_diff = abs(avg_similarity_sats - avg_current_sats)
        if sat_diff <= tolerance:
            logger.info(f"找到合适的参数! 卫星数量差异: {sat_diff:.2f} <= {tolerance}")
            prop_fedavg_stats = current_stats
            prop_fedavg_exp = current_exp
            break
        
        # 保存最后一次结果，即使不是最佳的
        if attempt == max_attempts:
            logger.warning(f"达到最大尝试次数，使用最后一次结果。卫星数量差异: {sat_diff:.2f}")
            prop_fedavg_stats = current_stats
            prop_fedavg_exp = current_exp
    
    # 如果所有尝试都失败
    if prop_fedavg_stats is None:
        logger.error("所有尝试都失败，无法继续比较")
        return None, similarity_stats, None
    
    # 获取有限传播FedAvg实际使用的卫星数量
    fedavg_sats = prop_fedavg_stats['satellite_stats']['training_satellites']
    avg_fedavg_sats = np.mean(fedavg_sats)
    
    # 创建描述性目录名
    output_dir = f"comparison_results/fair_adaptive_sim{avg_similarity_sats:.1f}_fedavg{avg_fedavg_sats:.1f}"
    
    # 3. 计算效率指标
    metrics = calculate_efficiency_metrics(prop_fedavg_stats, similarity_stats)
    
    # 4. 创建可视化图表
    create_fair_comparison_plots(prop_fedavg_stats, similarity_stats, output_dir=output_dir)
    
    # 5. 生成比较报告
    generate_fair_comparison_report(metrics, prop_fedavg_stats, similarity_stats, 
                                output_path=f"{output_dir}/report.txt")
    
    logger.info("\n=== 自适应公平比较实验完成 ===")
    logger.info(f"结果保存在 {output_dir}/ 目录下")
    
    # 打印关键指标对比
    logger.info("\n=== 关键指标对比 ===")
    logger.info(f"平均卫星数量: 相似度分组 {avg_similarity_sats:.2f} vs 有限传播FedAvg {avg_fedavg_sats:.2f}")
    logger.info(f"卫星数量差异: {abs(avg_similarity_sats - avg_fedavg_sats):.2f}")
    
    if 'accuracies' in prop_fedavg_stats and 'accuracies' in similarity_stats:
        fedavg_acc = max(prop_fedavg_stats['accuracies']) if prop_fedavg_stats['accuracies'] else 0
        similarity_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
        logger.info(f"最终准确率: 相似度分组 {similarity_acc:.2f}% vs 有限传播FedAvg {fedavg_acc:.2f}%")
        logger.info(f"准确率差异: {similarity_acc - fedavg_acc:+.2f}%")
    
    return prop_fedavg_stats, similarity_stats, metrics

def run_vanilla_comparison():
    """运行标准比较实验 - 标准FedAvg vs 相似度分组"""
    logger.info(f"=== 开始标准比较实验: 标准FedAvg vs 相似度分组 ===")
    
    # 1. 运行相似度分组实验
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_config_path = "configs/similarity_grouping_config.yaml"
    similarity_stats, similarity_exp = run_experiment(similarity_config_path, SimilarityGroupingExperiment)
    
    # 2. 运行标准FedAvg实验
    logger.info("\n=== 运行标准FedAvg实验 ===")
    fedavg_config_path = "configs/fedavg_config.yaml"
    fedavg_stats, fedavg_exp = run_experiment(fedavg_config_path, FedAvgExperiment)
    
    # 获取两种算法使用的卫星数量
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    
    # 创建描述性目录名
    output_dir = f"comparison_results/vanilla_sim{similarity_sats:.1f}_fedavg{fedavg_sats:.1f}"
    
    # 3. 计算效率指标
    metrics = calculate_efficiency_metrics(fedavg_stats, similarity_stats)
    
    # 4. 创建可视化图表
    create_fair_comparison_plots(fedavg_stats, similarity_stats, output_dir=output_dir)
    
    # 5. 生成比较报告
    generate_fair_comparison_report(metrics, fedavg_stats, similarity_stats, 
                                  output_path=f"{output_dir}/report.txt")
    
    logger.info("\n=== 标准比较实验完成 ===")
    logger.info(f"结果保存在 {output_dir}/ 目录下")
    
    return fedavg_stats, similarity_stats, metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run fair comparison experiments')
    parser.add_argument('--mode', type=str, default='propagation', choices=['propagation', 'vanilla', 'adaptive'],
                      help='Comparison mode: propagation (limited propagation FedAvg), vanilla (standard FedAvg), or adaptive')
    parser.add_argument('--target-sats', type=int, default=22,
                      help='Target number of satellites for the fair comparison (used if mode is propagation)')
    parser.add_argument('--hops', type=int, default=3,
                      help='Number of propagation hops (used if mode is propagation)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 创建结果目录
    os.makedirs("comparison_results", exist_ok=True)
    
    if args.mode == 'propagation':
        # 使用有限传播FedAvg模式运行公平比较
        logger.info(f"使用有限传播FedAvg模式运行公平比较")
        fedavg_stats, similarity_stats, metrics = run_fair_comparison()
    elif args.mode == 'adaptive':
        # 使用自适应调整模式运行公平比较
        logger.info(f"使用自适应调整模式运行公平比较")
        fedavg_stats, similarity_stats, metrics = run_fair_comparison_adaptive()
    else:
        # 运行标准比较实验
        logger.info(f"使用标准FedAvg模式运行比较")
        fedavg_stats, similarity_stats, metrics = run_vanilla_comparison()
    
    # 打印最终结果
    if fedavg_stats and similarity_stats:
        fedavg_acc = max(fedavg_stats['accuracies']) if fedavg_stats['accuracies'] else 0
        similarity_acc = max(similarity_stats['accuracies']) if similarity_stats['accuracies'] else 0
        
        logger.info("\n=== 最终比较结果 ===")
        logger.info(f"FedAvg 最终准确率: {fedavg_acc:.2f}%")
        logger.info(f"相似度分组最终准确率: {similarity_acc:.2f}%")
        logger.info(f"准确率差异: {similarity_acc - fedavg_acc:+.2f}%")