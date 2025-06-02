#!/usr/bin/env python3
"""
公平对比实验 - 比较有限传播FedProx、有限传播FedAvg与相似度分组算法
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import argparse
import seaborn as sns
from datetime import datetime
from pathlib import Path
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
from experiments.propagation_fedprox_experiment import LimitedPropagationFedProx
from experiments.grouping_experiment import SimilarityGroupingExperiment
from visualization.visualization import Visualization

import pickle
import json
import copy
import argparse
from pathlib import Path

# 设置matplotlib不使用中文
plt.rcParams['font.sans-serif'] = ['Arial']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fair_comparison_fedprox.log")
    ]
)
logger = logging.getLogger('fair_comparison_fedprox')

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

def calculate_communication_overhead(stats):
    """计算通信开销"""
    try:
        return np.cumsum(stats['energy_stats']['communication_energy'])
    except Exception as e:
        logger.error(f"计算通信开销时出错: {str(e)}")
        return np.array([0])

def calculate_convergence_speed(accuracies, target_accuracy=None):
    """计算收敛速度 - 达到目标准确率所需轮次"""
    if not accuracies:
        return float('inf')
        
    if target_accuracy is None:
        # 如果未指定目标准确率，使用最终准确率的90%
        target_accuracy = 0.9 * max(accuracies)
    
    # 找到第一个达到或超过目标准确率的轮次
    for round_num, acc in enumerate(accuracies):
        if acc >= target_accuracy:
            return round_num + 1
    
    # 如果没有达到目标，返回总轮次
    return len(accuracies)

def create_comparison_plots(fedprox_stats, fedavg_stats, similarity_stats, output_dir, 
                           fedprox_exp=None, fedavg_exp=None, similarity_exp=None,
                           custom_style=None, show_grid=True, figure_format='png', dpi=150):
    """
    创建对比图表，支持自定义图表样式
    
    Args:
        fedprox_stats: FedProx实验统计数据
        fedavg_stats: FedAvg实验统计数据
        similarity_stats: 相似度分组实验统计数据
        output_dir: 输出目录
        fedprox_exp: FedProx实验对象(可选)
        fedavg_exp: FedAvg实验对象(可选)
        similarity_exp: 相似度分组实验对象(可选)
        custom_style: 自定义图表样式字典
        show_grid: 是否显示网格
        figure_format: 图片格式(png, pdf, svg等)
        dpi: 图像DPI
    """
    # 获取实际参与的卫星数量
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedProx: {fedprox_sats:.1f}, FedAvg: {fedavg_sats:.1f}, Similarity: {similarity_sats:.1f} satellites)"
    
    # 应用自定义样式
    style = {
        'figsize': (10, 6),
        'title_fontsize': 14,
        'label_fontsize': 12,
        'tick_fontsize': 10,
        'legend_fontsize': 10,
        'linewidth': 2,
        'marker_size': 6,
        'grid_alpha': 0.3,
        'grid_linestyle': '--',
        'save_format': figure_format,
        'dpi': dpi
    }
    
    if custom_style:
        style.update(custom_style)
    
    # 设置默认样式
    plt.rcParams.update({
        'font.size': style['label_fontsize'],
        'axes.titlesize': style['title_fontsize'],
        'axes.labelsize': style['label_fontsize'],
        'xtick.labelsize': style['tick_fontsize'],
        'ytick.labelsize': style['tick_fontsize'],
        'legend.fontsize': style['legend_fontsize']
    })
    
    # 定义算法颜色和标记
    algo_styles = {
        'FedProx': {'color': 'g', 'marker': 'o', 'label': 'FedProx'},
        'FedAvg': {'color': 'b', 'marker': 's', 'label': 'FedAvg'},
        'Similarity': {'color': 'r', 'marker': '^', 'label': 'Similarity Grouping'}
    }
    
    # 1. 准确率对比
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['accuracies'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['accuracies'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['accuracies'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Accuracy Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['losses'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['losses'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['losses'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Loss Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 3. 能耗对比 - 训练能耗
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['training_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Training Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 4. 能耗对比 - 通信能耗
    plt.figure(figsize=style['figsize'])
    if 'energy_stats' in fedprox_stats and 'communication_energy' in fedprox_stats['energy_stats']:
        plt.plot(fedprox_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'energy_stats' in fedavg_stats and 'communication_energy' in fedavg_stats['energy_stats']:
        plt.plot(fedavg_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'energy_stats' in similarity_stats and 'communication_energy' in similarity_stats['energy_stats']:
        plt.plot(similarity_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Communication Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 5. 能耗对比 - 总能耗
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['total_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Total Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 6. 能效比对比(准确率/能耗)
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedprox_stats['accuracies'], fedprox_stats['energy_stats']['total_energy'])]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_efficiency, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_efficiency, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_efficiency, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Energy Efficiency (Accuracy/Energy) {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 7. 活跃卫星数量对比
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Number of Training Satellites {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_satellites_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()

    # 8. 通信开销对比
    plt.figure(figsize=style['figsize'])
    fedprox_comm = calculate_communication_overhead(fedprox_stats)
    fedavg_comm = calculate_communication_overhead(fedavg_stats)
    similarity_comm = calculate_communication_overhead(similarity_stats)
    
    plt.plot(fedprox_comm, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_comm, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_comm, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Cumulative Communication Overhead {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Communication Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_overhead.{style['save_format']}", dpi=style['dpi'])
    plt.close()

    # 9. 能效比对比 (准确率/累积能耗)
    plt.figure(figsize=style['figsize'])
    
    # 计算能效比 - 每单位能量获得的准确率
    fedprox_cumulative_energy = np.cumsum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_cumulative_energy = np.cumsum(fedavg_stats['energy_stats']['total_energy'])
    similarity_cumulative_energy = np.cumsum(similarity_stats['energy_stats']['total_energy'])
    
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedprox_stats['accuracies'], fedprox_cumulative_energy)]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_cumulative_energy)]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_cumulative_energy)]
    
    plt.plot(fedprox_efficiency, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_efficiency, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_efficiency, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Energy Efficiency (Accuracy/Cumulative Energy) {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_efficiency_cumulative.{style['save_format']}", dpi=style['dpi'])
    plt.close()

    # 10. 区域内一致性比较 (如果可用)
    if all(hasattr(exp, 'region_coherence') for exp in [fedprox_exp, fedavg_exp, similarity_exp] if exp is not None):
        try:
            plt.figure(figsize=style['figsize'])
            
            # 获取所有轨道ID
            all_orbits = sorted(list(set().union(
                *(getattr(exp, 'region_coherence', {}).keys() for exp in [fedprox_exp, fedavg_exp, similarity_exp] if exp is not None)
            )))
            
            if all_orbits:
                x = list(range(1, len(all_orbits) + 1))
                bar_width = 0.25  # 三组数据，所以条形宽度更窄
                
                fedprox_coherence = [getattr(fedprox_exp, 'region_coherence', {}).get(orbit, 0) for orbit in all_orbits]
                fedavg_coherence = [getattr(fedavg_exp, 'region_coherence', {}).get(orbit, 0) for orbit in all_orbits]
                similarity_coherence = [getattr(similarity_exp, 'region_coherence', {}).get(orbit, 0) for orbit in all_orbits]
                
                plt.bar(np.array(x) - bar_width, fedprox_coherence, width=bar_width, 
                       color=algo_styles['FedProx']['color'], label=algo_styles['FedProx']['label'])
                plt.bar(np.array(x), fedavg_coherence, width=bar_width, 
                       color=algo_styles['FedAvg']['color'], label=algo_styles['FedAvg']['label'])
                plt.bar(np.array(x) + bar_width, similarity_coherence, width=bar_width, 
                       color=algo_styles['Similarity']['color'], label=algo_styles['Similarity']['label'])
                
                plt.title('Model Coherence Within Regions')
                plt.xlabel('Region (Orbit)')
                plt.ylabel('Intra-Region Model Similarity')
                plt.xticks(x, all_orbits)
                plt.legend()
                if show_grid:
                    plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'], axis='y')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/region_coherence.{style['save_format']}", dpi=style['dpi'])
            plt.close()
        except Exception as e:
            logger.error(f"绘制区域内一致性对比图时出错: {str(e)}")

    # 11. 如果有FedProx特有的指标，绘制FedProx的接近性项
    if 'proximal_terms' in fedprox_stats:
        plt.figure(figsize=style['figsize'])
        plt.plot(fedprox_stats['proximal_terms'], 
                color=algo_styles['FedProx']['color'], 
                marker=algo_styles['FedProx']['marker'],
                linewidth=style['linewidth'],
                markersize=style['marker_size'])
        plt.title(f'FedProx Proximal Term (μ={fedprox_stats.get("mu", 0.01)})')
        plt.xlabel('Round')
        plt.ylabel('Proximal Term Value')
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fedprox_proximal_term.{style['save_format']}", dpi=style['dpi'])
        plt.close()

    # 12. 跨区域性能评估 (如果可用)
    if all(hasattr(exp, 'cross_region_performance') for exp in [fedprox_exp, fedavg_exp, similarity_exp] if exp is not None):
        try:
            # 获取所有区域
            all_regions = sorted(list(set().union(
                *(getattr(exp, 'cross_region_performance', {}).keys() for exp in [fedprox_exp, fedavg_exp, similarity_exp] if exp is not None)
            )))
            
            if all_regions:
                # 为每个算法创建热图
                for name, exp, color in [
                    ('FedProx', fedprox_exp, algo_styles['FedProx']['color']), 
                    ('FedAvg', fedavg_exp, algo_styles['FedAvg']['color']), 
                    ('Similarity', similarity_exp, algo_styles['Similarity']['color'])
                ]:
                    if exp is not None and hasattr(exp, 'cross_region_performance'):
                        plt.figure(figsize=(10, 8))
                        perf = exp.cross_region_performance
                        
                        # 转换为矩阵形式
                        perf_matrix = np.zeros((len(all_regions), len(all_regions)))
                        
                        for i, model_region in enumerate(all_regions):
                            for j, test_region in enumerate(all_regions):
                                if model_region in perf and test_region in perf.get(model_region, {}):
                                    perf_matrix[i, j] = perf[model_region][test_region]
                        
                        # 创建自定义colormap
                        cmap = plt.cm.get_cmap('YlGnBu')
                        
                        sns.heatmap(perf_matrix, annot=True, fmt=".1f", cmap=cmap,
                                   xticklabels=all_regions, yticklabels=all_regions)
                        plt.title(f'{name} Cross-Region Performance (%)')
                        plt.xlabel('Test Region')
                        plt.ylabel('Model Region')
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/{name.lower()}_cross_region_performance.{style['save_format']}", dpi=style['dpi'])
                        plt.close()
                
                # 创建对比热图 (Similarity vs FedAvg, FedProx vs FedAvg)
                if fedavg_exp is not None and similarity_exp is not None:
                    plt.figure(figsize=(10, 8))
                    fedavg_perf = getattr(fedavg_exp, 'cross_region_performance', {})
                    similarity_perf = getattr(similarity_exp, 'cross_region_performance', {})
                    
                    diff_matrix = np.zeros((len(all_regions), len(all_regions)))
                    
                    for i, model_region in enumerate(all_regions):
                        for j, test_region in enumerate(all_regions):
                            sim_val = 0
                            fed_val = 0
                            
                            if model_region in similarity_perf and test_region in similarity_perf.get(model_region, {}):
                                sim_val = similarity_perf[model_region][test_region]
                                
                            if model_region in fedavg_perf and test_region in fedavg_perf.get(model_region, {}):
                                fed_val = fedavg_perf[model_region][test_region]
                                
                            diff_matrix[i, j] = sim_val - fed_val
                    
                    sns.heatmap(diff_matrix, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                               xticklabels=all_regions, yticklabels=all_regions)
                    plt.title('Performance Difference (Similarity - FedAvg) (%)')
                    plt.xlabel('Test Region')
                    plt.ylabel('Model Region')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/similarity_fedavg_performance_diff.{style['save_format']}", dpi=style['dpi'])
                    plt.close()
                
                # FedProx vs FedAvg对比热图
                if fedavg_exp is not None and fedprox_exp is not None:
                    plt.figure(figsize=(10, 8))
                    fedavg_perf = getattr(fedavg_exp, 'cross_region_performance', {})
                    fedprox_perf = getattr(fedprox_exp, 'cross_region_performance', {})
                    
                    diff_matrix = np.zeros((len(all_regions), len(all_regions)))
                    
                    for i, model_region in enumerate(all_regions):
                        for j, test_region in enumerate(all_regions):
                            prox_val = 0
                            fed_val = 0
                            
                            if model_region in fedprox_perf and test_region in fedprox_perf.get(model_region, {}):
                                prox_val = fedprox_perf[model_region][test_region]
                                
                            if model_region in fedavg_perf and test_region in fedavg_perf.get(model_region, {}):
                                fed_val = fedavg_perf[model_region][test_region]
                                
                            diff_matrix[i, j] = prox_val - fed_val
                    
                    sns.heatmap(diff_matrix, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                               xticklabels=all_regions, yticklabels=all_regions)
                    plt.title('Performance Difference (FedProx - FedAvg) (%)')
                    plt.xlabel('Test Region')
                    plt.ylabel('Model Region')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/fedprox_fedavg_performance_diff.{style['save_format']}", dpi=style['dpi'])
                    plt.close()
        except Exception as e:
            logger.error(f"绘制跨区域性能热图时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    logger.info(f"图表生成完成，保存在 {output_dir}/ 目录")

def create_comparison_plots_extended(fedprox_stats, fedavg_stats, similarity_stats, output_dir, 
                                   fedprox_exp=None, fedavg_exp=None, similarity_exp=None,
                                   custom_style=None, show_grid=True, figure_format='png', dpi=150):
    """
    创建扩展的对比图表，包含分类性能指标
    
    Args:
        fedprox_stats: FedProx实验统计数据
        fedavg_stats: FedAvg实验统计数据
        similarity_stats: 相似度分组实验统计数据
        output_dir: 输出目录
        fedprox_exp: FedProx实验对象(可选)
        fedavg_exp: FedAvg实验对象(可选)
        similarity_exp: 相似度分组实验对象(可选)
        custom_style: 自定义图表样式字典
        show_grid: 是否显示网格
        figure_format: 图片格式(png, pdf, svg等)
        dpi: 图像DPI
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from math import pi
    
    # 获取实际参与的卫星数量
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedProx: {fedprox_sats:.1f}, FedAvg: {fedavg_sats:.1f}, Similarity: {similarity_sats:.1f} satellites)"
    
    # 应用自定义样式
    style = {
        'figsize': (10, 6),
        'title_fontsize': 14,
        'label_fontsize': 12,
        'tick_fontsize': 10,
        'legend_fontsize': 10,
        'linewidth': 2,
        'marker_size': 6,
        'grid_alpha': 0.3,
        'grid_linestyle': '--',
        'save_format': figure_format,
        'dpi': dpi
    }
    
    if custom_style:
        style.update(custom_style)
    
    # 设置默认样式
    plt.rcParams.update({
        'font.size': style['label_fontsize'],
        'axes.titlesize': style['title_fontsize'],
        'axes.labelsize': style['label_fontsize'],
        'xtick.labelsize': style['tick_fontsize'],
        'ytick.labelsize': style['tick_fontsize'],
        'legend.fontsize': style['legend_fontsize']
    })
    
    # 定义算法颜色和标记
    algo_styles = {
        'FedProx': {'color': 'g', 'marker': 'o', 'label': 'FedProx'},
        'FedAvg': {'color': 'b', 'marker': 's', 'label': 'FedAvg'},
        'Similarity': {'color': 'r', 'marker': '^', 'label': 'Similarity Grouping'}
    }
    
    # 辅助函数：安全获取指标数据
    def safe_get_metric(stats, metric_name, default_value=None):
        """安全获取指标数据，如果不存在则返回默认值"""
        if metric_name in stats and stats[metric_name]:
            return stats[metric_name]
        elif default_value is not None:
            return default_value
        else:
            # 如果没有指定默认值，返回与accuracies等长的零列表
            acc_len = len(stats.get('accuracies', []))
            return [0] * acc_len if acc_len > 0 else [0]
    
    # =============================================================================
    # 1. 准确率对比 (保持原有)
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['accuracies'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['accuracies'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['accuracies'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Accuracy Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 2. F1值对比 (新增)
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    
    fedprox_f1 = safe_get_metric(fedprox_stats, 'f1_macros')
    fedavg_f1 = safe_get_metric(fedavg_stats, 'f1_macros')
    similarity_f1 = safe_get_metric(similarity_stats, 'f1_macros')
    
    if any(sum(data) > 0 for data in [fedprox_f1, fedavg_f1, similarity_f1]):
        plt.plot(fedprox_f1, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(fedavg_f1, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(similarity_f1, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.title(f'F1-Score (Macro) Comparison {title_suffix}')
        plt.xlabel('Round')
        plt.ylabel('F1-Score (%)')
        plt.legend()
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    else:
        plt.text(0.5, 0.5, 'F1-Score data not available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title(f'F1-Score (Macro) Comparison {title_suffix}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 3. 精确率对比 (新增)
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    
    fedprox_precision = safe_get_metric(fedprox_stats, 'precision_macros')
    fedavg_precision = safe_get_metric(fedavg_stats, 'precision_macros')
    similarity_precision = safe_get_metric(similarity_stats, 'precision_macros')
    
    if any(sum(data) > 0 for data in [fedprox_precision, fedavg_precision, similarity_precision]):
        plt.plot(fedprox_precision, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(fedavg_precision, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(similarity_precision, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.title(f'Precision (Macro) Comparison {title_suffix}')
        plt.xlabel('Round')
        plt.ylabel('Precision (%)')
        plt.legend()
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    else:
        plt.text(0.5, 0.5, 'Precision data not available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title(f'Precision (Macro) Comparison {title_suffix}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 4. 召回率对比 (新增)
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    
    fedprox_recall = safe_get_metric(fedprox_stats, 'recall_macros')
    fedavg_recall = safe_get_metric(fedavg_stats, 'recall_macros')
    similarity_recall = safe_get_metric(similarity_stats, 'recall_macros')
    
    if any(sum(data) > 0 for data in [fedprox_recall, fedavg_recall, similarity_recall]):
        plt.plot(fedprox_recall, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(fedavg_recall, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(similarity_recall, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.title(f'Recall (Macro) Comparison {title_suffix}')
        plt.xlabel('Round')
        plt.ylabel('Recall (%)')
        plt.legend()
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    else:
        plt.text(0.5, 0.5, 'Recall data not available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title(f'Recall (Macro) Comparison {title_suffix}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 5. 所有分类指标综合对比 (四宫格)
    # =============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Classification Metrics Comparison {title_suffix}', fontsize=16)
    
    # 准确率
    ax1.plot(fedprox_stats['accuracies'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    ax1.plot(fedavg_stats['accuracies'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    ax1.plot(similarity_stats['accuracies'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    if show_grid:
        ax1.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # F1值
    if any(sum(data) > 0 for data in [fedprox_f1, fedavg_f1, similarity_f1]):
        ax2.plot(fedprox_f1, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax2.plot(fedavg_f1, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax2.plot(similarity_f1, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    ax2.set_title('F1-Score (Macro)')
    ax2.set_ylabel('F1-Score (%)')
    ax2.legend()
    if show_grid:
        ax2.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 精确率
    if any(sum(data) > 0 for data in [fedprox_precision, fedavg_precision, similarity_precision]):
        ax3.plot(fedprox_precision, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax3.plot(fedavg_precision, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax3.plot(similarity_precision, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    ax3.set_title('Precision (Macro)')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Precision (%)')
    ax3.legend()
    if show_grid:
        ax3.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 召回率
    if any(sum(data) > 0 for data in [fedprox_recall, fedavg_recall, similarity_recall]):
        ax4.plot(fedprox_recall, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax4.plot(fedavg_recall, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        ax4.plot(similarity_recall, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    ax4.set_title('Recall (Macro)')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Recall (%)')
    ax4.legend()
    if show_grid:
        ax4.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_classification_metrics.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 6. 分类性能雷达图对比（最终指标）
    # =============================================================================
    try:
        # 获取最终指标
        def get_final_metrics(stats):
            metrics = {}
            if 'accuracies' in stats and stats['accuracies']:
                metrics['Accuracy'] = stats['accuracies'][-1]
            if 'f1_macros' in stats and stats['f1_macros'] and sum(stats['f1_macros']) > 0:
                metrics['F1'] = stats['f1_macros'][-1]
            if 'precision_macros' in stats and stats['precision_macros'] and sum(stats['precision_macros']) > 0:
                metrics['Precision'] = stats['precision_macros'][-1]
            if 'recall_macros' in stats and stats['recall_macros'] and sum(stats['recall_macros']) > 0:
                metrics['Recall'] = stats['recall_macros'][-1]
            return metrics
        
        fedprox_final = get_final_metrics(fedprox_stats)
        fedavg_final = get_final_metrics(fedavg_stats)
        similarity_final = get_final_metrics(similarity_stats)
        
        if fedprox_final and fedavg_final and similarity_final:
            # 确保所有算法都有相同的指标
            common_metrics = set(fedprox_final.keys()) & set(fedavg_final.keys()) & set(similarity_final.keys())
            
            if common_metrics and len(common_metrics) >= 2:  # 至少要有2个指标才绘制雷达图
                metrics = list(common_metrics)
                
                # 创建雷达图
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                # 计算角度
                angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
                angles += angles[:1]  # 闭合图形
                
                # 绘制每个算法的雷达图
                for name, final_metrics, color in [
                    ('FedProx', fedprox_final, algo_styles['FedProx']['color']),
                    ('FedAvg', fedavg_final, algo_styles['FedAvg']['color']),
                    ('Similarity', similarity_final, algo_styles['Similarity']['color'])
                ]:
                    values = [final_metrics[metric] for metric in metrics]
                    values += values[:1]  # 闭合图形
                    
                    ax.plot(angles, values, color=color, linewidth=style['linewidth'], label=name)
                    ax.fill(angles, values, color=color, alpha=0.1)
                
                # 添加标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 100)
                ax.set_title('Final Classification Performance Comparison', 
                           size=16, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/radar_comparison.{style['save_format']}", 
                           dpi=style['dpi'], bbox_inches='tight')
                plt.close()
    except Exception as e:
        logger.warning(f"无法生成雷达图: {str(e)}")
    
    # =============================================================================
    # 7. 损失函数对比 (保持原有)
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['losses'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['losses'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['losses'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Loss Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 8. 能耗对比 - 训练能耗
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['training_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Training Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 9. 能耗对比 - 通信能耗
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    if 'energy_stats' in fedprox_stats and 'communication_energy' in fedprox_stats['energy_stats']:
        plt.plot(fedprox_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'energy_stats' in fedavg_stats and 'communication_energy' in fedavg_stats['energy_stats']:
        plt.plot(fedavg_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'energy_stats' in similarity_stats and 'communication_energy' in similarity_stats['energy_stats']:
        plt.plot(similarity_stats['energy_stats']['communication_energy'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Communication Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 10. 能耗对比 - 总能耗
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['energy_stats']['total_energy'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Total Energy Consumption {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_energy_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # =============================================================================
    # 11. F1值能效比对比(F1值/能耗)
    # =============================================================================
    if any(sum(data) > 0 for data in [fedprox_f1, fedavg_f1, similarity_f1]):
        plt.figure(figsize=style['figsize'])
        
        fedprox_f1_efficiency = [f1 / (energy + 1e-10) for f1, energy in 
                                zip(fedprox_f1, fedprox_stats['energy_stats']['total_energy'])]
        fedavg_f1_efficiency = [f1 / (energy + 1e-10) for f1, energy in 
                               zip(fedavg_f1, fedavg_stats['energy_stats']['total_energy'])]
        similarity_f1_efficiency = [f1 / (energy + 1e-10) for f1, energy in 
                                   zip(similarity_f1, similarity_stats['energy_stats']['total_energy'])]
        
        plt.plot(fedprox_f1_efficiency, 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(fedavg_f1_efficiency, 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.plot(similarity_f1_efficiency, 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
        plt.title(f'F1-Score Energy Efficiency {title_suffix}')
        plt.xlabel('Round')
        plt.ylabel('F1-Score per Energy (%/Wh)')
        plt.legend()
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_energy_efficiency.{style['save_format']}", dpi=style['dpi'])
        plt.close()
    
    # =============================================================================
    # 12. 活跃卫星数量对比
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    plt.plot(fedprox_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Number of Training Satellites {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_satellites_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()

    # =============================================================================
    # 13. 通信开销对比
    # =============================================================================
    plt.figure(figsize=style['figsize'])
    fedprox_comm = calculate_communication_overhead(fedprox_stats)
    fedavg_comm = calculate_communication_overhead(fedavg_stats)
    similarity_comm = calculate_communication_overhead(similarity_stats)
    
    plt.plot(fedprox_comm, 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_comm, 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(similarity_comm, 
             color=algo_styles['Similarity']['color'], 
             marker=algo_styles['Similarity']['marker'], 
             label=algo_styles['Similarity']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.title(f'Cumulative Communication Overhead {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Communication Energy (Wh)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/communication_overhead.{style['save_format']}", dpi=style['dpi'])
    plt.close()

    # =============================================================================
    # 14. 如果有FedProx特有的指标，绘制FedProx的接近性项
    # =============================================================================
    if 'proximal_terms' in fedprox_stats and fedprox_stats['proximal_terms']:
        plt.figure(figsize=style['figsize'])
        plt.plot(fedprox_stats['proximal_terms'], 
                color=algo_styles['FedProx']['color'], 
                marker=algo_styles['FedProx']['marker'],
                linewidth=style['linewidth'],
                markersize=style['marker_size'])
        plt.title(f'FedProx Proximal Term (μ={fedprox_stats.get("mu", 0.01)})')
        plt.xlabel('Round')
        plt.ylabel('Proximal Term Value')
        if show_grid:
            plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fedprox_proximal_term.{style['save_format']}", dpi=style['dpi'])
        plt.close()

    # =============================================================================
    # 15. 最终性能对比柱状图
    # =============================================================================
    try:
        # 获取最终性能指标
        final_metrics = {}
        algorithms = ['FedProx', 'FedAvg', 'Similarity Grouping']
        stats_list = [fedprox_stats, fedavg_stats, similarity_stats]
        
        # 收集最终指标
        metrics_to_compare = ['accuracies', 'f1_macros', 'precision_macros', 'recall_macros']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Final Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (metric, name) in enumerate(zip(metrics_to_compare, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            for stats in stats_list:
                if metric in stats and stats[metric] and len(stats[metric]) > 0:
                    values.append(stats[metric][-1])  # 最终值
                else:
                    values.append(0)
            
            colors = [algo_styles[algo]['color'] for algo in ['FedProx', 'FedAvg', 'Similarity']]
            bars = ax.bar(algorithms, values, color=colors, alpha=0.7)
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'Final {name}')
            ax.set_ylabel(f'{name} (%)')
            if show_grid:
                ax.grid(axis='y', alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
            
            # 旋转x轴标签以避免重叠
            ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/final_performance_comparison.{style['save_format']}", 
                   dpi=style['dpi'], bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"无法生成最终性能对比图: {str(e)}")
            
    logger.info(f"扩展图表生成完成，保存在 {output_dir}/ 目录")
def save_experiment_data(output_dir, fedprox_stats, fedavg_stats, similarity_stats, timestamp):
    """保存实验数据，以便后续重新绘图"""
    data_dir = os.path.join(output_dir, 'raw_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 将统计数据转换为可序列化的格式
    def prepare_for_serialization(stats_dict):
        # 深复制以避免修改原始数据
        serializable_stats = copy.deepcopy(stats_dict)
        
        # 将numpy数组转换为列表
        for key, value in serializable_stats.items():
            if isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_stats[key][k] = v.tolist()
        return serializable_stats
    
    # 准备数据
    fedprox_data = prepare_for_serialization(fedprox_stats)
    fedavg_data = prepare_for_serialization(fedavg_stats)
    similarity_data = prepare_for_serialization(similarity_stats)
    
    # 保存为pickle格式(包含完整数据)
    with open(os.path.join(data_dir, 'experiment_data.pkl'), 'wb') as f:
        pickle.dump({
            'fedprox': fedprox_data,
            'fedavg': fedavg_data,
            'similarity': similarity_data,
            'timestamp': timestamp,
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'description': '公平对比实验数据'
            }
        }, f)
    
    # 同时保存为JSON格式(便于查看和跨平台使用)
    try:
        with open(os.path.join(data_dir, 'experiment_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'fedprox': fedprox_data,
                'fedavg': fedavg_data,
                'similarity': similarity_data,
                'timestamp': timestamp,
                'metadata': {
                    'creation_time': datetime.now().isoformat(),
                    'description': '公平对比实验数据'
                }
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"无法保存为JSON格式: {str(e)}")
    
    # 保存实验配置
    try:
        with open(os.path.join(data_dir, 'fedprox_config.yaml'), 'w') as f:
            with open(f"configs/temp/fedprox_{fedprox_stats['satellite_stats']['training_satellites'][0]}sats.yaml", 'r') as src:
                f.write(src.read())
                
        with open(os.path.join(data_dir, 'fedavg_config.yaml'), 'w') as f:
            with open(f"configs/temp/fedavg_{fedavg_stats['satellite_stats']['training_satellites'][0]}sats.yaml", 'r') as src:
                f.write(src.read())
                
        with open(os.path.join(data_dir, 'similarity_config.yaml'), 'w') as f:
            with open("configs/similarity_grouping_config.yaml", 'r') as src:
                f.write(src.read())
    except Exception as e:
        logger.warning(f"无法保存配置文件: {str(e)}")
    
    # 创建元数据文件，记录关键指标，便于快速查看
    with open(os.path.join(data_dir, 'metadata.txt'), 'w') as f:
        f.write(f"实验时间: {timestamp}\n\n")
        
        f.write("平均卫星数量:\n")
        f.write(f"  FedProx: {np.mean(fedprox_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  FedAvg: {np.mean(fedavg_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  相似度分组: {np.mean(similarity_stats['satellite_stats']['training_satellites']):.2f}\n\n")
        
        f.write("最终准确率:\n")
        f.write(f"  FedProx: {max(fedprox_stats['accuracies']):.2f}%\n")
        f.write(f"  FedAvg: {max(fedavg_stats['accuracies']):.2f}%\n")
        f.write(f"  相似度分组: {max(similarity_stats['accuracies']):.2f}%\n")
    
    logger.info(f"实验数据已保存到 {data_dir}/")

def load_experiment_data(data_dir):
    """加载保存的实验数据"""
    # 优先尝试加载pickle格式
    pickle_path = os.path.join(data_dir, 'raw_data', 'experiment_data.pkl')
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"无法加载pickle数据: {str(e)}")
    
    # 尝试加载JSON格式
    json_path = os.path.join(data_dir, 'raw_data', 'experiment_data.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"无法加载JSON数据: {str(e)}")
    
    raise FileNotFoundError(f"在 {data_dir}/raw_data/ 中找不到实验数据文件")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行公平对比实验或重新绘制已有实验的图表')
    parser.add_argument('--target-sats', type=int, default=0,
                      help='目标卫星数量 (0表示使用相似度分组的平均卫星数)')
    parser.add_argument('--fedprox-mu', type=float, default=0.01,
                      help='FedProx的接近性参数μ')
    parser.add_argument('--config-dir', type=str, default='configs',
                      help='配置文件目录')
    
    # 添加重新绘图相关参数
    parser.add_argument('--replot', action='store_true',
                      help='重新绘图模式，不运行实验，只加载已有数据并重新绘制图表')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='数据目录(用于重新绘图模式)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='图表输出目录(默认为data-dir，仅用于重新绘图模式)')
    parser.add_argument('--format', type=str, default='png',
                      choices=['png', 'pdf', 'svg', 'jpg'],
                      help='图表保存格式(仅用于重新绘图模式)')
    parser.add_argument('--dpi', type=int, default=150,
                      help='图表DPI(仅用于重新绘图模式)')
    parser.add_argument('--no-grid', action='store_true',
                      help='不显示网格(仅用于重新绘图模式)')
    
    return parser.parse_args()

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
        f.write("# 公平对比报告: FedProx vs FedAvg vs 相似度分组\n\n")
        
        f.write("## 实验设置\n")
        f.write(f"- FedProx 参数 μ: {fedprox_stats.get('mu', 0.01)}\n")
        f.write(f"- 总轮次: {len(fedprox_stats['accuracies'])}\n\n")
        
        f.write("## 参与卫星数量\n")
        f.write(f"- FedProx 平均训练卫星数: {fedprox_avg_sats:.2f}\n")
        f.write(f"- FedAvg 平均训练卫星数: {fedavg_avg_sats:.2f}\n")
        f.write(f"- 相似度分组平均训练卫星数: {similarity_avg_sats:.2f}\n\n")
        
        f.write("## 准确率性能\n")
        f.write(f"- FedProx 最高准确率: {fedprox_max_acc:.2f}%\n")
        f.write(f"- FedAvg 最高准确率: {fedavg_max_acc:.2f}%\n")
        f.write(f"- 相似度分组最高准确率: {similarity_max_acc:.2f}%\n\n")
        
        f.write(f"- FedProx vs FedAvg: {fedprox_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- FedProx vs 相似度分组: {fedprox_max_acc - similarity_max_acc:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedAvg: {similarity_max_acc - fedavg_max_acc:+.2f}%\n\n")
        
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
        f.write("- **优势**: 在非IID数据分布下更稳定的收敛性，对数据异质性更鲁棒。\n")
        f.write("- **劣势**: 额外的计算开销，需要调整接近性参数μ。\n\n")
        
        f.write("### FedAvg\n")
        f.write("- **优势**: 简单，计算开销低。\n")
        f.write("- **劣势**: 在非IID数据上可能发散，对异质性数据敏感。\n\n")
        
        f.write("### 相似度分组\n")
        f.write("- **优势**: 高效的资源利用，适应数据分布特点，每卫星性能更好。\n")
        f.write("- **劣势**: 需要计算数据相似度的开销，实现更复杂。\n\n")
        
        # 确定哪种方法表现最佳
        best_method = ""
        if similarity_max_acc > fedprox_max_acc and similarity_max_acc > fedavg_max_acc:
            best_method = "相似度分组在准确率上表现最好"
        elif fedprox_max_acc > similarity_max_acc and fedprox_max_acc > fedavg_max_acc:
            best_method = "FedProx在准确率上表现最好"
        elif fedavg_max_acc > similarity_max_acc and fedavg_max_acc > fedprox_max_acc:
            best_method = "FedAvg在准确率上表现最好"
        else:
            best_method = "各方法准确率相近"
            
        best_efficiency = ""
        if similarity_efficiency > fedprox_efficiency and similarity_efficiency > fedavg_efficiency:
            best_efficiency = "相似度分组在资源效率上表现最好"
        elif fedprox_efficiency > similarity_efficiency and fedprox_efficiency > fedavg_efficiency:
            best_efficiency = "FedProx在资源效率上表现最好"
        elif fedavg_efficiency > similarity_efficiency and fedavg_efficiency > fedprox_efficiency:
            best_efficiency = "FedAvg在资源效率上表现最好"
        else:
            best_efficiency = "各方法资源效率相近"
            
        f.write("### 结论\n")
        f.write(f"{best_method}，{best_efficiency}。在卫星网络环境中，")
        
        if similarity_max_acc >= fedprox_max_acc and similarity_efficiency > fedprox_efficiency:
            f.write("相似度分组算法综合表现最佳，特别是在资源受限的场景下。\n")
        elif fedprox_max_acc > similarity_max_acc and fedprox_efficiency >= similarity_efficiency:
            f.write("FedProx算法综合表现最佳，特别是在数据异质性较高的场景下。\n")
        else:
            f.write("需要根据具体场景权衡选择合适的联邦学习算法。\n")

def generate_comparison_report_extended(fedprox_stats, fedavg_stats, similarity_stats, output_path):
    """生成包含分类指标的详细对比报告"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 计算各种指标
    def get_metrics(stats, metric_name):
        if metric_name in stats and stats[metric_name]:
            return max(stats[metric_name])
        return 0.0
    
    # 准确率
    fedprox_max_acc = get_metrics(fedprox_stats, 'accuracies')
    fedavg_max_acc = get_metrics(fedavg_stats, 'accuracies')
    similarity_max_acc = get_metrics(similarity_stats, 'accuracies')
    
    # F1值
    fedprox_max_f1 = get_metrics(fedprox_stats, 'f1_macros')
    fedavg_max_f1 = get_metrics(fedavg_stats, 'f1_macros')
    similarity_max_f1 = get_metrics(similarity_stats, 'f1_macros')
    
    # 精确率
    fedprox_max_precision = get_metrics(fedprox_stats, 'precision_macros')
    fedavg_max_precision = get_metrics(fedavg_stats, 'precision_macros')
    similarity_max_precision = get_metrics(similarity_stats, 'precision_macros')
    
    # 召回率
    fedprox_max_recall = get_metrics(fedprox_stats, 'recall_macros')
    fedavg_max_recall = get_metrics(fedavg_stats, 'recall_macros')
    similarity_max_recall = get_metrics(similarity_stats, 'recall_macros')
    
    # 卫星数量
    fedprox_avg_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 能耗
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算效率指标
    def safe_divide(a, b):
        return a / b if b > 0 else 0
    
    # 每卫星F1值
    fedprox_f1_per_sat = safe_divide(fedprox_max_f1, fedprox_avg_sats)
    fedavg_f1_per_sat = safe_divide(fedavg_max_f1, fedavg_avg_sats)
    similarity_f1_per_sat = safe_divide(similarity_max_f1, similarity_avg_sats)
    
    # 能源效率（F1/能耗）
    fedprox_f1_per_energy = safe_divide(fedprox_max_f1, fedprox_energy)
    fedavg_f1_per_energy = safe_divide(fedavg_max_f1, fedavg_energy)
    similarity_f1_per_energy = safe_divide(similarity_max_f1, similarity_energy)
    
    # 计算收敛速度
    def calculate_convergence(values, target_percentage=0.9):
        if not values:
            return float('inf')
        target = max(values) * target_percentage
        for i, val in enumerate(values):
            if val >= target:
                return i + 1
        return len(values)
    
    fedprox_acc_convergence = calculate_convergence(fedprox_stats['accuracies'])
    fedavg_acc_convergence = calculate_convergence(fedavg_stats['accuracies'])
    similarity_acc_convergence = calculate_convergence(similarity_stats['accuracies'])
    
    fedprox_f1_convergence = calculate_convergence(fedprox_stats.get('f1_macros', []))
    fedavg_f1_convergence = calculate_convergence(fedavg_stats.get('f1_macros', []))
    similarity_f1_convergence = calculate_convergence(similarity_stats.get('f1_macros', []))
    
    # 生成报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 公平对比报告: FedProx vs FedAvg vs 相似度分组\n\n")
        f.write("## 实验设置\n")
        f.write(f"- FedProx 参数 μ: {fedprox_stats.get('mu', 0.01)}\n")
        f.write(f"- 总轮次: {len(fedprox_stats['accuracies'])}\n")
        f.write(f"- 数据集: 网络流量分类（二分类：恶意/良性）\n\n")
        
        f.write("## 参与卫星数量\n")
        f.write(f"- FedProx 平均训练卫星数: {fedprox_avg_sats:.2f}\n")
        f.write(f"- FedAvg 平均训练卫星数: {fedavg_avg_sats:.2f}\n")
        f.write(f"- 相似度分组平均训练卫星数: {similarity_avg_sats:.2f}\n\n")
        
        f.write("## 分类性能指标\n\n")
        
        f.write("### 准确率 (Accuracy)\n")
        f.write(f"- FedProx 最高准确率: {fedprox_max_acc:.2f}%\n")
        f.write(f"- FedAvg 最高准确率: {fedavg_max_acc:.2f}%\n")
        f.write(f"- 相似度分组最高准确率: {similarity_max_acc:.2f}%\n")
        f.write(f"- **FedProx vs FedAvg**: {fedprox_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- **相似度分组 vs FedAvg**: {similarity_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- **FedProx vs 相似度分组**: {fedprox_max_acc - similarity_max_acc:+.2f}%\n\n")
        
        f.write("### F1值 (Macro Average)\n")
        f.write(f"- FedProx 最高F1值: {fedprox_max_f1:.2f}%\n")
        f.write(f"- FedAvg 最高F1值: {fedavg_max_f1:.2f}%\n")
        f.write(f"- 相似度分组最高F1值: {similarity_max_f1:.2f}%\n")
        f.write(f"- **FedProx vs FedAvg**: {fedprox_max_f1 - fedavg_max_f1:+.2f}%\n")
        f.write(f"- **相似度分组 vs FedAvg**: {similarity_max_f1 - fedavg_max_f1:+.2f}%\n")
        f.write(f"- **FedProx vs 相似度分组**: {fedprox_max_f1 - similarity_max_f1:+.2f}%\n\n")
        
        f.write("### 精确率 (Precision - Macro Average)\n")
        f.write(f"- FedProx 最高精确率: {fedprox_max_precision:.2f}%\n")
        f.write(f"- FedAvg 最高精确率: {fedavg_max_precision:.2f}%\n")
        f.write(f"- 相似度分组最高精确率: {similarity_max_precision:.2f}%\n")
        f.write(f"- **FedProx vs FedAvg**: {fedprox_max_precision - fedavg_max_precision:+.2f}%\n")
        f.write(f"- **相似度分组 vs FedAvg**: {similarity_max_precision - fedavg_max_precision:+.2f}%\n")
        f.write(f"- **FedProx vs 相似度分组**: {fedprox_max_precision - similarity_max_precision:+.2f}%\n\n")
        
        f.write("### 召回率 (Recall - Macro Average)\n")
        f.write(f"- FedProx 最高召回率: {fedprox_max_recall:.2f}%\n")
        f.write(f"- FedAvg 最高召回率: {fedavg_max_recall:.2f}%\n")
        f.write(f"- 相似度分组最高召回率: {similarity_max_recall:.2f}%\n")
        f.write(f"- **FedProx vs FedAvg**: {fedprox_max_recall - fedavg_max_recall:+.2f}%\n")
        f.write(f"- **相似度分组 vs FedAvg**: {similarity_max_recall - fedavg_max_recall:+.2f}%\n")
        f.write(f"- **FedProx vs 相似度分组**: {fedprox_max_recall - similarity_max_recall:+.2f}%\n\n")
        
        f.write("## 能耗分析\n")
        f.write(f"- FedProx 总能耗: {fedprox_energy:.2f} Wh\n")
        f.write(f"- FedAvg 总能耗: {fedavg_energy:.2f} Wh\n")
        f.write(f"- 相似度分组总能耗: {similarity_energy:.2f} Wh\n\n")
        
        f.write("## 资源效率指标\n\n")
        
        f.write("### 每卫星F1值效率\n")
        f.write(f"- FedProx: {fedprox_f1_per_sat:.2f}% F1/satellite\n")
        f.write(f"- FedAvg: {fedavg_f1_per_sat:.2f}% F1/satellite\n")
        f.write(f"- 相似度分组: {similarity_f1_per_sat:.2f}% F1/satellite\n\n")
        
        f.write("### 能源效率 (F1值/能耗)\n")
        f.write(f"- FedProx: {fedprox_f1_per_energy:.4f}% F1/Wh\n")
        f.write(f"- FedAvg: {fedavg_f1_per_energy:.4f}% F1/Wh\n")
        f.write(f"- 相似度分组: {similarity_f1_per_energy:.4f}% F1/Wh\n\n")
        
        f.write("## 收敛性能\n\n")
        
        f.write("### 准确率收敛速度 (达到90%最高值的轮次)\n")
        f.write(f"- FedProx: {fedprox_acc_convergence} 轮\n")
        f.write(f"- FedAvg: {fedavg_acc_convergence} 轮\n")
        f.write(f"- 相似度分组: {similarity_acc_convergence} 轮\n\n")
        
        f.write("### F1值收敛速度 (达到90%最高值的轮次)\n")
        f.write(f"- FedProx: {fedprox_f1_convergence} 轮\n")
        f.write(f"- FedAvg: {fedavg_f1_convergence} 轮\n")
        f.write(f"- 相似度分组: {similarity_f1_convergence} 轮\n\n")
        
        f.write("## 算法特性分析\n\n")
        
        f.write("### FedProx\n")
        f.write("- **优势**: \n")
        f.write("  - 接近性正则化项提高非IID数据的稳定性\n")
        f.write("  - 在数据异质性高的环境下表现稳定\n")
        f.write("  - 有助于防止客户端偏移过大\n")
        f.write("- **劣势**: \n")
        f.write("  - 额外的计算开销（接近性项计算）\n")
        f.write("  - 需要调整超参数μ\n")
        f.write("  - 可能收敛速度较慢\n\n")
        
        f.write("### FedAvg\n")
        f.write("- **优势**: \n")
        f.write("  - 算法简单，计算开销低\n")
        f.write("  - 实现简单，易于部署\n")
        f.write("  - 在IID数据上表现良好\n")
        f.write("- **劣势**: \n")
        f.write("  - 在非IID数据上可能发散\n")
        f.write("  - 对数据异质性敏感\n")
        f.write("  - 可能出现权重偏移问题\n\n")
        
        f.write("### 相似度分组\n")
        f.write("- **优势**: \n")
        f.write("  - 高效的资源利用（更少卫星达到相近性能）\n")
        f.write("  - 适应数据分布特点，减少通信开销\n")
        f.write("  - 基于相似度的智能分组策略\n")
        f.write("- **劣势**: \n")
        f.write("  - 相似度计算带来额外开销\n")
        f.write("  - 实现复杂度较高\n")
        f.write("  - 需要定期重新计算分组\n\n")
        
        f.write("## 综合评估与建议\n\n")
        
        # 确定最佳方法
        metrics_comparison = {
            'FedProx': {'f1': fedprox_max_f1, 'efficiency': fedprox_f1_per_sat, 'energy_eff': fedprox_f1_per_energy},
            'FedAvg': {'f1': fedavg_max_f1, 'efficiency': fedavg_f1_per_sat, 'energy_eff': fedavg_f1_per_energy},
            'Similarity': {'f1': similarity_max_f1, 'efficiency': similarity_f1_per_sat, 'energy_eff': similarity_f1_per_energy}
        }
        
        # 找出各项指标的最佳方法
        best_f1 = max(metrics_comparison.items(), key=lambda x: x[1]['f1'])
        best_efficiency = max(metrics_comparison.items(), key=lambda x: x[1]['efficiency'])
        best_energy_eff = max(metrics_comparison.items(), key=lambda x: x[1]['energy_eff'])
        
        f.write("### 性能排名\n")
        f.write(f"- **最佳F1值**: {best_f1[0]} ({best_f1[1]['f1']:.2f}%)\n")
        f.write(f"- **最佳资源效率**: {best_efficiency[0]} ({best_efficiency[1]['efficiency']:.2f}% F1/satellite)\n")
        f.write(f"- **最佳能源效率**: {best_energy_eff[0]} ({best_energy_eff[1]['energy_eff']:.4f}% F1/Wh)\n\n")
        
        f.write("### 应用场景建议\n\n")
        f.write("**1. 资源受限场景（卫星数量有限）**\n")
        if similarity_f1_per_sat >= fedprox_f1_per_sat and similarity_f1_per_sat >= fedavg_f1_per_sat:
            f.write("   - **推荐**: 相似度分组算法\n")
            f.write("   - **理由**: 能用更少的卫星实现相近或更好的性能\n\n")
        else:
            f.write(f"   - **推荐**: {best_efficiency[0]}\n")
            f.write("   - **理由**: 资源利用效率最高\n\n")
        
        f.write("**2. 数据异质性高的场景**\n")
        if fedprox_max_f1 >= similarity_max_f1 and fedprox_max_f1 >= fedavg_max_f1:
            f.write("   - **推荐**: FedProx算法\n")
            f.write("   - **理由**: 接近性正则化能更好处理非IID数据\n\n")
        else:
            f.write(f"   - **推荐**: {best_f1[0]}\n")
            f.write("   - **理由**: 在复杂数据分布下表现最佳\n\n")
        
        f.write("**3. 能耗敏感场景**\n")
        f.write(f"   - **推荐**: {best_energy_eff[0]}\n")
        f.write("   - **理由**: 单位能耗获得的性能提升最高\n\n")
        
        f.write("### 最终结论\n")
        
        # 计算综合得分（考虑F1值、效率和能耗）
        def calculate_score(metrics):
            # 归一化各项指标到0-1范围
            max_f1 = max(fedprox_max_f1, fedavg_max_f1, similarity_max_f1)
            max_eff = max(fedprox_f1_per_sat, fedavg_f1_per_sat, similarity_f1_per_sat)
            max_energy_eff = max(fedprox_f1_per_energy, fedavg_f1_per_energy, similarity_f1_per_energy)
            
            norm_f1 = metrics['f1'] / max_f1 if max_f1 > 0 else 0
            norm_eff = metrics['efficiency'] / max_eff if max_eff > 0 else 0
            norm_energy_eff = metrics['energy_eff'] / max_energy_eff if max_energy_eff > 0 else 0
            
            # 加权平均 (F1值权重40%, 资源效率权重35%, 能源效率权重25%)
            return 0.4 * norm_f1 + 0.35 * norm_eff + 0.25 * norm_energy_eff
        
        scores = {name: calculate_score(metrics) for name, metrics in metrics_comparison.items()}
        best_overall = max(scores.items(), key=lambda x: x[1])
        
        f.write(f"基于综合评估，**{best_overall[0]}算法**在卫星网络流量分类任务中表现最佳，")
        f.write(f"综合得分为{best_overall[1]:.3f}。\n\n")
        
        if best_overall[0] == 'Similarity':
            f.write("相似度分组算法通过智能的资源调度和数据相似性分析，")
            f.write("在保证分类性能的同时显著提高了资源利用效率，")
            f.write("特别适合卫星网络这种资源受限的分布式环境。\n")
        elif best_overall[0] == 'FedProx':
            f.write("FedProx算法通过接近性正则化有效处理了网络流量数据的异质性，")
            f.write("在分类准确性方面表现出色，适合对分类精度要求较高的场景。\n")
        else:
            f.write("FedAvg算法作为经典方法，在计算简单性和实现便利性方面具有优势，")
            f.write("适合对实时性要求较高的应用场景。\n")
        
        f.write(f"\n---\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验数据基于: 真实网络流量数据集 (二分类任务)\n")

def create_modified_config(base_config_path, target_satellite_count, output_path):
    """
    创建修改后的配置文件，设置目标卫星数量
    """
    try:
        # 读取基础配置
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 修改传播配置
        if 'propagation' not in config:
            config['propagation'] = {}
        
        config['propagation']['max_satellites'] = target_satellite_count
        
        # 根据卫星数量调整跳数
        if target_satellite_count <= 10:
            config['propagation']['hops'] = 1
        elif target_satellite_count <= 20:
            config['propagation']['hops'] = 2
        else:
            config['propagation']['hops'] = 3
        
        # 处理client配置，移除不支持的参数
        if 'client' in config and 'optimizer' in config['client']:
            # FedProx的ClientConfig不支持optimizer参数，所以移除它
            config['client'].pop('optimizer', None)
            logger.info("从client配置中移除了'optimizer'参数")
        
        # 保存修改后的配置
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"已创建修改后的配置文件：{output_path}")
        logger.info(f"- 目标卫星数：{target_satellite_count}")
        logger.info(f"- 传播跳数：{config['propagation']['hops']}")
        
        return output_path
    except Exception as e:
        logger.error(f"创建修改后的配置文件时出错：{str(e)}")
        return base_config_path

def run_fair_comparison():
    """运行公平比较实验"""
    logger.info("=== 开始公平比较实验 ===")
    
    # 1. 运行相似度分组实验
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_stats, similarity_exp = run_experiment(
        "configs/similarity_grouping_config.yaml", 
        SimilarityGroupingExperiment
    )
    
    if not similarity_stats:
        logger.error("相似度分组实验失败")
        return
    
    # 获取相似度分组使用的卫星数量
    similarity_sats = similarity_stats['satellite_stats']['training_satellites']
    avg_similarity_sats = np.mean(similarity_sats)
    logger.info(f"相似度分组平均使用卫星数: {avg_similarity_sats:.2f}")
    
    # 2. 为FedProx和FedAvg创建配置文件
    # target_sats = int(avg_similarity_sats)
    target_sats = 24  # 固定为24个卫星
    logger.info(f"为FedProx和FedAvg设置目标卫星数: {target_sats}")
    
    # 创建配置目录
    os.makedirs("configs/temp", exist_ok=True)
    
    fedprox_config = create_modified_config(
        "configs/propagation_fedprox_config.yaml",
        target_sats,
        f"configs/temp/fedprox_{target_sats}sats.yaml"
    )
    
    fedavg_config = create_modified_config(
        "configs/propagation_fedavg_config.yaml",
        target_sats,
        f"configs/temp/fedavg_{target_sats}sats.yaml"
    )
    
    # 3. 运行有限传播FedProx实验
    logger.info(f"\n=== 运行有限传播FedProx实验 (目标卫星数: {target_sats}) ===")
    fedprox_stats, fedprox_exp = run_experiment(
        fedprox_config, 
        LimitedPropagationFedProx
    )
    
    if not fedprox_stats:
        logger.error("有限传播FedProx实验失败")
        return
    
    # 4. 运行有限传播FedAvg实验
    logger.info(f"\n=== 运行有限传播FedAvg实验 (目标卫星数: {target_sats}) ===")
    fedavg_stats, fedavg_exp = run_experiment(
        fedavg_config, 
        LimitedPropagationFedAvg
    )
    
    if not fedavg_stats:
        logger.error("有限传播FedAvg实验失败")
        return
    
    # 5. 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"comparison_results/fair_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验数据到文件，以便后续可以重新绘制图表
    save_experiment_data(
        output_dir,
        fedprox_stats=fedprox_stats,
        fedavg_stats=fedavg_stats,
        similarity_stats=similarity_stats,
        timestamp=timestamp
    )
    
    # 6. 生成对比报告和图表
    create_comparison_plots(
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        output_dir, 
        fedprox_exp=fedprox_exp, 
        fedavg_exp=fedavg_exp, 
        similarity_exp=similarity_exp
    )
    
    generate_comparison_report(
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        f"{output_dir}/comparison_report.md"
    )
    
    # 7. 打印关键指标
    print_key_metrics(fedprox_stats, fedavg_stats, similarity_stats)
    
    logger.info(f"公平比较实验完成，结果保存在 {output_dir}/")
    logger.info(f"实验原始数据已保存，可使用 'python run_fair_comparison_fedprox_new.py --replot --data-dir {output_dir}' 重新绘制图表")
    
    return output_dir

def print_key_metrics(fedprox_stats, fedavg_stats, similarity_stats):
    """打印关键指标"""
    # 计算平均卫星数
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 获取最高准确率
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    
    # 计算总能耗
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算每卫星准确率
    fedprox_efficiency = fedprox_max_acc / fedprox_sats if fedprox_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_sats if fedavg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_sats if similarity_sats > 0 else 0
    
    # 打印结果
    logger.info("\n=== 关键指标对比 ===")
    logger.info(f"平均卫星数量:")
    logger.info(f"  FedProx: {fedprox_sats:.2f}")
    logger.info(f"  FedAvg: {fedavg_sats:.2f}")
    logger.info(f"  相似度分组: {similarity_sats:.2f}")
    
    logger.info(f"\n最终准确率:")
    logger.info(f"  FedProx: {fedprox_max_acc:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_acc:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_acc:.2f}%")
    
    logger.info(f"\n总能耗:")
    logger.info(f"  FedProx: {fedprox_energy:.2f} Wh")
    logger.info(f"  FedAvg: {fedavg_energy:.2f} Wh")
    logger.info(f"  相似度分组: {similarity_energy:.2f} Wh")
    
    logger.info(f"\n每卫星准确率:")
    logger.info(f"  FedProx: {fedprox_efficiency:.2f}%/satellite")
    logger.info(f"  FedAvg: {fedavg_efficiency:.2f}%/satellite")
    logger.info(f"  相似度分组: {similarity_efficiency:.2f}%/satellite")


if __name__ == "__main__":
    args = parse_args()
    
    if args.replot:
        # 重新绘图模式
        if not args.data_dir:
            logger.error("重新绘图模式需要指定 --data-dir 参数")
            exit(1)
            
        try:
            # 设置输出目录
            output_dir = args.output_dir if args.output_dir else args.data_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载实验数据
            logger.info(f"从 {args.data_dir} 加载实验数据")
            data = load_experiment_data(args.data_dir)
            
            # 提取各个算法的统计数据
            fedprox_stats = data['fedprox']
            fedavg_stats = data['fedavg']
            similarity_stats = data['similarity']
            
            # 重新绘制图表
            logger.info(f"开始重新生成图表")
            create_comparison_plots(
                fedprox_stats, 
                fedavg_stats, 
                similarity_stats, 
                output_dir,
                show_grid=not args.no_grid,
                figure_format=args.format,
                dpi=args.dpi
            )
            
            logger.info(f"图表重绘完成，保存在 {output_dir}/")
            
        except Exception as e:
            logger.error(f"重新绘制图表时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            exit(1)
    else:
        # 正常实验模式
        # 如果指定了目标卫星数，则使用命令行参数
        if args.target_sats > 0:
            logger.info(f"使用命令行指定的目标卫星数: {args.target_sats}")
        
        # 运行公平比较实验
        run_fair_comparison()