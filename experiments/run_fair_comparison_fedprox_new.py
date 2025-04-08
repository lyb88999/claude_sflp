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