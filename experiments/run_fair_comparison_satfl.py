#!/usr/bin/env python3
"""
公平对比实验 - 比较SDA-FL、FedAvg、FedProx与相似度分组算法
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
from experiments.sda_fl_experiment import SDAFLExperiment
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
        logging.FileHandler("comparison_with_sda_fl.log")
    ]
)
logger = logging.getLogger('comparison_with_sda_fl')

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

def create_comparison_plots(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, output_dir, 
                           satfl_exp=None, fedprox_exp=None, fedavg_exp=None, similarity_exp=None,
                           custom_style=None, show_grid=True, figure_format='png', dpi=150):
    """
    创建对比图表，支持自定义图表样式
    """
    # 获取实际参与的卫星数量
    satfl_sats = np.mean(satfl_stats['satellite_stats']['training_satellites'])
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(SDA-FL: {satfl_sats:.1f}, FedProx: {fedprox_sats:.1f}, FedAvg: {fedavg_sats:.1f}, Similarity: {similarity_sats:.1f} satellites)"
    
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
    offsets = {
        'SDA-FL': 0.3,
        'FedProx': 0.1, 
        'FedAvg': -0.1,
        'Similarity': -0.3
    }
    
    # 算法样式定义
    algo_styles = {
        'SDA-FL': {'color': 'purple', 'marker': '*', 'label': 'SDA-FL'},
        'FedProx': {'color': 'g', 'marker': 'o', 'label': 'FedProx'},
        'FedAvg': {'color': 'b', 'marker': 's', 'label': 'FedAvg'},
        'Similarity': {'color': 'r', 'marker': '^', 'label': 'Similarity Grouping'}
    }
    
    # 1. 准确率对比
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['accuracies'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    
    # 1.1 F1 Score 对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'f1_macros' in satfl_stats and satfl_stats['f1_macros']:
        plt.plot(satfl_stats['f1_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in fedprox_stats and fedprox_stats['f1_macros']:
        plt.plot(fedprox_stats['f1_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in fedavg_stats and fedavg_stats['f1_macros']:
        plt.plot(fedavg_stats['f1_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_macros' in similarity_stats and similarity_stats['f1_macros']:
        plt.plot(similarity_stats['f1_macros'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'F1 Score (Macro) Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('F1 Score (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.2 精确率对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'precision_macros' in satfl_stats and satfl_stats['precision_macros']:
        plt.plot(satfl_stats['precision_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in fedprox_stats and fedprox_stats['precision_macros']:
        plt.plot(fedprox_stats['precision_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in fedavg_stats and fedavg_stats['precision_macros']:
        plt.plot(fedavg_stats['precision_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_macros' in similarity_stats and similarity_stats['precision_macros']:
        plt.plot(similarity_stats['precision_macros'], 
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
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.3 召回率对比 (Macro)
    plt.figure(figsize=style['figsize'])
    if 'recall_macros' in satfl_stats and satfl_stats['recall_macros']:
        plt.plot(satfl_stats['recall_macros'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in fedprox_stats and fedprox_stats['recall_macros']:
        plt.plot(fedprox_stats['recall_macros'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in fedavg_stats and fedavg_stats['recall_macros']:
        plt.plot(fedavg_stats['recall_macros'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_macros' in similarity_stats and similarity_stats['recall_macros']:
        plt.plot(similarity_stats['recall_macros'], 
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
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_macro_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 1.4 综合性能指标对比（在一张图中显示多个指标）
    plt.figure(figsize=(14, 8))
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 准确率
    axes[0, 0].plot(satfl_stats['accuracies'], 
                    color=algo_styles['SDA-FL']['color'], 
                    marker=algo_styles['SDA-FL']['marker'], 
                    label=algo_styles['SDA-FL']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(fedprox_stats['accuracies'], 
                    color=algo_styles['FedProx']['color'], 
                    marker=algo_styles['FedProx']['marker'], 
                    label=algo_styles['FedProx']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(fedavg_stats['accuracies'], 
                    color=algo_styles['FedAvg']['color'], 
                    marker=algo_styles['FedAvg']['marker'], 
                    label=algo_styles['FedAvg']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].plot(similarity_stats['accuracies'], 
                    color=algo_styles['Similarity']['color'], 
                    marker=algo_styles['Similarity']['marker'], 
                    label=algo_styles['Similarity']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[0, 0].set_title('Accuracy (%)')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    if show_grid:
        axes[0, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图2: F1 Score (Macro)
    if all('f1_macros' in stats and stats['f1_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 1].plot(satfl_stats['f1_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedprox_stats['f1_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedavg_stats['f1_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(similarity_stats['f1_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 1].set_title('F1 Score (Macro)')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('F1 Score (%)')
    if show_grid:
        axes[0, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图3: 精确率 (Macro)
    if all('precision_macros' in stats and stats['precision_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 0].plot(satfl_stats['precision_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedprox_stats['precision_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedavg_stats['precision_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(similarity_stats['precision_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 0].set_title('Precision (Macro)')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Precision (%)')
    if show_grid:
        axes[1, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图4: 召回率 (Macro)
    if all('recall_macros' in stats and stats['recall_macros'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 1].plot(satfl_stats['recall_macros'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(fedprox_stats['recall_macros'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(fedavg_stats['recall_macros'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 1].plot(similarity_stats['recall_macros'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 1].set_title('Recall (Macro)')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Recall (%)')
    if show_grid:
        axes[1, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 添加总标题和图例
    fig.suptitle(f'Classification Metrics Comparison {title_suffix}', fontsize=16, y=0.98)
    
    # 在图的底部添加通用图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    plt.savefig(f"{output_dir}/comprehensive_metrics_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_stats['losses'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    plt.plot(satfl_stats['energy_stats']['training_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    plt.plot(satfl_stats['energy_stats']['communication_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedprox_stats['energy_stats']['communication_energy'], 
             color=algo_styles['FedProx']['color'], 
             marker=algo_styles['FedProx']['marker'], 
             label=algo_styles['FedProx']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
    plt.plot(fedavg_stats['energy_stats']['communication_energy'], 
             color=algo_styles['FedAvg']['color'], 
             marker=algo_styles['FedAvg']['marker'], 
             label=algo_styles['FedAvg']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    plt.plot(satfl_stats['energy_stats']['total_energy'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    satfl_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                      zip(satfl_stats['accuracies'], satfl_stats['energy_stats']['total_energy'])]
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedprox_stats['accuracies'], fedprox_stats['energy_stats']['total_energy'])]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=style['figsize'])
    plt.plot(satfl_efficiency, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    import matplotlib as mpl
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=style['figsize'])
    
    # 对每个算法应用偏移显示，使用相同的线型
    for algo_name, data, style_name in [
        ('SDA-FL', satfl_stats['satellite_stats']['training_satellites'], 'SDA-FL'),
        ('FedProx', fedprox_stats['satellite_stats']['training_satellites'], 'FedProx'),
        ('FedAvg', fedavg_stats['satellite_stats']['training_satellites'], 'FedAvg'),
        ('Similarity', similarity_stats['satellite_stats']['training_satellites'], 'Similarity')
    ]:
        # 创建带偏移的数据副本
        offset_data = [val + offsets[style_name] for val in data]
        
        plt.plot(offset_data, 
             color=algo_styles[style_name]['color'], 
             marker=algo_styles[style_name]['marker'], 
             label=algo_styles[style_name]['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'],
             markevery=2)  # 减少标记数量以避免拥挤
    
    # 添加说明文本，解释偏移显示
    plt.text(0.02, 0.02, "注：应用了垂直偏移以区分重叠的线条。\n实际数据值见表名。", 
             transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加参考线，显示实际数据值
    plt.axhline(y=24, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.axhline(y=19.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # 其他代码保持不变
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
    satfl_comm = calculate_communication_overhead(satfl_stats)
    fedprox_comm = calculate_communication_overhead(fedprox_stats)
    fedavg_comm = calculate_communication_overhead(fedavg_stats)
    similarity_comm = calculate_communication_overhead(similarity_stats)
    
    plt.plot(satfl_comm, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    satfl_cumulative_energy = np.cumsum(satfl_stats['energy_stats']['total_energy'])
    fedprox_cumulative_energy = np.cumsum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_cumulative_energy = np.cumsum(fedavg_stats['energy_stats']['total_energy'])
    similarity_cumulative_energy = np.cumsum(similarity_stats['energy_stats']['total_energy'])
    
    satfl_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                      zip(satfl_stats['accuracies'], satfl_cumulative_energy)]
    fedprox_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedprox_stats['accuracies'], fedprox_cumulative_energy)]
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_cumulative_energy)]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_cumulative_energy)]
    
    plt.plot(satfl_efficiency, 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
             linewidth=style['linewidth'],
             markersize=style['marker_size'])
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
    
    # 10. 加权指标对比
    # 10.1 F1 Score (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'f1_weighteds' in satfl_stats and satfl_stats['f1_weighteds']:
        plt.plot(satfl_stats['f1_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in fedprox_stats and fedprox_stats['f1_weighteds']:
        plt.plot(fedprox_stats['f1_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in fedavg_stats and fedavg_stats['f1_weighteds']:
        plt.plot(fedavg_stats['f1_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'f1_weighteds' in similarity_stats and similarity_stats['f1_weighteds']:
        plt.plot(similarity_stats['f1_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'F1 Score (Weighted) Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('F1 Score (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.2 精确率 (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'precision_weighteds' in satfl_stats and satfl_stats['precision_weighteds']:
        plt.plot(satfl_stats['precision_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in fedprox_stats and fedprox_stats['precision_weighteds']:
        plt.plot(fedprox_stats['precision_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in fedavg_stats and fedavg_stats['precision_weighteds']:
        plt.plot(fedavg_stats['precision_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'precision_weighteds' in similarity_stats and similarity_stats['precision_weighteds']:
        plt.plot(similarity_stats['precision_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Precision (Weighted) Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Precision (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.3 召回率 (Weighted) 对比
    plt.figure(figsize=style['figsize'])
    if 'recall_weighteds' in satfl_stats and satfl_stats['recall_weighteds']:
        plt.plot(satfl_stats['recall_weighteds'], 
                 color=algo_styles['SDA-FL']['color'], 
                 marker=algo_styles['SDA-FL']['marker'], 
                 label=algo_styles['SDA-FL']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in fedprox_stats and fedprox_stats['recall_weighteds']:
        plt.plot(fedprox_stats['recall_weighteds'], 
                 color=algo_styles['FedProx']['color'], 
                 marker=algo_styles['FedProx']['marker'], 
                 label=algo_styles['FedProx']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in fedavg_stats and fedavg_stats['recall_weighteds']:
        plt.plot(fedavg_stats['recall_weighteds'], 
                 color=algo_styles['FedAvg']['color'], 
                 marker=algo_styles['FedAvg']['marker'], 
                 label=algo_styles['FedAvg']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    if 'recall_weighteds' in similarity_stats and similarity_stats['recall_weighteds']:
        plt.plot(similarity_stats['recall_weighteds'], 
                 color=algo_styles['Similarity']['color'], 
                 marker=algo_styles['Similarity']['marker'], 
                 label=algo_styles['Similarity']['label'],
                 linewidth=style['linewidth'],
                 markersize=style['marker_size'])
    plt.title(f'Recall (Weighted) Comparison {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Recall (%)')
    plt.legend()
    if show_grid:
        plt.grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_weighted_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    # 10.4 加权指标综合对比图
    # 创建2x2的子图显示加权指标
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: F1 Score (Weighted)
    if all('f1_weighteds' in stats and stats['f1_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 0].plot(satfl_stats['f1_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(fedprox_stats['f1_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(fedavg_stats['f1_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 0].plot(similarity_stats['f1_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 0].set_title('F1 Score (Weighted)')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('F1 Score (%)')
    if show_grid:
        axes[0, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图2: 精确率 (Weighted)
    if all('precision_weighteds' in stats and stats['precision_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[0, 1].plot(satfl_stats['precision_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedprox_stats['precision_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(fedavg_stats['precision_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[0, 1].plot(similarity_stats['precision_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[0, 1].set_title('Precision (Weighted)')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Precision (%)')
    if show_grid:
        axes[0, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图3: 召回率 (Weighted)
    if all('recall_weighteds' in stats and stats['recall_weighteds'] for stats in [satfl_stats, fedprox_stats, fedavg_stats, similarity_stats]):
        axes[1, 0].plot(satfl_stats['recall_weighteds'], 
                        color=algo_styles['SDA-FL']['color'], 
                        marker=algo_styles['SDA-FL']['marker'], 
                        label=algo_styles['SDA-FL']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedprox_stats['recall_weighteds'], 
                        color=algo_styles['FedProx']['color'], 
                        marker=algo_styles['FedProx']['marker'], 
                        label=algo_styles['FedProx']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(fedavg_stats['recall_weighteds'], 
                        color=algo_styles['FedAvg']['color'], 
                        marker=algo_styles['FedAvg']['marker'], 
                        label=algo_styles['FedAvg']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
        axes[1, 0].plot(similarity_stats['recall_weighteds'], 
                        color=algo_styles['Similarity']['color'], 
                        marker=algo_styles['Similarity']['marker'], 
                        label=algo_styles['Similarity']['label'],
                        linewidth=style['linewidth'],
                        markersize=style['marker_size'])
    axes[1, 0].set_title('Recall (Weighted)')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Recall (%)')
    if show_grid:
        axes[1, 0].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 子图4: 准确率（作为参考）
    axes[1, 1].plot(satfl_stats['accuracies'], 
                    color=algo_styles['SDA-FL']['color'], 
                    marker=algo_styles['SDA-FL']['marker'], 
                    label=algo_styles['SDA-FL']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(fedprox_stats['accuracies'], 
                    color=algo_styles['FedProx']['color'], 
                    marker=algo_styles['FedProx']['marker'], 
                    label=algo_styles['FedProx']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(fedavg_stats['accuracies'], 
                    color=algo_styles['FedAvg']['color'], 
                    marker=algo_styles['FedAvg']['marker'], 
                    label=algo_styles['FedAvg']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].plot(similarity_stats['accuracies'], 
                    color=algo_styles['Similarity']['color'], 
                    marker=algo_styles['Similarity']['marker'], 
                    label=algo_styles['Similarity']['label'],
                    linewidth=style['linewidth'],
                    markersize=style['marker_size'])
    axes[1, 1].set_title('Accuracy (Reference)')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accuracy (%)')
    if show_grid:
        axes[1, 1].grid(alpha=style['grid_alpha'], linestyle=style['grid_linestyle'])
    
    # 添加总标题和图例
    fig.suptitle(f'Weighted Classification Metrics Comparison {title_suffix}', fontsize=16, y=0.98)
    
    # 在图的底部添加通用图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    plt.savefig(f"{output_dir}/weighted_metrics_comparison.{style['save_format']}", dpi=style['dpi'])
    plt.close()
    
    logger.info(f"图表生成完成，保存在 {output_dir}/ 目录")

def save_experiment_data(output_dir, satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, timestamp):
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
    satfl_data = prepare_for_serialization(satfl_stats)
    fedprox_data = prepare_for_serialization(fedprox_stats)
    fedavg_data = prepare_for_serialization(fedavg_stats)
    similarity_data = prepare_for_serialization(similarity_stats)
    
    # 保存为pickle格式(包含完整数据)
    with open(os.path.join(data_dir, 'experiment_data.pkl'), 'wb') as f:
        pickle.dump({
            'satfl': satfl_data,
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
                'satfl': satfl_data,
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
        with open(os.path.join(data_dir, 'satfl_config.yaml'), 'w') as f:
            with open("configs/sda_fl_config.yaml", 'r') as src:
                f.write(src.read())
                
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
        f.write(f"  SDA-FL: {np.mean(satfl_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  FedProx: {np.mean(fedprox_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  FedAvg: {np.mean(fedavg_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  相似度分组: {np.mean(similarity_stats['satellite_stats']['training_satellites']):.2f}\n\n")
        
        f.write("最终准确率:\n")
        f.write(f"  SDA-FL: {max(satfl_stats['accuracies']):.2f}%\n")
        f.write(f"  FedProx: {max(fedprox_stats['accuracies']):.2f}%\n")
        f.write(f"  FedAvg: {max(fedavg_stats['accuracies']):.2f}%\n")
        f.write(f"  相似度分组: {max(similarity_stats['accuracies']):.2f}%\n")
    
    logger.info(f"实验数据已保存到 {data_dir}/")

def load_experiment_data(data_dir):
    """加载保存的实验数据"""
    # 优先尝试加载pickle格式
    # pickle_path = os.path.join(data_dir, 'raw_data', 'experiment_data.pkl')
    # if os.path.exists(pickle_path):
    #     try:
    #         with open(pickle_path, 'rb') as f:
    #             return pickle.load(f)
    #     except Exception as e:
    #         logger.error(f"无法加载pickle数据: {str(e)}")
    
    # 尝试加载JSON格式
    logger.info(f"尝试加载JSON格式数据: {data_dir}")
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
    parser.add_argument('--satfl-noise-dim', type=int, default=100,
                      help='SDA-FL的噪声维度')
    parser.add_argument('--satfl-samples', type=int, default=1000,
                      help='SDA-FL生成的合成样本数量')
    
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

def generate_comparison_report(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats, output_path):
    """生成对比报告"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 计算各种指标
    satfl_max_acc = max(satfl_stats['accuracies'])
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    
    # 计算F1分数（如果有的话）
    satfl_max_f1 = max(satfl_stats.get('f1_macros', [0])) if satfl_stats.get('f1_macros') else 0
    fedprox_max_f1 = max(fedprox_stats.get('f1_macros', [0])) if fedprox_stats.get('f1_macros') else 0
    fedavg_max_f1 = max(fedavg_stats.get('f1_macros', [0])) if fedavg_stats.get('f1_macros') else 0
    similarity_max_f1 = max(similarity_stats.get('f1_macros', [0])) if similarity_stats.get('f1_macros') else 0
    
    # 计算精确率（如果有的话）
    satfl_max_precision = max(satfl_stats.get('precision_macros', [0])) if satfl_stats.get('precision_macros') else 0
    fedprox_max_precision = max(fedprox_stats.get('precision_macros', [0])) if fedprox_stats.get('precision_macros') else 0
    fedavg_max_precision = max(fedavg_stats.get('precision_macros', [0])) if fedavg_stats.get('precision_macros') else 0
    similarity_max_precision = max(similarity_stats.get('precision_macros', [0])) if similarity_stats.get('precision_macros') else 0
    
    # 计算召回率（如果有的话）
    satfl_max_recall = max(satfl_stats.get('recall_macros', [0])) if satfl_stats.get('recall_macros') else 0
    fedprox_max_recall = max(fedprox_stats.get('recall_macros', [0])) if fedprox_stats.get('recall_macros') else 0
    fedavg_max_recall = max(fedavg_stats.get('recall_macros', [0])) if fedavg_stats.get('recall_macros') else 0
    similarity_max_recall = max(similarity_stats.get('recall_macros', [0])) if similarity_stats.get('recall_macros') else 0
    
    satfl_avg_sats = np.mean(satfl_stats['satellite_stats']['training_satellites'])
    fedprox_avg_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    satfl_energy = sum(satfl_stats['energy_stats']['total_energy'])
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算各种效率指标
    satfl_efficiency = satfl_max_acc / satfl_avg_sats if satfl_avg_sats > 0 else 0
    fedprox_efficiency = fedprox_max_acc / fedprox_avg_sats if fedprox_avg_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_avg_sats if fedavg_avg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_avg_sats if similarity_avg_sats > 0 else 0
    
    satfl_energy_efficiency = satfl_max_acc / satfl_energy if satfl_energy > 0 else 0
    fedprox_energy_efficiency = fedprox_max_acc / fedprox_energy if fedprox_energy > 0 else 0
    fedavg_energy_efficiency = fedavg_max_acc / fedavg_energy if fedavg_energy > 0 else 0
    similarity_energy_efficiency = similarity_max_acc / similarity_energy if similarity_energy > 0 else 0
    
    # 计算F1效率指标
    satfl_f1_efficiency = satfl_max_f1 / satfl_avg_sats if satfl_avg_sats > 0 else 0
    fedprox_f1_efficiency = fedprox_max_f1 / fedprox_avg_sats if fedprox_avg_sats > 0 else 0
    fedavg_f1_efficiency = fedavg_max_f1 / fedavg_avg_sats if fedavg_avg_sats > 0 else 0
    similarity_f1_efficiency = similarity_max_f1 / similarity_avg_sats if similarity_avg_sats > 0 else 0
    
    # 计算收敛速度 (达到90%最终准确率的轮次)
    def calculate_convergence(accuracies, target_percentage=0.9):
        target = max(accuracies) * target_percentage
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i + 1
        return len(accuracies)  # 如果没有达到，返回总轮次
    
    satfl_convergence = calculate_convergence(satfl_stats['accuracies'])
    fedprox_convergence = calculate_convergence(fedprox_stats['accuracies'])
    fedavg_convergence = calculate_convergence(fedavg_stats['accuracies'])
    similarity_convergence = calculate_convergence(similarity_stats['accuracies'])
    
    # 计算F1分数收敛速度
    def calculate_f1_convergence(f1_scores, target_percentage=0.9):
        if not f1_scores:
            return float('inf')
        target = max(f1_scores) * target_percentage
        for i, f1 in enumerate(f1_scores):
            if f1 >= target:
                return i + 1
        return len(f1_scores)
    
    satfl_f1_convergence = calculate_f1_convergence(satfl_stats.get('f1_macros', []))
    fedprox_f1_convergence = calculate_f1_convergence(fedprox_stats.get('f1_macros', []))
    fedavg_f1_convergence = calculate_f1_convergence(fedavg_stats.get('f1_macros', []))
    similarity_f1_convergence = calculate_f1_convergence(similarity_stats.get('f1_macros', []))
    
    # 生成报告
    with open(output_path, 'w') as f:
        f.write("# 公平对比报告: SDA-FL vs FedProx vs FedAvg vs 相似度分组\n\n")
        
        f.write("## 实验设置\n")
        f.write(f"- SDA-FL 参数 - 合成样本数: {satfl_stats.get('num_synthetic_samples', 1000)}\n")
        f.write(f"- FedProx 参数 μ: {fedprox_stats.get('mu', 0.01)}\n")
        f.write(f"- 总轮次: {len(satfl_stats['accuracies'])}\n\n")
        
        f.write("## 参与卫星数量\n")
        f.write(f"- SDA-FL 平均训练卫星数: {satfl_avg_sats:.2f}\n")
        f.write(f"- FedProx 平均训练卫星数: {fedprox_avg_sats:.2f}\n")
        f.write(f"- FedAvg 平均训练卫星数: {fedavg_avg_sats:.2f}\n")
        f.write(f"- 相似度分组平均训练卫星数: {similarity_avg_sats:.2f}\n\n")
        
        f.write("## 分类性能指标\n")
        f.write("### 准确率性能\n")
        f.write(f"- SDA-FL 最高准确率: {satfl_max_acc:.2f}%\n")
        f.write(f"- FedProx 最高准确率: {fedprox_max_acc:.2f}%\n")
        f.write(f"- FedAvg 最高准确率: {fedavg_max_acc:.2f}%\n")
        f.write(f"- 相似度分组最高准确率: {similarity_max_acc:.2f}%\n\n")
        
        f.write("### F1分数性能 (Macro)\n")
        f.write(f"- SDA-FL 最高F1分数: {satfl_max_f1:.2f}%\n")
        f.write(f"- FedProx 最高F1分数: {fedprox_max_f1:.2f}%\n")
        f.write(f"- FedAvg 最高F1分数: {fedavg_max_f1:.2f}%\n")
        f.write(f"- 相似度分组最高F1分数: {similarity_max_f1:.2f}%\n\n")
        
        f.write("### 精确率性能 (Macro)\n")
        f.write(f"- SDA-FL 最高精确率: {satfl_max_precision:.2f}%\n")
        f.write(f"- FedProx 最高精确率: {fedprox_max_precision:.2f}%\n")
        f.write(f"- FedAvg 最高精确率: {fedavg_max_precision:.2f}%\n")
        f.write(f"- 相似度分组最高精确率: {similarity_max_precision:.2f}%\n\n")
        
        f.write("### 召回率性能 (Macro)\n")
        f.write(f"- SDA-FL 最高召回率: {satfl_max_recall:.2f}%\n")
        f.write(f"- FedProx 最高召回率: {fedprox_max_recall:.2f}%\n")
        f.write(f"- FedAvg 最高召回率: {fedavg_max_recall:.2f}%\n")
        f.write(f"- 相似度分组最高召回率: {similarity_max_recall:.2f}%\n\n")
        
        f.write("## 性能对比分析\n")
        f.write(f"- SDA-FL vs FedProx (准确率): {satfl_max_acc - fedprox_max_acc:+.2f}%\n")
        f.write(f"- SDA-FL vs FedAvg (准确率): {satfl_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- SDA-FL vs 相似度分组 (准确率): {satfl_max_acc - similarity_max_acc:+.2f}%\n")
        f.write(f"- FedProx vs FedAvg (准确率): {fedprox_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- FedProx vs 相似度分组 (准确率): {fedprox_max_acc - similarity_max_acc:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedAvg (准确率): {similarity_max_acc - fedavg_max_acc:+.2f}%\n\n")
        
        f.write(f"- SDA-FL vs FedProx (F1分数): {satfl_max_f1 - fedprox_max_f1:+.2f}%\n")
        f.write(f"- SDA-FL vs FedAvg (F1分数): {satfl_max_f1 - fedavg_max_f1:+.2f}%\n")
        f.write(f"- SDA-FL vs 相似度分组 (F1分数): {satfl_max_f1 - similarity_max_f1:+.2f}%\n")
        f.write(f"- FedProx vs FedAvg (F1分数): {fedprox_max_f1 - fedavg_max_f1:+.2f}%\n")
        f.write(f"- FedProx vs 相似度分组 (F1分数): {fedprox_max_f1 - similarity_max_f1:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedAvg (F1分数): {similarity_max_f1 - fedavg_max_f1:+.2f}%\n\n")
        
        f.write("## 能耗\n")
        f.write(f"- SDA-FL 总能耗: {satfl_energy:.2f} Wh\n")
        f.write(f"- FedProx 总能耗: {fedprox_energy:.2f} Wh\n")
        f.write(f"- FedAvg 总能耗: {fedavg_energy:.2f} Wh\n")
        f.write(f"- 相似度分组总能耗: {similarity_energy:.2f} Wh\n\n")
        
        f.write("## 效率指标\n")
        f.write("### 卫星效率（每卫星准确率）\n")
        f.write(f"- SDA-FL 每卫星准确率: {satfl_efficiency:.2f}%\n")
        f.write(f"- FedProx 每卫星准确率: {fedprox_efficiency:.2f}%\n")
        f.write(f"- FedAvg 每卫星准确率: {fedavg_efficiency:.2f}%\n")
        f.write(f"- 相似度分组每卫星准确率: {similarity_efficiency:.2f}%\n\n")
        
        f.write("### 卫星F1效率（每卫星F1分数）\n")
        f.write(f"- SDA-FL 每卫星F1分数: {satfl_f1_efficiency:.2f}%\n")
        f.write(f"- FedProx 每卫星F1分数: {fedprox_f1_efficiency:.2f}%\n")
        f.write(f"- FedAvg 每卫星F1分数: {fedavg_f1_efficiency:.2f}%\n")
        f.write(f"- 相似度分组每卫星F1分数: {similarity_f1_efficiency:.2f}%\n\n")
        
        f.write("### 能源效率\n")
        f.write(f"- SDA-FL 能源效率: {satfl_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- FedProx 能源效率: {fedprox_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- FedAvg 能源效率: {fedavg_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- 相似度分组能源效率: {similarity_energy_efficiency:.4f}%/Wh\n\n")
        
        f.write("## 收敛速度\n")
        f.write("### 准确率收敛速度\n")
        f.write(f"- SDA-FL 达到90%最高准确率轮次: {satfl_convergence}\n")
        f.write(f"- FedProx 达到90%最高准确率轮次: {fedprox_convergence}\n")
        f.write(f"- FedAvg 达到90%最高准确率轮次: {fedavg_convergence}\n")
        f.write(f"- 相似度分组达到90%最高准确率轮次: {similarity_convergence}\n\n")
        
        f.write("### F1分数收敛速度\n")
        f.write(f"- SDA-FL 达到90%最高F1分数轮次: {satfl_f1_convergence}\n")
        f.write(f"- FedProx 达到90%最高F1分数轮次: {fedprox_f1_convergence}\n")
        f.write(f"- FedAvg 达到90%最高F1分数轮次: {fedavg_f1_convergence}\n")
        f.write(f"- 相似度分组达到90%最高F1分数轮次: {similarity_f1_convergence}\n\n")
        
        f.write("## 总结\n")
        # 添加各个方法的优缺点和总结
        f.write("### SDA-FL\n")
        f.write("- **优势**: 使用合成数据增强训练，在数据稀缺情况下能提高准确率。\n")
        f.write("- **劣势**: 需要训练GAN模型，增加了计算复杂度和能耗。\n\n")

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
        methods = {
            'SDA-FL': satfl_max_acc,
            'FedProx': fedprox_max_acc,
            'FedAvg': fedavg_max_acc,
            '相似度分组': similarity_max_acc
        }
        best_method = max(methods.items(), key=lambda x: x[1])[0]
        
        f1_methods = {
            'SDA-FL': satfl_max_f1,
            'FedProx': fedprox_max_f1,
            'FedAvg': fedavg_max_f1,
            '相似度分组': similarity_max_f1
        }
        best_f1_method = max(f1_methods.items(), key=lambda x: x[1])[0]
            
        efficiency_methods = {
            'SDA-FL': satfl_efficiency,
            'FedProx': fedprox_efficiency,
            'FedAvg': fedavg_efficiency,
            '相似度分组': similarity_efficiency
        }
        best_efficiency = max(efficiency_methods.items(), key=lambda x: x[1])[0]
            
        f.write("### 结论\n")
        f.write(f"**准确率**: {best_method}在准确率上表现最好 ({methods[best_method]:.2f}%)。\n")
        f.write(f"**F1分数**: {best_f1_method}在F1分数上表现最好 ({f1_methods[best_f1_method]:.2f}%)。\n")
        f.write(f"**资源效率**: {best_efficiency}在资源效率上表现最好 ({efficiency_methods[best_efficiency]:.2f}%/satellite)。\n\n")
        
        # 综合评估
        if best_method == best_f1_method and best_method == best_efficiency:
            f.write(f"{best_method}算法综合表现最佳，在所有指标上都领先。\n")
        elif best_method == best_f1_method:
            f.write(f"{best_method}在分类性能（准确率和F1分数）方面表现最佳，而{best_efficiency}在资源效率方面更优。\n")
        else:
            f.write(f"需要根据具体场景权衡选择合适的联邦学习算法：\n")
            f.write(f"- 注重准确率：选择{best_method}\n")
            f.write(f"- 注重F1分数：选择{best_f1_method}\n")
            f.write(f"- 注重资源效率：选择{best_efficiency}\n")

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

def create_satfl_config(args, target_satellite_count, output_path="configs/sda_fl_config.yaml"):
    """创建SDA-FL配置文件"""
    try:
        # 基于baseline_config.yaml创建SDA-FL配置
        with open("configs/baseline_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # 添加SDA-FL特有配置
        config['sda_fl'] = {
            'noise_dim': args.satfl_noise_dim,
            'num_synthetic_samples': args.satfl_samples,
            'dp_epsilon': 1.0,
            'dp_delta': 1e-5,
            'pseudo_threshold': 0.8,
            'initial_rounds': 3,
            'gan_epochs': 50,
            'gan_samples_per_client': 100,
            'regenerate_interval': 5
        }
        
        # 保存配置
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"已创建SDA-FL配置文件：{output_path}")
        logger.info(f"- 噪声维度：{args.satfl_noise_dim}")
        logger.info(f"- 合成样本数：{args.satfl_samples}")
        
        return output_path
    except Exception as e:
        logger.error(f"创建SDA-FL配置文件时出错：{str(e)}")
        return None

def run_fair_comparison():
    """运行公平比较实验"""
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
            satfl_stats = data['satfl']
            fedprox_stats = data['fedprox']
            fedavg_stats = data['fedavg']
            similarity_stats = data['similarity']
            
            # 重新绘制图表
            logger.info(f"开始重新生成图表")
            create_comparison_plots(
                satfl_stats,
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
        return
    
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
    
    # 2. 为FedProx、FedAvg和SDA-FL创建配置文件
    target_sats = 24 if args.target_sats == 0 else args.target_sats
    logger.info(f"为SDA-FL、FedProx和FedAvg设置目标卫星数: {target_sats}")
    
    # 创建配置目录
    os.makedirs("configs/temp", exist_ok=True)
    
    # 为SDA-FL创建配置
    satfl_config = create_satfl_config(args, target_sats)
    
    # 为FedProx和FedAvg创建配置
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
    
    # 3. 运行SDA-FL实验
    logger.info(f"\n=== 运行SDA-FL实验 (目标卫星数: {target_sats}) ===")
    satfl_stats, satfl_exp = run_experiment(
        satfl_config, 
        SDAFLExperiment
    )
    
    if not satfl_stats:
        logger.error("SDA-FL实验失败")
        return
    
    # 4. 运行有限传播FedProx实验
    logger.info(f"\n=== 运行有限传播FedProx实验 (目标卫星数: {target_sats}) ===")
    fedprox_stats, fedprox_exp = run_experiment(
        fedprox_config, 
        LimitedPropagationFedProx
    )
    
    if not fedprox_stats:
        logger.error("有限传播FedProx实验失败")
        return
    
    # 5. 运行有限传播FedAvg实验
    logger.info(f"\n=== 运行有限传播FedAvg实验 (目标卫星数: {target_sats}) ===")
    fedavg_stats, fedavg_exp = run_experiment(
        fedavg_config, 
        LimitedPropagationFedAvg
    )
    
    if not fedavg_stats:
        logger.error("有限传播FedAvg实验失败")
        return
    
    # 6. 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"comparison_results/with_satfl_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验数据到文件，以便后续可以重新绘制图表
    save_experiment_data(
        output_dir,
        satfl_stats=satfl_stats,
        fedprox_stats=fedprox_stats,
        fedavg_stats=fedavg_stats,
        similarity_stats=similarity_stats,
        timestamp=timestamp
    )
    
    # 7. 生成对比报告和图表
    create_comparison_plots(
        satfl_stats,
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        output_dir, 
        satfl_exp=satfl_exp,
        fedprox_exp=fedprox_exp, 
        fedavg_exp=fedavg_exp, 
        similarity_exp=similarity_exp
    )
    
    generate_comparison_report(
        satfl_stats, 
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        f"{output_dir}/comparison_report.md"
    )
    
    # 8. 打印关键指标
    print_key_metrics(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats)
    
    logger.info(f"公平比较实验完成，结果保存在 {output_dir}/")
    logger.info(f"实验原始数据已保存，可使用 'python run_fair_comparison_satfl.py --replot --data-dir {output_dir}' 重新绘制图表")
    
    return output_dir

def print_key_metrics(satfl_stats, fedprox_stats, fedavg_stats, similarity_stats):
    """打印关键指标"""
    # 计算平均卫星数
    satfl_sats = np.mean(satfl_stats['satellite_stats']['training_satellites'])
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 获取最高准确率
    satfl_max_acc = max(satfl_stats['accuracies'])
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    
    # 获取最高F1分数
    satfl_max_f1 = max(satfl_stats.get('f1_macros', [0])) if satfl_stats.get('f1_macros') else 0
    fedprox_max_f1 = max(fedprox_stats.get('f1_macros', [0])) if fedprox_stats.get('f1_macros') else 0
    fedavg_max_f1 = max(fedavg_stats.get('f1_macros', [0])) if fedavg_stats.get('f1_macros') else 0
    similarity_max_f1 = max(similarity_stats.get('f1_macros', [0])) if similarity_stats.get('f1_macros') else 0
    
    # 获取最高精确率
    satfl_max_precision = max(satfl_stats.get('precision_macros', [0])) if satfl_stats.get('precision_macros') else 0
    fedprox_max_precision = max(fedprox_stats.get('precision_macros', [0])) if fedprox_stats.get('precision_macros') else 0
    fedavg_max_precision = max(fedavg_stats.get('precision_macros', [0])) if fedavg_stats.get('precision_macros') else 0
    similarity_max_precision = max(similarity_stats.get('precision_macros', [0])) if similarity_stats.get('precision_macros') else 0
    
    # 获取最高召回率
    satfl_max_recall = max(satfl_stats.get('recall_macros', [0])) if satfl_stats.get('recall_macros') else 0
    fedprox_max_recall = max(fedprox_stats.get('recall_macros', [0])) if fedprox_stats.get('recall_macros') else 0
    fedavg_max_recall = max(fedavg_stats.get('recall_macros', [0])) if fedavg_stats.get('recall_macros') else 0
    similarity_max_recall = max(similarity_stats.get('recall_macros', [0])) if similarity_stats.get('recall_macros') else 0
    
    # 计算总能耗
    satfl_energy = sum(satfl_stats['energy_stats']['total_energy'])
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    
    # 计算每卫星准确率
    satfl_efficiency = satfl_max_acc / satfl_sats if satfl_sats > 0 else 0
    fedprox_efficiency = fedprox_max_acc / fedprox_sats if fedprox_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_sats if fedavg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_sats if similarity_sats > 0 else 0
    
    # 计算每卫星F1分数
    satfl_f1_efficiency = satfl_max_f1 / satfl_sats if satfl_sats > 0 else 0
    fedprox_f1_efficiency = fedprox_max_f1 / fedprox_sats if fedprox_sats > 0 else 0
    fedavg_f1_efficiency = fedavg_max_f1 / fedavg_sats if fedavg_sats > 0 else 0
    similarity_f1_efficiency = similarity_max_f1 / similarity_sats if similarity_sats > 0 else 0
    
    # 打印结果
    logger.info("\n=== 关键指标对比 ===")
    logger.info(f"平均卫星数量:")
    logger.info(f"  SDA-FL: {satfl_sats:.2f}")
    logger.info(f"  FedProx: {fedprox_sats:.2f}")
    logger.info(f"  FedAvg: {fedavg_sats:.2f}")
    logger.info(f"  相似度分组: {similarity_sats:.2f}")
    
    logger.info(f"\n最终准确率:")
    logger.info(f"  SDA-FL: {satfl_max_acc:.2f}%")
    logger.info(f"  FedProx: {fedprox_max_acc:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_acc:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_acc:.2f}%")
    
    logger.info(f"\n最高F1分数 (Macro):")
    logger.info(f"  SDA-FL: {satfl_max_f1:.2f}%")
    logger.info(f"  FedProx: {fedprox_max_f1:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_f1:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_f1:.2f}%")
    
    logger.info(f"\n最高精确率 (Macro):")
    logger.info(f"  SDA-FL: {satfl_max_precision:.2f}%")
    logger.info(f"  FedProx: {fedprox_max_precision:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_precision:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_precision:.2f}%")
    
    logger.info(f"\n最高召回率 (Macro):")
    logger.info(f"  SDA-FL: {satfl_max_recall:.2f}%")
    logger.info(f"  FedProx: {fedprox_max_recall:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_recall:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_recall:.2f}%")
    
    logger.info(f"\n总能耗:")
    logger.info(f"  SDA-FL: {satfl_energy:.2f} Wh")
    logger.info(f"  FedProx: {fedprox_energy:.2f} Wh")
    logger.info(f"  FedAvg: {fedavg_energy:.2f} Wh")
    logger.info(f"  相似度分组: {similarity_energy:.2f} Wh")
    
    logger.info(f"\n每卫星准确率:")
    logger.info(f"  SDA-FL: {satfl_efficiency:.2f}%/satellite")
    logger.info(f"  FedProx: {fedprox_efficiency:.2f}%/satellite")
    logger.info(f"  FedAvg: {fedavg_efficiency:.2f}%/satellite")
    logger.info(f"  相似度分组: {similarity_efficiency:.2f}%/satellite")
    
    logger.info(f"\n每卫星F1分数:")
    logger.info(f"  SDA-FL: {satfl_f1_efficiency:.2f}%/satellite")
    logger.info(f"  FedProx: {fedprox_f1_efficiency:.2f}%/satellite")
    logger.info(f"  FedAvg: {fedavg_f1_efficiency:.2f}%/satellite")
    logger.info(f"  相似度分组: {similarity_f1_efficiency:.2f}%/satellite")
    
    # 打印性能排名
    logger.info(f"\n=== 性能排名 ===")
    
    # 准确率排名
    acc_ranking = sorted([
        ('SDA-FL', satfl_max_acc),
        ('FedProx', fedprox_max_acc),
        ('FedAvg', fedavg_max_acc),
        ('相似度分组', similarity_max_acc)
    ], key=lambda x: x[1], reverse=True)
    
    logger.info("准确率排名:")
    for i, (method, acc) in enumerate(acc_ranking, 1):
        logger.info(f"  {i}. {method}: {acc:.2f}%")
    
    # F1分数排名
    f1_ranking = sorted([
        ('SDA-FL', satfl_max_f1),
        ('FedProx', fedprox_max_f1),
        ('FedAvg', fedavg_max_f1),
        ('相似度分组', similarity_max_f1)
    ], key=lambda x: x[1], reverse=True)
    
    logger.info("\nF1分数排名:")
    for i, (method, f1) in enumerate(f1_ranking, 1):
        logger.info(f"  {i}. {method}: {f1:.2f}%")
    
    # 资源效率排名（每卫星准确率）
    eff_ranking = sorted([
        ('SDA-FL', satfl_efficiency),
        ('FedProx', fedprox_efficiency),
        ('FedAvg', fedavg_efficiency),
        ('相似度分组', similarity_efficiency)
    ], key=lambda x: x[1], reverse=True)
    
    logger.info("\n资源效率排名 (每卫星准确率):")
    for i, (method, eff) in enumerate(eff_ranking, 1):
        logger.info(f"  {i}. {method}: {eff:.2f}%/satellite")
    
    # 综合评分（准确率 + F1分数）
    combined_scores = [
        ('SDA-FL', satfl_max_acc + satfl_max_f1),
        ('FedProx', fedprox_max_acc + fedprox_max_f1),
        ('FedAvg', fedavg_max_acc + fedavg_max_f1),
        ('相似度分组', similarity_max_acc + similarity_max_f1)
    ]
    combined_ranking = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    
    logger.info("\n综合性能排名 (准确率 + F1分数):")
    for i, (method, score) in enumerate(combined_ranking, 1):
        logger.info(f"  {i}. {method}: {score:.2f}")
    
    # 显示最佳方法
    best_acc_method = acc_ranking[0][0]
    best_f1_method = f1_ranking[0][0]
    best_eff_method = eff_ranking[0][0]
    best_combined_method = combined_ranking[0][0]
    
    logger.info(f"\n=== 最佳表现 ===")
    logger.info(f"最佳准确率: {best_acc_method}")
    logger.info(f"最佳F1分数: {best_f1_method}")
    logger.info(f"最佳资源效率: {best_eff_method}")
    logger.info(f"最佳综合性能: {best_combined_method}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.replot:
        # 重新绘图模式 - 已在函数中实现
        run_fair_comparison()
    else:
        # 正常实验模式
        run_fair_comparison()
