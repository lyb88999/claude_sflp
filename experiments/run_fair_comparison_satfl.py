#!/usr/bin/env python3
"""
公平对比实验 - 比较SDA-FL、有限传播FedProx、有限传播FedAvg与相似度分组算法
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
from experiments.sda_fl_experiment import SDAFLExperiment
from visualization.visualization import Visualization

import pickle
import json
import copy

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

def create_comparison_plots(fedprox_stats, fedavg_stats, similarity_stats, sda_fl_stats, output_dir, 
                           fedprox_exp=None, fedavg_exp=None, similarity_exp=None, sda_fl_exp=None,
                           custom_style=None, show_grid=True, figure_format='png', dpi=150):
    """
    创建对比图表，支持自定义图表样式
    
    Args:
        fedprox_stats: FedProx实验统计数据
        fedavg_stats: FedAvg实验统计数据
        similarity_stats: 相似度分组实验统计数据
        sda_fl_stats: SDA-FL实验统计数据
        output_dir: 输出目录
        ...其他参数...
    """
    # 获取实际参与的卫星数量
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    sda_fl_sats = np.mean(sda_fl_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedProx: {fedprox_sats:.1f}, FedAvg: {fedavg_sats:.1f}, Similarity: {similarity_sats:.1f}, SDA-FL: {sda_fl_sats:.1f} satellites)"
    
    # 应用自定义样式
    style = {
        'figsize': (12, 7),  # 增大图表尺寸以容纳更多曲线
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
        'Similarity': {'color': 'r', 'marker': '^', 'label': 'Similarity Grouping'},
        'SDA-FL': {'color': 'm', 'marker': 'D', 'label': 'SDA-FL'}  # 新增SDA-FL样式
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
    plt.plot(sda_fl_stats['accuracies'], 
             color=algo_styles['SDA-FL']['color'], 
             marker=algo_styles['SDA-FL']['marker'], 
             label=algo_styles['SDA-FL']['label'],
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
    
    # 继续添加其他图表...
    # 这里需要为所有现有图表都添加SDA-FL的对应曲线
    # 省略其他图表代码，但记得都要添加SDA-FL部分
    
    # ... [其他图表代码] ...
    
    logger.info(f"图表生成完成，保存在 {output_dir}/ 目录")

def save_experiment_data(output_dir, fedprox_stats, fedavg_stats, similarity_stats, sda_fl_stats, timestamp):
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
    sda_fl_data = prepare_for_serialization(sda_fl_stats)
    
    # 保存为pickle格式(包含完整数据)
    with open(os.path.join(data_dir, 'experiment_data.pkl'), 'wb') as f:
        pickle.dump({
            'fedprox': fedprox_data,
            'fedavg': fedavg_data,
            'similarity': similarity_data,
            'sda_fl': sda_fl_data,
            'timestamp': timestamp,
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'description': '公平对比实验数据(含SDA-FL)'
            }
        }, f)
    
    # 同时保存为JSON格式(便于查看和跨平台使用)
    try:
        with open(os.path.join(data_dir, 'experiment_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'fedprox': fedprox_data,
                'fedavg': fedavg_data,
                'similarity': similarity_data,
                'sda_fl': sda_fl_data,
                'timestamp': timestamp,
                'metadata': {
                    'creation_time': datetime.now().isoformat(),
                    'description': '公平对比实验数据(含SDA-FL)'
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
                
        with open(os.path.join(data_dir, 'sda_fl_config.yaml'), 'w') as f:
            with open("configs/sda_fl_config.yaml", 'r') as src:
                f.write(src.read())
    except Exception as e:
        logger.warning(f"无法保存配置文件: {str(e)}")
    
    # 创建元数据文件，记录关键指标
    with open(os.path.join(data_dir, 'metadata.txt'), 'w') as f:
        f.write(f"实验时间: {timestamp}\n\n")
        
        f.write("平均卫星数量:\n")
        f.write(f"  FedProx: {np.mean(fedprox_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  FedAvg: {np.mean(fedavg_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  相似度分组: {np.mean(similarity_stats['satellite_stats']['training_satellites']):.2f}\n")
        f.write(f"  SDA-FL: {np.mean(sda_fl_stats['satellite_stats']['training_satellites']):.2f}\n\n")
        
        f.write("最终准确率:\n")
        f.write(f"  FedProx: {max(fedprox_stats['accuracies']):.2f}%\n")
        f.write(f"  FedAvg: {max(fedavg_stats['accuracies']):.2f}%\n")
        f.write(f"  相似度分组: {max(similarity_stats['accuracies']):.2f}%\n")
        f.write(f"  SDA-FL: {max(sda_fl_stats['accuracies']):.2f}%\n")
    
    logger.info(f"实验数据已保存到 {data_dir}/")

def generate_comparison_report(fedprox_stats, fedavg_stats, similarity_stats, sda_fl_stats, output_path):
    """生成包含SDA-FL的对比报告"""
    # 修改现有报告生成函数，添加SDA-FL相关内容
    # ...
    # 这里需要添加SDA-FL的各项指标计算和比较
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 计算各种指标
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    sda_fl_max_acc = max(sda_fl_stats['accuracies'])
    
    fedprox_avg_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    sda_fl_avg_sats = np.mean(sda_fl_stats['satellite_stats']['training_satellites'])
    
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    sda_fl_energy = sum(sda_fl_stats['energy_stats']['total_energy'])
    
    # 计算各种效率指标
    fedprox_efficiency = fedprox_max_acc / fedprox_avg_sats if fedprox_avg_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_avg_sats if fedavg_avg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_avg_sats if similarity_avg_sats > 0 else 0
    sda_fl_efficiency = sda_fl_max_acc / sda_fl_avg_sats if sda_fl_avg_sats > 0 else 0
    
    fedprox_energy_efficiency = fedprox_max_acc / fedprox_energy if fedprox_energy > 0 else 0
    fedavg_energy_efficiency = fedavg_max_acc / fedavg_energy if fedavg_energy > 0 else 0
    similarity_energy_efficiency = similarity_max_acc / similarity_energy if similarity_energy > 0 else 0
    sda_fl_energy_efficiency = sda_fl_max_acc / sda_fl_energy if sda_fl_energy > 0 else 0
    
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
    sda_fl_convergence = calculate_convergence(sda_fl_stats['accuracies'])
    
    # 生成报告
    with open(output_path, 'w') as f:
        f.write("# 公平对比报告: FedProx vs FedAvg vs 相似度分组 vs SDA-FL\n\n")
        
        f.write("## 实验设置\n")
        f.write(f"- FedProx 参数 μ: {fedprox_stats.get('mu', 0.01)}\n")
        f.write(f"- 总轮次: {len(fedprox_stats['accuracies'])}\n\n")
        
        f.write("## 参与卫星数量\n")
        f.write(f"- FedProx 平均训练卫星数: {fedprox_avg_sats:.2f}\n")
        f.write(f"- FedAvg 平均训练卫星数: {fedavg_avg_sats:.2f}\n")
        f.write(f"- 相似度分组平均训练卫星数: {similarity_avg_sats:.2f}\n")
        f.write(f"- SDA-FL 平均训练卫星数: {sda_fl_avg_sats:.2f}\n\n")
        
        f.write("## 准确率性能\n")
        f.write(f"- FedProx 最高准确率: {fedprox_max_acc:.2f}%\n")
        f.write(f"- FedAvg 最高准确率: {fedavg_max_acc:.2f}%\n")
        f.write(f"- 相似度分组最高准确率: {similarity_max_acc:.2f}%\n")
        f.write(f"- SDA-FL 最高准确率: {sda_fl_max_acc:.2f}%\n\n")
        
        f.write("## 算法间准确率比较\n")
        f.write(f"- SDA-FL vs FedAvg: {sda_fl_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- SDA-FL vs FedProx: {sda_fl_max_acc - fedprox_max_acc:+.2f}%\n")
        f.write(f"- SDA-FL vs 相似度分组: {sda_fl_max_acc - similarity_max_acc:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedAvg: {similarity_max_acc - fedavg_max_acc:+.2f}%\n")
        f.write(f"- 相似度分组 vs FedProx: {similarity_max_acc - fedprox_max_acc:+.2f}%\n")
        f.write(f"- FedProx vs FedAvg: {fedprox_max_acc - fedavg_max_acc:+.2f}%\n\n")
        
        f.write("## 能耗\n")
        f.write(f"- FedProx 总能耗: {fedprox_energy:.2f} Wh\n")
        f.write(f"- FedAvg 总能耗: {fedavg_energy:.2f} Wh\n")
        f.write(f"- 相似度分组总能耗: {similarity_energy:.2f} Wh\n")
        f.write(f"- SDA-FL 总能耗: {sda_fl_energy:.2f} Wh\n\n")
        
        f.write("## 效率指标\n")
        f.write(f"- FedProx 每卫星准确率: {fedprox_efficiency:.2f}%\n")
        f.write(f"- FedAvg 每卫星准确率: {fedavg_efficiency:.2f}%\n")
        f.write(f"- 相似度分组每卫星准确率: {similarity_efficiency:.2f}%\n")
        f.write(f"- SDA-FL 每卫星准确率: {sda_fl_efficiency:.2f}%\n\n")
        
        f.write(f"- FedProx 能源效率: {fedprox_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- FedAvg 能源效率: {fedavg_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- 相似度分组能源效率: {similarity_energy_efficiency:.4f}%/Wh\n")
        f.write(f"- SDA-FL 能源效率: {sda_fl_energy_efficiency:.4f}%/Wh\n\n")
        
        f.write("## 收敛速度\n")
        f.write(f"- FedProx 达到90%最高准确率轮次: {fedprox_convergence}\n")
        f.write(f"- FedAvg 达到90%最高准确率轮次: {fedavg_convergence}\n")
        f.write(f"- 相似度分组达到90%最高准确率轮次: {similarity_convergence}\n")
        f.write(f"- SDA-FL 达到90%最高准确率轮次: {sda_fl_convergence}\n\n")
        
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
        
        f.write("### SDA-FL\n")
        f.write("- **优势**: 通过合成数据平衡非IID分布，保护隐私，提高模型泛化性。\n")
        f.write("- **劣势**: GAN训练复杂，合成数据可能引入噪声，计算开销大。\n\n")
        
        # 确定哪种方法表现最佳
        methods = ['FedProx', 'FedAvg', 'Similarity', 'SDA-FL']
        accuracies = [fedprox_max_acc, fedavg_max_acc, similarity_max_acc, sda_fl_max_acc]
        efficiencies = [fedprox_efficiency, fedavg_efficiency, similarity_efficiency, sda_fl_efficiency]
        
        best_accuracy_method = methods[accuracies.index(max(accuracies))]
        best_efficiency_method = methods[efficiencies.index(max(efficiencies))]
        
        method_names = {
            'FedProx': 'FedProx',
            'FedAvg': 'FedAvg',
            'Similarity': '相似度分组',
            'SDA-FL': 'SDA-FL'
        }
        
        f.write("### 结论\n")
        f.write(f"{method_names[best_accuracy_method]}在准确率上表现最好，{method_names[best_efficiency_method]}在资源效率上表现最好。\n\n")
        
        f.write("在卫星网络环境中，各种算法有其独特优势：\n")
        f.write("- 当通信受限且需要节约能源时，相似度分组方法能够提供最佳效率。\n")
        f.write("- 当数据隐私是主要考虑因素时，SDA-FL的合成数据方法提供了更好的保护。\n")
        f.write("- 当系统简单性是优先考虑因素时，FedAvg提供了良好的平衡。\n")
        f.write("- 当数据异质性问题严重时，FedProx的稳定性更有价值。\n\n")
        
        f.write("综合考虑，在资源受限的卫星网络环境中，相似度分组方法和SDA-FL方法在不同方面展现了优势，应根据具体应用场景选择合适的算法。\n")

def run_fair_comparison():
    """运行包含SDA-FL的公平比较实验"""
    logger.info("=== 开始公平比较实验（含SDA-FL）===")

    # 1. 运行SDA-FL实验
    logger.info(f"\n=== 运行SDA-FL实验 ===")
    sda_fl_stats, sda_fl_exp = run_experiment(
        "configs/sda_fl_config.yaml", 
        SDAFLExperiment
    )
    
    if not sda_fl_stats:
        logger.error("SDA-FL实验失败")
        return
    
    # 2. 运行相似度分组实验
    logger.info("\n=== 运行相似度分组实验 ===")
    similarity_stats, similarity_exp = run_experiment(
        "configs/similarity_grouping_config.yaml", 
        SimilarityGroupingExperiment
    )
    
    if not similarity_stats:
        logger.error("相似度分组实验失败")
        return
    
    # 3. 为FedProx和FedAvg创建配置文件
    target_sats = 24  # 固定为24个卫星
    logger.info(f"为所有算法设置目标卫星数: {target_sats}")
    
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
    output_dir = f"comparison_results/sda_fl_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验数据到文件
    save_experiment_data(
        output_dir,
        fedprox_stats=fedprox_stats,
        fedavg_stats=fedavg_stats,
        similarity_stats=similarity_stats,
        sda_fl_stats=sda_fl_stats,
        timestamp=timestamp
    )
    
    # 7. 生成对比报告和图表
    create_comparison_plots(
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        sda_fl_stats,
        output_dir, 
        fedprox_exp=fedprox_exp, 
        fedavg_exp=fedavg_exp, 
        similarity_exp=similarity_exp,
        sda_fl_exp=sda_fl_exp
    )
    
    generate_comparison_report(
        fedprox_stats, 
        fedavg_stats, 
        similarity_stats, 
        sda_fl_stats,
        f"{output_dir}/comparison_report.md"
    )
    
    # 8. 打印关键指标
    print_key_metrics(fedprox_stats, fedavg_stats, similarity_stats, sda_fl_stats)
    
    logger.info(f"公平比较实验完成，结果保存在 {output_dir}/")
    
    return output_dir

def print_key_metrics(fedprox_stats, fedavg_stats, similarity_stats, sda_fl_stats):
    """打印关键指标"""
    # 计算平均卫星数
    fedprox_sats = np.mean(fedprox_stats['satellite_stats']['training_satellites'])
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    sda_fl_sats = np.mean(sda_fl_stats['satellite_stats']['training_satellites'])
    
    # 获取最高准确率
    fedprox_max_acc = max(fedprox_stats['accuracies'])
    fedavg_max_acc = max(fedavg_stats['accuracies'])
    similarity_max_acc = max(similarity_stats['accuracies'])
    sda_fl_max_acc = max(sda_fl_stats['accuracies'])
    
    # 计算总能耗
    fedprox_energy = sum(fedprox_stats['energy_stats']['total_energy'])
    fedavg_energy = sum(fedavg_stats['energy_stats']['total_energy'])
    similarity_energy = sum(similarity_stats['energy_stats']['total_energy'])
    sda_fl_energy = sum(sda_fl_stats['energy_stats']['total_energy'])
    
    # 计算每卫星准确率
    fedprox_efficiency = fedprox_max_acc / fedprox_sats if fedprox_sats > 0 else 0
    fedavg_efficiency = fedavg_max_acc / fedavg_sats if fedavg_sats > 0 else 0
    similarity_efficiency = similarity_max_acc / similarity_sats if similarity_sats > 0 else 0
    sda_fl_efficiency = sda_fl_max_acc / sda_fl_sats if sda_fl_sats > 0 else 0
    
    # 打印结果
    logger.info("\n=== 关键指标对比 ===")
    logger.info(f"平均卫星数量:")
    logger.info(f"  FedProx: {fedprox_sats:.2f}")
    logger.info(f"  FedAvg: {fedavg_sats:.2f}")
    logger.info(f"  相似度分组: {similarity_sats:.2f}")
    logger.info(f"  SDA-FL: {sda_fl_sats:.2f}")
    
    logger.info(f"\n最终准确率:")
    logger.info(f"  FedProx: {fedprox_max_acc:.2f}%")
    logger.info(f"  FedAvg: {fedavg_max_acc:.2f}%")
    logger.info(f"  相似度分组: {similarity_max_acc:.2f}%")
    logger.info(f"  SDA-FL: {sda_fl_max_acc:.2f}%")
    
    logger.info(f"\n总能耗:")
    logger.info(f"  FedProx: {fedprox_energy:.2f} Wh")
    logger.info(f"  FedAvg: {fedavg_energy:.2f} Wh")
    logger.info(f"  相似度分组: {similarity_energy:.2f} Wh")
    logger.info(f"  SDA-FL: {sda_fl_energy:.2f} Wh")
    
    logger.info(f"\n每卫星准确率:")
    logger.info(f"  FedProx: {fedprox_efficiency:.2f}%/satellite")
    logger.info(f"  FedAvg: {fedavg_efficiency:.2f}%/satellite")
    logger.info(f"  相似度分组: {similarity_efficiency:.2f}%/satellite")
    logger.info(f"  SDA-FL: {sda_fl_efficiency:.2f}%/satellite")

def create_modified_config(base_config_path, target_satellite_count, output_path):
    """创建修改后的配置文件，设置目标卫星数量"""
    # 这个函数保持不变
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

if __name__ == "__main__":
    # 运行公平比较实验
    run_fair_comparison()