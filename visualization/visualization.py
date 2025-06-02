import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import logging

class Visualization:
    def __init__(self):
        """初始化可视化类"""
        self.logger = logging.getLogger(__name__)
        # 使用 matplotlib 默认样式
        plt.style.use('default')
        # 设置一些基本的样式参数
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def plot_training_metrics(self,
                       accuracies,
                       losses,
                       energy_stats,
                       satellite_stats,
                       save_path='training_metrics.png'):
        """绘制训练指标"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 准确率变化
        ax1.plot(accuracies, 'b-', marker='o')
        ax1.set_title('Accuracy Trend')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True)

        # 2. 能耗分解
        if energy_stats:
            ax2.plot(energy_stats['training_energy'], 'r--', label='Training')
            ax2.plot(energy_stats['communication_energy'], 'g--', label='Communication')
            ax2.plot(energy_stats['total_energy'], 'b-', label='Total')
            ax2.set_title('Energy Consumption Breakdown')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Energy (Wh)')
            ax2.legend()
            ax2.grid(True)

        # 3. 参与卫星数量
        if satellite_stats:
            ax3.plot(satellite_stats['training_satellites'], 'r--', label='Training')
            ax3.plot(satellite_stats['receiving_satellites'], 'g--', label='Receiving')
            ax3.plot(satellite_stats['total_active'], 'b-', label='Total')
            ax3.set_title('Active Satellites')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Number of Satellites')
            ax3.legend()
            ax3.grid(True)

        # 4. 损失函数变化
        ax4.plot(losses, 'r-', marker='o')
        ax4.set_title('Loss Trend')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Loss')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"训练指标可视化已保存至: {save_path}")

    # 在 visualization/visualization.py 中添加新的方法

def plot_training_metrics_extended(self, stats: dict, save_path: str = None):
    """
    绘制扩展的训练指标图表，包含分类性能指标
    
    Args:
        stats: 包含所有训练统计信息的字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # 创建大图表，包含多个子图
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    rounds = range(1, len(stats['accuracies']) + 1)
    
    # 1. 准确率 (保持原有)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, stats['accuracies'], 'b-', marker='o', linewidth=2)
    ax1.set_title('Accuracy over Training Rounds', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    
    # 2. F1值对比 (macro vs weighted)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rounds, stats['f1_macros'], 'g-', marker='s', linewidth=2, label='F1 Macro')
    ax2.plot(rounds, stats['f1_weighteds'], 'orange', marker='^', linewidth=2, label='F1 Weighted')
    ax2.set_title('F1-Score over Training Rounds', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('F1-Score (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 精确率对比
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rounds, stats['precision_macros'], 'r-', marker='d', linewidth=2, label='Precision Macro')
    ax3.plot(rounds, stats['precision_weighteds'], 'purple', marker='v', linewidth=2, label='Precision Weighted')
    ax3.set_title('Precision over Training Rounds', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Precision (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 召回率对比
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(rounds, stats['recall_macros'], 'brown', marker='p', linewidth=2, label='Recall Macro')
    ax4.plot(rounds, stats['recall_weighteds'], 'pink', marker='h', linewidth=2, label='Recall Weighted')
    ax4.set_title('Recall over Training Rounds', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Recall (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 所有分类指标综合对比 (使用macro平均)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rounds, stats['accuracies'], 'b-', marker='o', linewidth=2, label='Accuracy')
    ax5.plot(rounds, stats['precision_macros'], 'r-', marker='s', linewidth=2, label='Precision')
    ax5.plot(rounds, stats['recall_macros'], 'g-', marker='^', linewidth=2, label='Recall')
    ax5.plot(rounds, stats['f1_macros'], 'orange', marker='d', linewidth=2, label='F1-Score')
    ax5.set_title('All Classification Metrics (Macro)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Metric Value (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 损失函数
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(rounds, stats['losses'], 'red', marker='x', linewidth=2)
    ax6.set_title('Training Loss over Rounds', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Round')
    ax6.set_ylabel('Loss')
    ax6.grid(True, alpha=0.3)
    
    # 7. 能耗统计
    ax7 = fig.add_subplot(gs[2, 0])
    energy_data = stats['energy_stats']
    ax7.plot(rounds, energy_data['training_energy'], 'g-', marker='o', linewidth=2, label='Training')
    ax7.plot(rounds, energy_data['communication_energy'], 'b-', marker='s', linewidth=2, label='Communication')
    ax7.plot(rounds, energy_data['total_energy'], 'r-', marker='^', linewidth=2, label='Total')
    ax7.set_title('Energy Consumption', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Round')
    ax7.set_ylabel('Energy (Wh)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 活跃卫星数量
    ax8 = fig.add_subplot(gs[2, 1])
    satellite_data = stats['satellite_stats']
    ax8.plot(rounds, satellite_data['training_satellites'], 'purple', marker='o', linewidth=2, label='Training')
    ax8.plot(rounds, satellite_data['receiving_satellites'], 'cyan', marker='s', linewidth=2, label='Receiving')
    ax8.plot(rounds, satellite_data['total_active'], 'orange', marker='^', linewidth=2, label='Total Active')
    ax8.set_title('Active Satellites per Round', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Round')
    ax8.set_ylabel('Number of Satellites')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 分类性能热力图（如果有最终混淆矩阵）
    ax9 = fig.add_subplot(gs[2, 2])
    if 'final_metrics' in stats and 'confusion_matrix' in stats['final_metrics']:
        cm = stats['final_metrics']['confusion_matrix']
        im = ax9.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax9.figure.colorbar(im, ax=ax9)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax9.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax9.set_title('Final Confusion Matrix', fontsize=12, fontweight='bold')
        ax9.set_ylabel('True Label')
        ax9.set_xlabel('Predicted Label')
        
        # 设置标签
        class_names = ['Malicious', 'Benign']  # 根据你的数据集调整
        ax9.set_xticks(range(len(class_names)))
        ax9.set_yticks(range(len(class_names)))
        ax9.set_xticklabels(class_names)
        ax9.set_yticklabels(class_names)
    else:
        ax9.text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Final Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 10. 性能效率对比
    ax10 = fig.add_subplot(gs[3, :2])
    if stats['energy_stats']['total_energy']:
        # 计算每单位能量的F1值
        f1_per_energy = [f1 / (energy + 1e-10) for f1, energy in 
                        zip(stats['f1_macros'], stats['energy_stats']['total_energy'])]
        
        # 计算每颗卫星的F1值
        f1_per_satellite = [f1 / (sats + 1e-10) for f1, sats in 
                           zip(stats['f1_macros'], stats['satellite_stats']['training_satellites'])]
        
        ax10_twin = ax10.twinx()
        
        line1 = ax10.plot(rounds, f1_per_energy, 'g-', marker='o', linewidth=2, label='F1/Energy')
        line2 = ax10_twin.plot(rounds, f1_per_satellite, 'b-', marker='s', linewidth=2, label='F1/Satellite')
        
        ax10.set_xlabel('Round')
        ax10.set_ylabel('F1 per Energy (F1%/Wh)', color='g')
        ax10_twin.set_ylabel('F1 per Satellite (F1%/Sat)', color='b')
        ax10.set_title('Resource Efficiency Metrics', fontsize=12, fontweight='bold')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax10.legend(lines, labels, loc='upper left')
        
        ax10.grid(True, alpha=0.3)
    
    # 11. 最终指标总结文本
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')
    
    if stats['accuracies']:
        summary_text = f"""Final Performance Summary:
        
Accuracy: {stats['accuracies'][-1]:.2f}%
F1-Score (Macro): {stats['f1_macros'][-1]:.2f}%
F1-Score (Weighted): {stats['f1_weighteds'][-1]:.2f}%
Precision (Macro): {stats['precision_macros'][-1]:.2f}%
Recall (Macro): {stats['recall_macros'][-1]:.2f}%

Best F1-Score: {max(stats['f1_macros']):.2f}%
Total Rounds: {len(stats['accuracies'])}
Avg Satellites: {np.mean(stats['satellite_stats']['training_satellites']):.1f}
Total Energy: {sum(stats['energy_stats']['total_energy']):.2f} Wh"""
        
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Extended Training Metrics Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_classification_metrics_comparison(self, stats_dict: dict, algorithm_names: list, 
                                         save_path: str = None):
    """
    绘制多个算法的分类指标对比图
    
    Args:
        stats_dict: 字典，键为算法名，值为统计数据
        algorithm_names: 算法名称列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Classification Metrics Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    metrics = ['accuracies', 'f1_macros', 'precision_macros', 'recall_macros']
    titles = ['Accuracy Comparison', 'F1-Score Comparison', 
              'Precision Comparison', 'Recall Comparison']
    y_labels = ['Accuracy (%)', 'F1-Score (%)', 'Precision (%)', 'Recall (%)']
    
    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
        ax = axes[idx // 2, idx % 2]
        
        for i, (algo_name, stats) in enumerate(stats_dict.items()):
            if metric in stats and stats[metric]:
                rounds = range(1, len(stats[metric]) + 1)
                ax.plot(rounds, stats[metric], 
                       color=colors[i % len(colors)], 
                       marker=markers[i % len(markers)],
                       linewidth=2, markersize=6, label=algo_name)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()