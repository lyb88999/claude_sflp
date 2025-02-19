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