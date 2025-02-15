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
        
    def plot_all_metrics(self, 
                        accuracies: List[float], 
                        losses: List[float], 
                        energies: List[float],
                        clients: Dict,
                        round_stats: List[Dict],
                        save_path: str = 'training_visualization.png'):
        """
        绘制所有指标的可视化图表
        
        Args:
            accuracies: 每轮的准确率列表
            losses: 每轮的损失值列表
            energies: 每轮的能耗列表
            clients: 所有客户端的字典 {client_id: client_object}
            round_stats: 每轮的统计信息列表
            save_path: 保存图片的路径
        """
        # 创建3x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. 准确率变化趋势
        rounds = range(1, len(accuracies) + 1)
        ax1.plot(rounds, accuracies, 'b-', marker='o')
        ax1.set_title('Global Accuracy Trend')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True)
        
        # 2. 每轨道的平均准确率
        orbit_accuracies = defaultdict(list)
        for sat_id, client in clients.items():
            orbit_num = int(sat_id.split('-')[0].split('_')[1])
            if client.train_stats:
                orbit_accuracies[orbit_num].append(
                    client.train_stats[-1]['summary']['train_accuracy'][-1]
                )
        
        orbits = sorted(orbit_accuracies.keys())
        avg_accuracies = [np.mean(orbit_accuracies[orbit]) for orbit in orbits]
        ax2.bar(orbits, avg_accuracies)
        ax2.set_title('Average Accuracy by Orbit')
        ax2.set_xlabel('Orbit Number')
        ax2.set_ylabel('Average Accuracy (%)')
        
        # 3. 损失函数收敛曲线
        ax3.plot(rounds, losses, 'r-', marker='o')
        ax3.set_title('Loss Convergence')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # 4. 能耗分布
        orbit_energies = defaultdict(list)
        for sat_id, client in clients.items():
            orbit_num = int(sat_id.split('-')[0].split('_')[1])
            if client.train_stats:
                orbit_energies[orbit_num].append(
                    client.train_stats[-1]['summary']['energy_consumption']
                )
        
        avg_energies = [np.mean(orbit_energies[orbit]) for orbit in orbits]
        ax4.bar(orbits, avg_energies)
        ax4.set_title('Average Energy Consumption by Orbit')
        ax4.set_xlabel('Orbit Number')
        ax4.set_ylabel('Energy (Wh)')
        
        # 5. 训练时间分布
        training_times = []
        for client in clients.values():
            if client.train_stats:
                training_times.append(client.train_stats[-1]['summary']['compute_time'])
                
        ax5.hist(training_times, bins=20, color='green', alpha=0.7)
        ax5.set_title('Training Time Distribution')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Frequency')
        
        # 6. 资源效率（准确率/能耗）
        efficiency = [acc/energy for acc, energy in zip(accuracies, energies)]
        ax6.plot(rounds, efficiency, 'm-', marker='o')
        ax6.set_title('Resource Efficiency')
        ax6.set_xlabel('Round')
        ax6.set_ylabel('Accuracy per Wh')
        ax6.grid(True)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"可视化结果已保存至: {save_path}")
        
    def plot_detailed_energy_analysis(self, 
                                    energies: List[float],
                                    clients: Dict,
                                    save_path: str = 'energy_analysis.png'):
        """
        绘制详细的能耗分析图表
        
        Args:
            energies: 每轮的能耗列表
            clients: 所有客户端的字典
            save_path: 保存图片的路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总能耗趋势
        rounds = range(1, len(energies) + 1)
        ax1.plot(rounds, energies, 'r-', marker='o')
        ax1.set_title('Total Energy Consumption Trend')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Energy (Wh)')
        ax1.grid(True)
        
        # 2. 每轮能耗箱线图
        round_energies = []
        for round_num in range(len(energies)):
            round_energy = []
            for client in clients.values():
                if len(client.train_stats) > round_num:
                    round_energy.append(
                        client.train_stats[round_num]['summary']['energy_consumption']
                    )
            round_energies.append(round_energy)
            
        ax2.boxplot(round_energies)
        ax2.set_title('Energy Distribution per Round')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Energy (Wh)')
        
        # 3. 累计能耗
        cumulative_energy = np.cumsum(energies)
        ax3.plot(rounds, cumulative_energy, 'g-', marker='o')
        ax3.set_title('Cumulative Energy Consumption')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Total Energy (Wh)')
        ax3.grid(True)
        
        # 4. 卫星能耗分布
        satellite_energies = []
        for client in clients.values():
            if client.train_stats:
                total_energy = sum(
                    stats['summary']['energy_consumption'] 
                    for stats in client.train_stats
                )
                satellite_energies.append(total_energy)
                
        ax4.hist(satellite_energies, bins=20, color='orange', alpha=0.7)
        ax4.set_title('Total Energy Distribution Across Satellites')
        ax4.set_xlabel('Total Energy (Wh)')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"能耗分析结果已保存至: {save_path}")