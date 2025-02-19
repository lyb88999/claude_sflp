import logging
import matplotlib.pyplot as plt
import numpy as np

class ComparisonVisualization:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')
        
    def plot_comparison(self, baseline_stats, grouping_stats):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 准确率对比（保持不变）
        ax1.plot(baseline_stats['accuracies'], 'b-', label='Baseline', marker='o')
        ax1.plot(grouping_stats['accuracies'], 'r-', label='Grouping', marker='o')
        ax1.set_title('Accuracy Convergence')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)

        # 2. 能耗细分对比
        rounds = range(len(baseline_stats['energy_stats']['total_energy']))
        ax2.plot(rounds, baseline_stats['energy_stats']['training_energy'], 'b--', 
                label='Baseline Training')
        ax2.plot(rounds, baseline_stats['energy_stats']['communication_energy'], 'b:', 
                label='Baseline Comm')
        ax2.plot(rounds, baseline_stats['energy_stats']['total_energy'], 'b-', 
                label='Baseline Total')
        ax2.plot(rounds, grouping_stats['energy_stats']['training_energy'], 'r--', 
                label='Grouping Training')
        ax2.plot(rounds, grouping_stats['energy_stats']['communication_energy'], 'r:', 
                label='Grouping Comm')
        ax2.plot(rounds, grouping_stats['energy_stats']['total_energy'], 'r-', 
                label='Grouping Total')
        ax2.set_title('Energy Consumption Breakdown')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Energy (Wh)')
        ax2.legend()
        ax2.grid(True)

        # 3. 活跃卫星细分对比
        ax3.plot(rounds, baseline_stats['satellite_stats']['training_satellites'], 'b--', 
                label='Baseline Training')
        ax3.plot(rounds, baseline_stats['satellite_stats']['receiving_satellites'], 'b:', 
                label='Baseline Receiving')
        ax3.plot(rounds, baseline_stats['satellite_stats']['total_active'], 'b-', 
                label='Baseline Total')
        ax3.plot(rounds, grouping_stats['satellite_stats']['training_satellites'], 'r--', 
                label='Grouping Training')
        ax3.plot(rounds, grouping_stats['satellite_stats']['receiving_satellites'], 'r:', 
                label='Grouping Receiving')
        ax3.plot(rounds, grouping_stats['satellite_stats']['total_active'], 'r-', 
                label='Grouping Total')
        ax3.set_title('Active Satellites Breakdown')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Number of Satellites')
        ax3.legend()
        ax3.grid(True)

        # 4. Loss对比（保持不变）
        ax4.plot(baseline_stats['losses'], 'b-', label='Baseline', marker='o')
        ax4.plot(grouping_stats['losses'], 'r-', label='Grouping', marker='o')
        ax4.set_title('Loss Convergence')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('experiment_comparison.png')
        plt.close()