from fedavg_experiment import FedAvgExperiment
from visualization.visualization import Visualization

def main():
    # 运行FedAvg实验
    experiment = FedAvgExperiment()
    
    # 准备数据
    experiment.prepare_data()
    
    # 设置客户端
    experiment.setup_clients()
    
    # 执行训练并获取统计信息
    stats = experiment.train()
    
    # 生成可视化
    visualizer = Visualization()
    visualizer.plot_training_metrics(
        accuracies=stats['accuracies'],
        losses=stats['losses'],
        energy_stats=stats['energy_stats'],
        satellite_stats=stats['satellite_stats'],
        save_path='fedavg_metrics.png'
    )
    
    print("FedAvg实验完成")
    return stats

if __name__ == "__main__":
    main()