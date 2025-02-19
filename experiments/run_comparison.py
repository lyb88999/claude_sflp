from baseline_experiment import BaselineExperiment
from grouping_experiment import GroupingExperiment
from visualization.comparison_visualization import ComparisonVisualization


def run_comparison():
    # 运行基准实验
    baseline = BaselineExperiment()
    baseline_stats = baseline.run()
    
    # 运行分组实验
    grouping = GroupingExperiment()
    grouping_stats = grouping.run()
    
    # 比较结果
    visualizer = ComparisonVisualization()
    visualizer.plot_comparison(baseline_stats, grouping_stats)

if __name__ == "__main__":
    run_comparison()