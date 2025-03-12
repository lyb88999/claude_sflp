from grouping_experiment import SimilarityGroupingExperiment
from visualization.comparison_visualization import ComparisonVisualization
from experiments.baseline_experiment import BaselineExperiment
import logging
import time
import argparse
import os

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("similarity_grouping.log")
        ]
    )

def run_similarity_grouping(config_path="configs/similarity_grouping_config.yaml"):
    """运行基于数据相似度分组的联邦学习实验"""
    logger = logging.getLogger("similarity_grouping")
    logger.info("启动基于数据相似度分组的联邦学习实验")
    
    start_time = time.time()
    
    # 创建并运行实验
    experiment = SimilarityGroupingExperiment(config_path)
    stats = experiment.run()
    
    end_time = time.time()
    logger.info(f"实验完成，耗时 {end_time - start_time:.2f} 秒")
    
    return stats

def run_comparison(baseline_config="configs/baseline_config.yaml", 
                  grouping_config="configs/similarity_grouping_config.yaml"):
    """运行基线与分组方法的对比实验"""
    logger = logging.getLogger("comparison")
    logger.info("启动对比实验")
    
    # 运行基线实验
    logger.info("开始运行基线实验")
    baseline = BaselineExperiment(baseline_config)
    baseline_stats = baseline.run()
    
    # 运行分组实验
    logger.info("开始运行基于数据相似度分组的实验")
    grouping = SimilarityGroupingExperiment(grouping_config)
    grouping_stats = grouping.run()
    
    # 比较结果
    logger.info("生成对比可视化")
    visualizer = ComparisonVisualization()
    visualizer.plot_comparison(baseline_stats, grouping_stats)
    
    logger.info("对比实验完成")
    
    return baseline_stats, grouping_stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行基于数据相似度分组的联邦学习实验")
    parser.add_argument("--mode", type=str, default="grouping", 
                      choices=["grouping", "comparison"],
                      help="实验模式：单独运行分组实验或运行对比实验")
    parser.add_argument("--config", type=str, default="configs/similarity_grouping_config.yaml",
                      help="分组实验配置文件路径")
    parser.add_argument("--baseline_config", type=str, default="configs/baseline_config.yaml",
                      help="基线实验配置文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    if args.mode == "grouping":
        # 运行单独的分组实验
        run_similarity_grouping(args.config)
    else:
        # 运行对比实验
        run_comparison(args.baseline_config, args.config)

if __name__ == "__main__":
    main()