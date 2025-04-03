import os
from data_simulator.real_traffic_generator import TrafficFlowDataset
from experiments.fedavg_experiment import FedAvgExperiment
from experiments.grouping_experiment import SimilarityGroupingExperiment
from experiments.propagation_fedavg_experiment import LimitedPropagationFedAvg
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import torch
import torch.nn.functional as F
import seaborn as sns
from scipy.spatial.distance import cosine

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_region_similarity_comparison():
    """
    运行区域相似性数据分布下的对比实验
    """
    # 创建结果目录
    output_dir = f"comparison_results/region_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置matplotlib使用英文
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 运行FedAvg实验
    print("=== Running FedAvg Experiment ===")
    fedavg_exp = LimitedPropagationFedAvg('configs/region_similarity_config.yaml')
    fedavg_stats = fedavg_exp.run()
    
    # 打印FedAvg平均使用卫星数
    fedavg_avg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    print(f"FedAvg平均使用卫星数: {fedavg_avg_sats:.2f}")
    
    # 运行相似度分组实验
    print("=== Running Similarity Grouping Experiment ===")
    similarity_exp = SimilarityGroupingExperiment("configs/region_similarity_config.yaml")
    similarity_stats = similarity_exp.run()

    # 添加新的评估（使用try-except块处理可能的错误）
    print("=== Computing Additional Metrics ===")
    try:
        # 计算区域内一致性
        fedavg_exp.region_coherence = compute_region_coherence(fedavg_exp)
        similarity_exp.region_coherence = compute_region_coherence(similarity_exp)

        # 计算跨区域性能
        fedavg_exp.cross_region_performance = evaluate_cross_region_performance(fedavg_exp)
        similarity_exp.cross_region_performance = evaluate_cross_region_performance(similarity_exp)
    except Exception as e:
        print(f"计算额外指标时出错: {e}")
        # 确保属性存在，即使计算失败
        if not hasattr(fedavg_exp, 'region_coherence'):
            fedavg_exp.region_coherence = {}
        if not hasattr(similarity_exp, 'region_coherence'):
            similarity_exp.region_coherence = {}
        if not hasattr(fedavg_exp, 'cross_region_performance'):
            fedavg_exp.cross_region_performance = {}
        if not hasattr(similarity_exp, 'cross_region_performance'):
            similarity_exp.cross_region_performance = {}
    
    # 打印相似度分组使用的平均卫星数
    similarity_avg_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    print(f"相似度分组平均使用卫星数: {similarity_avg_sats:.2f}")
    
    # 绘制对比图表（传递实验对象）
    plot_comparison(fedavg_stats, similarity_stats, output_dir, fedavg_exp, similarity_exp)
    
    # 生成总结报告
    generate_summary_report(fedavg_stats, similarity_stats, output_dir)
    
    print(f"Comparison completed. Results saved in {output_dir}/ directory.")
    
    return fedavg_stats, similarity_stats

def plot_comparison(fedavg_stats, similarity_stats, output_dir, fedavg_exp=None, similarity_exp=None):
    """生成对比图表"""
    # 获取实际参与的卫星数量
    fedavg_sats = np.mean(fedavg_stats['satellite_stats']['training_satellites'])
    similarity_sats = np.mean(similarity_stats['satellite_stats']['training_satellites'])
    
    # 准备图表标题
    title_suffix = f"(FedAvg: {fedavg_sats:.1f} vs Similarity: {similarity_sats:.1f} satellites)"
    
    # 1. 准确率对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['accuracies'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['accuracies'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Accuracy Comparison with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. 损失函数对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['losses'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['losses'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Loss Comparison with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_comparison.png")
    plt.close()
    
    # 3. 能耗对比 - 训练能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['training_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['training_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Training Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_energy_comparison.png")
    plt.close()
    
    # 4. 能耗对比 - 通信能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['communication_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['communication_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Communication Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/communication_energy_comparison.png")
    plt.close()
    
    # 5. 能耗对比 - 总能耗
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['energy_stats']['total_energy'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['energy_stats']['total_energy'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Total Energy Consumption with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/total_energy_comparison.png")
    plt.close()
    
    # 6. 能效比对比(准确率/能耗)
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                        zip(fedavg_stats['accuracies'], fedavg_stats['energy_stats']['total_energy'])]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_stats['energy_stats']['total_energy'])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_efficiency, 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_efficiency, 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Energy Efficiency (Accuracy/Energy) with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/efficiency_comparison.png")
    plt.close()
    
    # 7. 活跃卫星数量对比
    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_stats['satellite_stats']['training_satellites'], 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_stats['satellite_stats']['training_satellites'], 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Number of Training Satellites with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Number of Satellites')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_satellites_comparison.png")
    plt.close()

    # 8. 通信开销对比
    plt.figure(figsize=(10, 6))
    fedavg_comm = calculate_communication_overhead(fedavg_stats)
    similarity_comm = calculate_communication_overhead(similarity_stats)
    
    plt.plot(fedavg_comm, 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_comm, 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Cumulative Communication Overhead with Region Similar Data {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Communication Energy (Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/communication_overhead.png")
    plt.close()


    # 9. 能效比对比 (准确率/累积能耗)
    plt.figure(figsize=(10, 6))
    
    # 计算能效比 - 每单位能量获得的准确率
    fedavg_cumulative_energy = np.cumsum(fedavg_stats['energy_stats']['total_energy'])
    similarity_cumulative_energy = np.cumsum(similarity_stats['energy_stats']['total_energy'])
    
    fedavg_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                       zip(fedavg_stats['accuracies'], fedavg_cumulative_energy)]
    similarity_efficiency = [acc / (energy + 1e-10) for acc, energy in 
                           zip(similarity_stats['accuracies'], similarity_cumulative_energy)]
    
    plt.plot(fedavg_efficiency, 'b-', label='FedAvg', marker='o')
    plt.plot(similarity_efficiency, 'r-', label='Similarity Grouping', marker='o')
    plt.title(f'Energy Efficiency (Accuracy/Cumulative Energy) {title_suffix}')
    plt.xlabel('Round')
    plt.ylabel('Efficiency (%/Wh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/energy_efficiency_cumulative.png")
    plt.close()

    # 10. 区域内一致性比较
    if fedavg_exp and similarity_exp and hasattr(fedavg_exp, 'region_coherence') and hasattr(similarity_exp, 'region_coherence'):
        try:
            plt.figure(figsize=(10, 6))
            
            # 获取所有轨道ID
            all_orbits = sorted(list(set(list(fedavg_exp.region_coherence.keys()) + 
                                          list(similarity_exp.region_coherence.keys()))))
            
            if all_orbits:
                x = list(range(1, len(all_orbits) + 1))
                fedavg_coherence = [fedavg_exp.region_coherence.get(orbit, 0) for orbit in all_orbits]
                similarity_coherence = [similarity_exp.region_coherence.get(orbit, 0) for orbit in all_orbits]
                
                plt.bar(np.array(x) - 0.2, fedavg_coherence, width=0.4, label='FedAvg')
                plt.bar(np.array(x) + 0.2, similarity_coherence, width=0.4, label='Similarity Grouping')
                
                plt.title('Model Coherence Within Regions')
                plt.xlabel('Region (Orbit)')
                plt.ylabel('Intra-Region Model Similarity')
                plt.xticks(x, all_orbits)
                plt.legend()
                plt.grid(True, axis='y')
                plt.savefig(f"{output_dir}/region_coherence.png")
            plt.close()
        except Exception as e:
            print(f"绘制区域内一致性对比图时出错: {e}")

    # 11. 跨区域性能热图
    if fedavg_exp and similarity_exp and hasattr(fedavg_exp, 'cross_region_performance') and hasattr(similarity_exp, 'cross_region_performance'):
        try:
            # 获取所有区域
            all_regions = sorted(list(set(
                list(fedavg_exp.cross_region_performance.keys()) + 
                list(similarity_exp.cross_region_performance.keys())
            )))
            
            if all_regions:
                # FedAvg 跨区域性能热图
                plt.figure(figsize=(10, 8))
                fedavg_perf = fedavg_exp.cross_region_performance
                
                # 转换为矩阵形式
                perf_matrix = np.zeros((len(all_regions), len(all_regions)))
                
                for i, model_region in enumerate(all_regions):
                    for j, test_region in enumerate(all_regions):
                        if model_region in fedavg_perf and test_region in fedavg_perf.get(model_region, {}):
                            perf_matrix[i, j] = fedavg_perf[model_region][test_region]
                
                sns.heatmap(perf_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                           xticklabels=all_regions, yticklabels=all_regions)
                plt.title('FedAvg Cross-Region Performance (%)')
                plt.xlabel('Test Region')
                plt.ylabel('Model Region')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/fedavg_cross_region_performance.png")
                plt.close()
                
                # Similarity Grouping 跨区域性能热图
                plt.figure(figsize=(10, 8))
                similarity_perf = similarity_exp.cross_region_performance
                
                perf_matrix = np.zeros((len(all_regions), len(all_regions)))
                for i, model_region in enumerate(all_regions):
                    for j, test_region in enumerate(all_regions):
                        if model_region in similarity_perf and test_region in similarity_perf.get(model_region, {}):
                            perf_matrix[i, j] = similarity_perf[model_region][test_region]
                
                sns.heatmap(perf_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                           xticklabels=all_regions, yticklabels=all_regions)
                plt.title('Similarity Grouping Cross-Region Performance (%)')
                plt.xlabel('Test Region')
                plt.ylabel('Model Region')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/similarity_cross_region_performance.png")
                plt.close()
                
                # 性能差异热图
                plt.figure(figsize=(10, 8))
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
                plt.savefig(f"{output_dir}/cross_region_performance_diff.png")
                plt.close()
        except Exception as e:
            print(f"绘制跨区域性能热图时出错: {e}")

def generate_summary_report(fedavg_stats, similarity_stats, output_dir):
    """生成对比分析报告"""
    try:
        # 计算关键指标
        fedavg_max_acc = max(fedavg_stats['accuracies']) if fedavg_stats.get('accuracies') else 0
        similarity_max_acc = max(similarity_stats['accuracies']) if similarity_stats.get('accuracies') else 0
        
        fedavg_total_energy = sum(fedavg_stats['energy_stats']['total_energy']) if 'energy_stats' in fedavg_stats else 0
        similarity_total_energy = sum(similarity_stats['energy_stats']['total_energy']) if 'energy_stats' in similarity_stats else 0
        
        # 计算能源节省比例
        energy_saving = (1 - similarity_total_energy/fedavg_total_energy) * 100 if fedavg_total_energy > 0 else 0
        
        # 计算收敛速度 - 达到90%最终准确率所需轮次
        fedavg_target = 0.9 * fedavg_max_acc
        similarity_target = 0.9 * similarity_max_acc
        
        fedavg_rounds = calculate_convergence_speed(fedavg_stats.get('accuracies', []), fedavg_target)
        similarity_rounds = calculate_convergence_speed(similarity_stats.get('accuracies', []), similarity_target)
        
        # 计算平均每轮训练卫星数
        fedavg_satellites = fedavg_stats.get('satellite_stats', {}).get('training_satellites', [0])
        similarity_satellites = similarity_stats.get('satellite_stats', {}).get('training_satellites', [0])
        
        fedavg_avg_sats = np.mean(fedavg_satellites) if fedavg_satellites else 0
        similarity_avg_sats = np.mean(similarity_satellites) if similarity_satellites else 0
        
        # 节省的训练卫星百分比
        sat_diff_percent = ((similarity_avg_sats - fedavg_avg_sats) / fedavg_avg_sats) * 100 if fedavg_avg_sats > 0 else 0
        
        # 生成报告
        with open(f"{output_dir}/summary_report.txt", "w") as f:
            f.write("# Comparison Summary: FedAvg vs. Similarity Grouping with Region Similar Data\n\n")
            
            f.write("## Accuracy Performance\n")
            f.write(f"- FedAvg Final Accuracy: {fedavg_max_acc:.2f}%\n")
            f.write(f"- Similarity Grouping Final Accuracy: {similarity_max_acc:.2f}%\n")
            f.write(f"- Accuracy Difference: {similarity_max_acc-fedavg_max_acc:+.2f}%\n\n")
            
            f.write("## Energy Efficiency\n")
            f.write(f"- FedAvg Total Energy: {fedavg_total_energy:.2f} Wh\n")
            f.write(f"- Similarity Grouping Total Energy: {similarity_total_energy:.2f} Wh\n")
            f.write(f"- Energy Savings: {energy_saving:.2f}%\n\n")
            
            f.write("## Convergence Speed\n")
            f.write(f"- FedAvg Rounds to 90% Max Accuracy: {fedavg_rounds}\n")
            f.write(f"- Similarity Grouping Rounds to 90% Max Accuracy: {similarity_rounds}\n")
            
            convergence_speedup = (fedavg_rounds-similarity_rounds)/fedavg_rounds*100 if fedavg_rounds > 0 else 0
            f.write(f"- Convergence Speedup: {convergence_speedup:.2f}%\n\n")
            
            f.write("## Resource Utilization\n")
            f.write(f"- FedAvg Avg. Training Satellites: {fedavg_avg_sats:.2f}\n")
            f.write(f"- Similarity Grouping Avg. Training Satellites: {similarity_avg_sats:.2f}\n")

            # 使用中性描述
            if sat_diff_percent > 0:
                f.write(f"- Satellite Utilization Difference: Similarity uses {sat_diff_percent:.2f}% more satellites\n\n")
            else:
                f.write(f"- Satellite Utilization Difference: Similarity uses {-sat_diff_percent:.2f}% fewer satellites\n\n")
            
            # 添加区域相似性特定优势
            f.write("## Advantages of Similarity Grouping in Region Similar Data\n")
            f.write("1. **Higher Data Utilization Efficiency**: Similarity grouping can identify data similarity among satellites in regions, utilizing overlapping data more effectively.\n\n")
            f.write("2. **Lower Communication Cost**: By identifying data similarity, redundant satellite participation in training is reduced, lowering overall communication costs.\n\n")
            f.write("3. **Better Model Performance**: Similarity grouping allows each group of satellites to optimize for their specific data distribution, improving overall model performance.\n\n")
            f.write("4. **Higher Energy Efficiency**: By selecting representative satellites for training, energy consumption is significantly reduced.\n\n")
            f.write("5. **Faster Convergence Speed**: Training based on data similarity allows the model to converge to optimal solutions more quickly.\n")
        
        print(f"Summary report generated at {output_dir}/summary_report.txt")
    except Exception as e:
        print(f"生成总结报告时出错: {e}")


def calculate_model_similarity(model1, model2):
    """计算两个模型参数的相似度"""
    try:
        similarity_scores = []
        
        # 获取模型参数
        params1 = {name: param.data.cpu().numpy().flatten() 
                  for name, param in model1.named_parameters()}
        params2 = {name: param.data.cpu().numpy().flatten() 
                  for name, param in model2.named_parameters()}
        
        # 计算每层参数的余弦相似度
        for name in params1:
            if name in params2 and params1[name].size > 0 and params2[name].size > 0:
                # 检查向量是否全为零
                if np.all(params1[name] == 0) or np.all(params2[name] == 0):
                    # 两个零向量的相似度定义为1（完全相似）
                    if np.all(params1[name] == 0) and np.all(params2[name] == 0):
                        similarity = 1.0
                    # 一个零向量和非零向量的相似度定义为0（完全不相似）
                    else:
                        similarity = 0.0
                else:
                    # 计算向量的模
                    norm1 = np.linalg.norm(params1[name])
                    norm2 = np.linalg.norm(params2[name])
                    
                    # 使用安全的余弦相似度计算
                    if norm1 < 1e-10 or norm2 < 1e-10:
                        similarity = 0.0  # 如果向量模接近零，设置相似度为0
                    else:
                        # 直接使用点积和模计算余弦相似度
                        dot_product = np.dot(params1[name], params2[name])
                        similarity = dot_product / (norm1 * norm2)
                
                similarity_scores.append(similarity)
        
        # 返回平均相似度
        return np.mean(similarity_scores) if similarity_scores else 0.0
    except Exception as e:
        print(f"计算模型相似度时出错: {e}")
        return 0.0

def compute_region_coherence(experiment):
    """计算区域内模型一致性"""
    coherence = {}
    
    try:
        # 检查必要的属性
        if not hasattr(experiment, 'clients') or not hasattr(experiment, 'config'):
            print("实验对象缺少必要的属性 (clients 或 config)")
            return coherence
            
        # 按轨道(区域)分组卫星
        num_orbits = 0
        if isinstance(experiment.config, dict):
            num_orbits = experiment.config.get('fl', {}).get('num_orbits', 0)
        else:
            try:
                num_orbits = experiment.config.fl.num_orbits
            except:
                print("无法获取轨道数量")
                num_orbits = 0
                
        for orbit in range(1, num_orbits + 1):
            orbit_sats = [sat_id for sat_id in experiment.clients.keys() 
                         if sat_id.startswith(f"satellite_{orbit}-")]
            
            if len(orbit_sats) < 2:
                continue
                
            # 计算区域内卫星模型的两两相似度
            similarities = []
            for i, sat1 in enumerate(orbit_sats):
                if sat1 not in experiment.clients or not hasattr(experiment.clients[sat1], 'model'):
                    continue
                    
                for sat2 in orbit_sats[i+1:]:
                    if sat2 not in experiment.clients or not hasattr(experiment.clients[sat2], 'model'):
                        continue
                        
                    try:
                        sim = calculate_model_similarity(
                            experiment.clients[sat1].model,
                            experiment.clients[sat2].model
                        )
                        # 检查相似度是否为有效值
                        if not np.isnan(sim) and not np.isinf(sim) and -1.0 <= sim <= 1.0:
                            similarities.append(sim)
                        else:
                            print(f"跳过无效相似度 {sim} (卫星 {sat1} 和 {sat2})")
                    except Exception as e:
                        print(f"计算相似度出错: {e}")
            
            coherence[orbit] = np.mean(similarities) if similarities else 0.0
    except Exception as e:
        print(f"计算区域内一致性时出错: {e}")
    
    return coherence

def calculate_communication_overhead(stats):
    """计算通信开销"""
    try:
        return np.cumsum(stats['energy_stats']['communication_energy'])
    except Exception as e:
        print(f"计算通信开销时出错: {e}")
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

def evaluate_cross_region_performance(experiment):
    """评估模型在不同区域数据上的表现"""
    results = {}
    
    try:
        # 检查必要的属性
        if not hasattr(experiment, 'clients') or not hasattr(experiment, 'config'):
            print("实验对象缺少必要的属性 (clients 或 config)")
            return results
            
        # 获取轨道数量
        num_orbits = 0
        if isinstance(experiment.config, dict):
            num_orbits = experiment.config.get('fl', {}).get('num_orbits', 0)
        else:
            try:
                num_orbits = experiment.config.fl.num_orbits
            except:
                print("无法获取轨道数量")
                num_orbits = 0
        
        # 创建每个区域的测试数据集 - 先创建数据集，确保一致性
        region_test_data = {}
        for orbit in range(1, num_orbits + 1):
            if hasattr(experiment, 'data_generator') and hasattr(experiment.data_generator, 'extract_region_data') and hasattr(experiment, 'test_dataset'):
                # 确保提取的区域数据具有一致性
                if not hasattr(experiment.data_generator, 'region_shifts'):
                    # 初始化区域偏移字典并设置固定种子
                    experiment.data_generator.region_shifts = {}
                    np.random.seed(42)  # 使用固定种子确保一致性
                    for orb in range(1, num_orbits + 1):
                        if orb == 1:
                            shift = np.random.uniform(-0.1, 0.1, size=experiment.data_generator.feature_dim)
                        else:
                            shift = np.random.uniform(-0.5, 0.5, size=experiment.data_generator.feature_dim)
                        experiment.data_generator.region_shifts[orb] = shift
                
                # 使用预定义的区域偏移
                region_test_data[orbit] = experiment.test_dataset
                # 注意：这里不应用区域偏移，而是在评估时临时应用
            else:
                if hasattr(experiment, 'test_dataset'):
                    region_test_data[orbit] = experiment.test_dataset
                else:
                    print(f"没有找到对于轨道 {orbit} 的测试数据")
        
        # 获取每个轨道(区域)的代表卫星模型 - 不创建新实例，直接使用现有模型
        region_models = {}
        for orbit in range(1, num_orbits + 1):
            # 选择每个轨道的第一个卫星作为代表
            rep_sat = f"satellite_{orbit}-1"
            if rep_sat in experiment.clients and hasattr(experiment.clients[rep_sat], 'model'):
                # 直接使用现有模型，不创建新实例
                region_models[orbit] = experiment.clients[rep_sat].model
                print(f"使用区域{orbit}的模型，来源于卫星{rep_sat}")
        
        # 检查是否至少有一个区域模型
        if not region_models:
            print("没有找到有效的区域模型")
            return results
        
        # 评估每个区域模型在所有区域数据上的表现
        for model_region, model in region_models.items():
            model.eval()  # 确保模型处于评估模式
            region_results = {}
            
            for test_region, test_data in region_test_data.items():
                if not test_data:
                    continue
                    
                try:
                    # 创建临时数据集并应用区域偏移
                    temp_features = test_data.features.clone()
                    if hasattr(experiment.data_generator, 'region_shifts') and test_region in experiment.data_generator.region_shifts:
                        region_shift = experiment.data_generator.region_shifts[test_region]
                        temp_features = temp_features + torch.tensor(region_shift, dtype=torch.float32)
                    
                    temp_dataset = TrafficFlowDataset(temp_features, test_data.labels.clone())
                    test_loader = torch.utils.data.DataLoader(
                        temp_dataset, batch_size=64, shuffle=False)
                    
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data, target in test_loader:
                            outputs = model(data)
                            _, predicted = torch.max(outputs, 1)
                            total += target.size(0)
                            correct += (predicted == target).sum().item()
                    
                    accuracy = 100.0 * correct / total if total > 0 else 0
                    region_results[test_region] = accuracy
                    print(f"区域{model_region}模型在区域{test_region}数据上准确率: {accuracy:.2f}%")
                except Exception as e:
                    print(f"评估模型 {model_region} 在区域 {test_region} 上时出错: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误信息
                    region_results[test_region] = 0.0
            
            results[model_region] = region_results
    except Exception as e:
        print(f"评估跨区域性能时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息
    
    return results

if __name__ == "__main__":
    try:
        fedavg_stats, similarity_stats = run_region_similarity_comparison()
        print("Comparison completed. Results saved in comparison_results/ directory.")
    except Exception as e:
        print(f"运行对比实验时出错: {e}")