from experiments.fedavg_experiment import FedAvgExperiment
import torch
import torch.nn as nn
from fl_core.client.satellite_client import ClientConfig
from fl_core.client.fedprox_client import FedProxClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np

class FedProxExperiment(FedAvgExperiment):
    def __init__(self, config_path: str = "configs/fedprox_config.yaml"):
        super().__init__(config_path)
        self.max_workers = 6
        
        # 从配置获取 FedProx 特有参数 
        self.mu = self.config.get('fedprox', {}).get('mu', 0.01)
        self.logger.info(f"初始化 FedProx 实验，μ={self.mu}")
        
        # 添加FedProx特有的统计信息记录
        self.proximal_stats = {
            'round_proximal_terms': [],  # 每轮的接近性项统计
            'client_drift': {}          # 客户端偏移统计
        }
        
    def train(self):
        """
        执行FedProx训练过程，增强版的train方法，添加了FedProx特有的统计收集
        """
        # 初始化记录列表
        accuracies = []
        losses = []
        energy_stats = {
            'training_energy': [],
            'communication_energy': [],
            'total_energy': []
        }
        satellite_stats = {
            'training_satellites': [],
            'receiving_satellites': [],
            'total_active': []
        }
        
        # FedProx特有统计
        proximal_terms = []

        current_time = datetime.now().timestamp()
        self.current_round = 0
        best_accuracy = 0
        rounds_without_improvement = 0
        
        # 禁用早停或修改参数
        max_rounds_without_improvement = float('inf')  # 设置为无穷大
        min_rounds = self.config['fl']['num_rounds']   # 最小轮数设为总轮数
        accuracy_threshold = 100.0                     # 设置一个不可能达到的准确率阈值

        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"=== FedProx: 开始第 {round_num + 1} 轮训练 (μ={self.mu}) === 时间：{datetime.fromtimestamp(current_time)}")
            
            # 使用线程池并行处理每个地面站的轨道
            orbit_successes = 0
            # 收集所有轨道的统计信息
            round_training_energy = 0
            round_comm_energy = 0
            round_training_sats = set()
            round_receiving_sats = set()
            
            # 收集本轮的接近性项统计
            round_proximal_term = 0.0
            round_proximal_samples = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 创建所有任务
                future_to_orbit = {}
                for station_id, station in self.ground_stations.items():
                    for orbit_id in station.responsible_orbits:
                        future = executor.submit(
                            self._handle_orbit_training,
                            station_id,
                            orbit_id,
                            current_time
                        )
                        future_to_orbit[future] = (station_id, orbit_id)

                # 等待所有任务完成
                for future in as_completed(future_to_orbit):
                    station_id, orbit_id = future_to_orbit[future]
                    try:
                        result = future.result()  # result 是一个元组 (success, orbit_stats)
                        if isinstance(result, tuple) and len(result) == 2:
                            success, orbit_stats = result
                            if success:
                                orbit_successes += 1
                            if orbit_stats:  # 如果有统计信息就收集
                                round_training_energy += orbit_stats['training_energy']
                                round_comm_energy += orbit_stats['communication_energy']
                                round_training_sats.update(orbit_stats['training_satellites'])
                                round_receiving_sats.update(orbit_stats['receiving_satellites'])
                                
                                # 收集FedProx特有统计
                                if 'proximal_term' in orbit_stats:
                                    round_proximal_term += orbit_stats['proximal_term']
                                    round_proximal_samples += orbit_stats.get('proximal_samples', 1)
                        else:
                            self.logger.warning(f"轨道 {orbit_id} 返回的结果格式不正确")
                    except Exception as e:
                        self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")
            
            # 记录本轮统计信息
            energy_stats['training_energy'].append(round_training_energy)
            energy_stats['communication_energy'].append(round_comm_energy)
            energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
            
            satellite_stats['training_satellites'].append(len(round_training_sats))
            satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
            satellite_stats['total_active'].append(len(round_training_sats | round_receiving_sats))
            
            # 记录接近性项统计
            if round_proximal_samples > 0:
                avg_proximal = round_proximal_term / round_proximal_samples
                proximal_terms.append(avg_proximal)
                self.logger.info(f"轮次 {round_num + 1} 平均接近性项: {avg_proximal:.6f}")
            else:
                proximal_terms.append(0.0)

            # 地面站聚合
            if orbit_successes > 0:
                self.logger.info(f"\n=== 地面站聚合阶段 === ({orbit_successes} 个轨道成功)")
                station_results = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_station = {
                        executor.submit(self._station_aggregation, station_id, station): station_id
                        for station_id, station in self.ground_stations.items()
                    }
                    
                    for future in as_completed(future_to_station):
                        station_id = future_to_station[future]
                        try:
                            result = future.result()
                            if result:
                                station_results.append(station_id)
                        except Exception as e:
                            self.logger.error(f"地面站 {station_id} 聚合出错: {str(e)}")

                # 全局聚合
                if len(station_results) == len(self.ground_stations):
                    self.logger.info("\n=== 全局聚合阶段 ===")
                    success = self._perform_global_aggregation(round_num)
                    
                    if success:
                        # 评估准确率
                        accuracy = self.evaluate()
                        accuracies.append(accuracy)
                        
                        # 计算当前轮次的总损失和能耗
                        round_energy = 0
                        round_loss = 0
                        round_stat = {}
                        client_count = 0
                        
                        for client in self.clients.values():
                            if client.train_stats and client.client_id in round_training_sats:
                                round_energy += client.train_stats[-1]['summary']['energy_consumption']
                                round_loss += client.train_stats[-1]['summary']['train_loss'][-1]
                                round_stat[client.client_id] = client.train_stats[-1]
                                client_count += 1
                                
                        if client_count > 0:
                            losses.append(round_loss / client_count)  # 平均损失
                        else:
                            losses.append(0)

                        self.logger.info(f"第 {round_num + 1} 轮全局准确率: {accuracy:.4f}")
                        self.logger.info(f"接近性项 (μ={self.mu}): {proximal_terms[-1]:.6f}")
                        
                        # 更新最佳准确率和检查提升情况
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            rounds_without_improvement = 0
                            self.logger.info(f"找到更好的模型！新的最佳准确率: {accuracy:.4f}")
                        else:
                            rounds_without_improvement += 1
                            self.logger.info(f"准确率未提升，已经 {rounds_without_improvement} 轮没有改进")

                        current_time += self.config['fl']['round_interval']
                        
                    else:
                        self.logger.warning("全局聚合失败")
                else:
                    self.logger.warning(f"只有 {len(station_results)}/{len(self.ground_stations)} 个地面站完成聚合，跳过全局聚合")
            else:
                self.logger.warning("所有轨道训练失败，跳过聚合阶段")

            current_time += self.config['fl']['round_interval']
            
        self.logger.info(f"\n=== FedProx 训练结束 ===")
        self.logger.info(f"总轮次: {self.current_round + 1}")
        self.logger.info(f"最佳准确率: {best_accuracy:.4f}")
        self.logger.info(f"接近性参数 μ: {self.mu}")
        
        # 保存接近性项统计
        self.proximal_stats['round_proximal_terms'] = proximal_terms
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats,
            'proximal_terms': proximal_terms,
            'mu': self.mu
        }

        # 生成可视化
        self.visualizer.plot_training_metrics(
            accuracies=stats['accuracies'],
            losses=stats['losses'],
            energy_stats=stats['energy_stats'],
            satellite_stats=stats['satellite_stats'],
            save_path=self.log_dir / 'training_metrics.png'  # 保存在实验日志目录
        )

        return stats
        
    def _handle_orbit_training(self, station_id, orbit_id, current_time):
        """
        处理单个轨道的训练过程，增强版适用于FedProx
        """
        # 调用父类方法处理大部分逻辑
        result = super()._handle_orbit_training(station_id, orbit_id, current_time)
        
        # 如果result有效，增加FedProx特有的统计信息
        if isinstance(result, tuple) and len(result) == 2:
            success, orbit_stats = result
            
            # 收集接近性项统计信息
            proximal_term_sum = 0.0
            proximal_samples = 0
            
            for sat_id in orbit_stats.get('training_satellites', []):
                if hasattr(self.clients[sat_id], 'mu') and sat_id in self.clients:
                    client = self.clients[sat_id]
                    # 这里假设FedProxClient有一个属性存储了上一次训练的接近性项
                    if hasattr(client, 'last_proximal_term'):
                        proximal_term_sum += client.last_proximal_term
                        proximal_samples += 1
            
            if proximal_samples > 0:
                orbit_stats['proximal_term'] = proximal_term_sum
                orbit_stats['proximal_samples'] = proximal_samples
                
            return success, orbit_stats
        
        return result
    
    def setup_clients(self):
        """设置卫星客户端，使用 FedProxClient 代替 SatelliteClient"""

        client_cfg = self.config.get('client', {}) 
        client_config = ClientConfig(
             local_epochs=client_cfg.get('local_epochs', 1), 
             batch_size=client_cfg.get('batch_size', 32), 
             learning_rate=client_cfg.get('learning_rate', 0.01), 
             momentum=client_cfg.get('momentum', 0.9), 
             # weight_decay=client_cfg.get('weight_decay', 0.0001) 
             # 不传入不支持的参数如'optimizer'和'shuffle' 
             )
        # 选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {device}")
        
        # 为每个轨道创建卫星
        for orbit in range(1, 7):  # 6个轨道
            for sat in range(1, 12):  # 每轨道11颗卫星
                sat_id = f"satellite_{orbit}-{sat}"
                
                # 创建客户端 - 使用适当的模型类型
                if self.config['data'].get('dataset') == 'real_traffic':
                    from fl_core.models.real_traffic_model import RealTrafficModel
                    
                    model_copy = RealTrafficModel(
                        input_dim=self.config['model']['feature_dim'],
                        hidden_dim=self.config['model'].get('hidden_dim', 64),
                        num_classes=self.config['model'].get('num_classes', 2)
                    )
                    
                    model_copy.load_state_dict(self.model.state_dict())
                    
                    client = FedProxClient(
                        sat_id,
                        model_copy,
                        client_config,
                        self.network_manager,
                        self.energy_model,
                        mu=self.mu,
                        device=device
                    )
                elif self.config['data'].get('dataset') == 'network_traffic':
                    from fl_core.models.traffic_model import SimpleTrafficModel
                    model_copy = SimpleTrafficModel(
                        input_dim=10,
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=2
                    )
                    model_copy.load_state_dict(self.model.state_dict())
                    client = FedProxClient(
                        sat_id,
                        model_copy,
                        client_config,
                        self.network_manager,
                        self.energy_model,
                        mu=self.mu,
                        device=device
                    )
                elif self.config['data'].get('dataset') == 'mnist':
                    from experiments.baseline_experiment import MNISTModel
                    model_copy = MNISTModel()
                    model_copy.load_state_dict(self.model.state_dict())
                    client = FedProxClient(
                        sat_id,
                        model_copy,
                        client_config,
                        self.network_manager,
                        self.energy_model,
                        mu=self.mu,
                        device=device
                    )
                else:
                    # 基础模型
                    from experiments.baseline_experiment import SimpleModel
                    base_model = SimpleModel(
                        input_dim=self.config['data'].get('feature_dim', 10),
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=self.config['data'].get('num_classes', 2)
                    )
                    base_model.load_state_dict(self.model.state_dict())
                    client = FedProxClient(
                        sat_id,
                        base_model,
                        client_config,
                        self.network_manager,
                        self.energy_model,
                        mu=self.mu,
                        device=device
                    )
                
                # 设置数据集
                if sat_id in self.satellite_datasets:
                    client.set_dataset(self.satellite_datasets[sat_id])
                else:
                    self.logger.warning(f"卫星 {sat_id} 没有对应的数据集")
                    if self.config['data'].get('dataset') == 'mnist':
                        # 为MNIST创建空数据集
                        from data_simulator.non_iid_generator import CustomMNISTDataset
                        client.set_dataset(CustomMNISTDataset([]))
                    else:
                        # 原有的空数据集
                        client.set_dataset(self.data_generator.generate_empty_dataset())
                
                self.clients[sat_id] = client