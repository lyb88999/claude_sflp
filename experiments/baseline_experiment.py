from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from fl_core.aggregation.global_aggregator import GlobalAggregator, GlobalConfig
from simulation.network_manager import NetworkManager
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import yaml
import logging
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from simulation.network_model import SatelliteNetwork
from simulation.comm_scheduler import CommunicationScheduler
from simulation.energy_model import EnergyModel
from simulation.topology_manager import TopologyManager
from data_simulator.non_iid_generator import CustomMNISTDataset, NonIIDGenerator, MNISTDataGenerator
from fl_core.client.satellite_client import SatelliteClient, ClientConfig
from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
from fl_core.aggregation.ground_station import GroundStationAggregator, GroundStationConfig
from visualization.visualization import Visualization

class SimpleModel(nn.Module):
    """简单的神经网络模型"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 2):
        super().__init__()
        self.__init__args__ = (input_dim,)
        self.__init__kwargs__ = {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes
        }
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 保存初始化参数
        self.__init__args__ = ()
        self.__init__kwargs__ = {}
        
        # 模型层定义
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.shape) == 3:
            # 如果输入是 [batch_size, height, width]
            x = x.unsqueeze(1)  # 添加channel维度 [batch_size, 1, height, width]
            
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class BaselineExperiment:
    def __init__(self, config_path: str = "configs/baseline_config.yaml"):
        """
        初始化基线实验
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._init_components()
        self.max_workers = 6

        # 添加全局聚合器
        global_config = GlobalConfig(
            min_ground_stations=2,
            consistency_threshold=0.8,
            max_version_diff=2,
            aggregation_timeout=1800.0,
            validation_required=True
        )
        self.global_aggregator = GlobalAggregator(global_config)
        
        # 为全局聚合器添加地面站
        for station_id in ['station_0', 'station_1', 'station_2']:
            self.global_aggregator.add_ground_station(station_id, 1.0)
        self.visualizer = Visualization()
        
    def _setup_logging(self):
        """设置日志"""
        # 获取实验类型名称
        experiment_type = self.__class__.__name__.lower().replace('experiment', '')
        
        # 创建顶级日志目录
        log_root_dir = Path("logs")
        log_root_dir.mkdir(exist_ok=True)
        
        # 创建实验类型目录
        experiment_dir = log_root_dir / experiment_type
        experiment_dir.mkdir(exist_ok=True)
        
        # 创建当前实验的时间戳目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_run_dir = experiment_dir / timestamp
        experiment_run_dir.mkdir(exist_ok=True)
        
        # 保存实验配置到日志目录
        with open(experiment_run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
        
        # 获取logger
        self.logger = logging.getLogger(f"{experiment_type}_{timestamp}")
        
        # 如果logger已经有处理器，说明已经配置过，直接返回
        if self.logger.handlers:
            return
            
        # 设置日志级别
        self.logger.setLevel(logging.DEBUG)
        
        # 1. 文件处理器 - 详细日志
        file_handler = logging.FileHandler(
            experiment_run_dir / "experiment.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # 2. 单独的错误日志文件
        error_handler = logging.FileHandler(
            experiment_run_dir / "errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        
        # 3. 控制台处理器 - 只显示简要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 为不同处理器设置不同的格式
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'  # 控制台输出简化
        )
        
        file_handler.setFormatter(file_formatter)
        error_handler.setFormatter(error_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
        # 保存实验目录路径
        self.log_dir = experiment_run_dir
        self.logger.info(f"日志保存在: {self.log_dir}")
        
    def _init_components(self):
        """初始化系统组件"""
        # 网络组件
        self.network_model = SatelliteNetwork(self.config['network']['tle_file'])
        self.comm_scheduler = CommunicationScheduler(self.network_model)
        self.energy_model = EnergyModel(self.network_model, self.config['energy']['config_file'])
        self.topology_manager = TopologyManager(
            self.network_model,
            self.comm_scheduler,
            self.energy_model
        )
        self.network_manager = NetworkManager(self.network_model, self.topology_manager)
        
        # 数据生成
        if self.config['data'].get('dataset') == 'real_traffic':
            from fl_core.models.real_traffic_model import RealTrafficModel
            from data_simulator.real_traffic_generator import RealTrafficGenerator
            self.data_generator = RealTrafficGenerator(
            num_satellites=self.config['fl']['num_satellites'],
            num_orbits=self.config['fl']['num_orbits'],
            satellites_per_orbit=self.config['fl']['satellites_per_orbit']
        )
            # 加载单个CSV文件（从配置中获取路径）
            csv_path = self.config['data'].get('csv_path', 'merged_traffic.csv')
            feature_dim, num_classes = self.data_generator.load_and_preprocess_data(csv_path)
            
            # 更新模型配置
            self.config['model']['feature_dim'] = feature_dim
            self.config['model']['num_classes'] = num_classes
        
        elif self.config['data'].get('dataset') == 'network_traffic':
            from data_simulator.network_traffic_generator import NetworkTrafficGenerator
            self.data_generator = NetworkTrafficGenerator(
                num_satellites=self.config['fl']['num_satellites'],
                num_orbits=self.config['fl']['num_orbits'],
                satellites_per_orbit=self.config['fl']['satellites_per_orbit']
            )
        elif self.config['data'].get('dataset') == 'mnist':
            self.data_generator = MNISTDataGenerator(
                num_satellites=self.config['fl']['num_satellites']
            )
        else:
            self.data_generator = NonIIDGenerator(
                num_satellites=self.config['fl']['num_satellites'],
                feature_dim=self.config['data']['feature_dim'],
                num_classes=self.config['data']['num_classes']
            )
        
        # 初始化卫星客户端
        self.clients = {}

        # 根据数据集类型选择模型
        dataset_type = self.config['data'].get('dataset')
        if dataset_type == 'real_traffic':
            self.model = RealTrafficModel(
                input_dim=feature_dim,
                hidden_dim=self.config['model'].get('hidden_dim', 64),
                num_classes=num_classes
            )
            self.logger.info(f"创建RealTrafficModel: 输入维度={feature_dim}, 类别数={num_classes}")
        elif dataset_type == 'mnist':
            self.model = MNISTModel()
        elif dataset_type == 'network_traffic':
            try:
                from fl_core.models.traffic_model import SimpleTrafficModel
                self.model = SimpleTrafficModel(
                    input_dim=10,  # 流量特征维度
                    hidden_dim=self.config['model'].get('hidden_dim', 20),
                    num_classes=2  # 二分类
                )
            except ImportError:
                self.logger.error("无法导入 SimpleTrafficModel，请确保已添加该文件")
                raise
        else:
            self.model = SimpleModel(
                input_dim=self.config['data'].get('feature_dim', 10),  # 提供默认值
                hidden_dim=self.config['model'].get('hidden_dim', 20),
                num_classes=self.config['data'].get('num_classes', 2)
            )
        
        # 初始化聚合器
        self.intra_orbit_aggregators = {}
        ground_station_config = GroundStationConfig(
            bandwidth_limit=1000.0,
            storage_limit=10000.0,
            priority_levels=3,
            batch_size=10,
            aggregation_interval=60.0,
            min_updates=self.config['aggregation']['min_updates'],
            max_staleness=self.config['aggregation']['max_staleness'],
            timeout=self.config['aggregation']['timeout'],
            weighted_average=self.config['aggregation']['weighted_average']
        )
        self.ground_station_aggregator = GroundStationAggregator(ground_station_config)

        # 初始化地面站
        self.ground_stations = {}
        self.setup_ground_stations()
        
    # def prepare_data(self):
    #     """准备训练数据"""
    #     # 生成训练数据，为66个卫星分配数据
    #     total_satellites = 66  # 6轨道 × 11卫星
    #     self.satellite_datasets = {}
        
    #     # 生成基础数据
    #     all_datasets = self.data_generator.generate_data(
    #         total_samples=self.config['data']['total_samples'],
    #         dirichlet_alpha=self.config['data']['dirichlet_alpha'],
    #         mean_samples_per_satellite=self.config['data']['mean_samples_per_satellite'],
    #         num_satellites=total_satellites  # 确保只生成66个数据集
    #     )
        
    #     # 为每个卫星分配数据集
    #     dataset_idx = 0
    #     for orbit_num in range(1, 7):  # 6个轨道
    #         for sat_num in range(1, 12):  # 每轨道11颗卫星
    #             sat_id = f"satellite_{orbit_num}-{sat_num}"
    #             if dataset_idx < len(all_datasets):
    #                 self.satellite_datasets[sat_id] = list(all_datasets.values())[dataset_idx]
    #                 dataset_idx += 1
        
    #     # 生成测试数据
    #     self.test_dataset = self.data_generator.generate_test_data(
    #         self.config['data']['test_samples']
    #     )
        
    #     # 记录数据分配情况
    #     for sat_id, dataset in self.satellite_datasets.items():
    #         self.logger.debug(f"卫星 {sat_id} 数据集大小: {len(dataset)}")
        
    #     self.logger.info(f"数据生成完成，共{len(self.satellite_datasets)}个卫星节点")

    def prepare_data(self):
        """准备训练数据"""
        if self.config['data'].get('dataset') == 'real_traffic':
            self.logger.info("开始准备真实网络流量数据")
            
            # 检查是否启用区域相似性
            region_similarity = self.config['data'].get('region_similarity', False)

            if region_similarity:
                self.logger.info("启用区域相似性")
                overlap_ratio = self.config['data'].get('overlap_ratio', 0.5)
                self.satellite_datasets = self.data_generator.generate_region_similar_data(
                iid=self.config['data'].get('iid', False),
                alpha=self.config['data'].get('alpha', 1.0),
                overlap_ratio=overlap_ratio)
            else:
                self.logger.info("未启用区域相似性")
                # 使用IID或非IID分布分配给卫星
                self.satellite_datasets = self.data_generator.generate_data(
                    iid=self.config['data'].get('iid', True),            # 是否使用IID分布
                    alpha=self.config['data'].get('alpha', 1.0)          # Dirichlet参数(仅在non-iid时使用)
                )
            
            # 获取测试数据集
            self.test_dataset = self.data_generator.generate_test_data()
            
            # 记录数据分配情况
            total_samples = 0
            benign_samples = 0
            malicious_samples = 0
            
            for sat_id, dataset in self.satellite_datasets.items():
                self.logger.debug(f"卫星 {sat_id} 数据集大小: {len(dataset)}")
                total_samples += len(dataset)
                
                # 统计每个卫星的良性/恶意流量比例
                labels = dataset.labels.numpy()
                sat_benign = np.sum(labels == 1)  # 假设1是benign
                sat_malicious = np.sum(labels == 0)  # 假设0是malicious
                
                benign_samples += sat_benign
                malicious_samples += sat_malicious
                
                # 输出每个卫星的标签分布
                benign_ratio = sat_benign / len(dataset) * 100 if len(dataset) > 0 else 0
                self.logger.debug(f"卫星 {sat_id} 标签分布: 良性={benign_ratio:.1f}%, "
                                f"恶意={(100-benign_ratio):.1f}%")
            
            # 输出整体统计信息
            self.logger.info(f"数据分配完成:")
            self.logger.info(f"  总样本数: {total_samples}")
            self.logger.info(f"  良性样本: {benign_samples} ({benign_samples/total_samples*100:.1f}%)")
            self.logger.info(f"  恶意样本: {malicious_samples} ({malicious_samples/total_samples*100:.1f}%)")
            self.logger.info(f"  卫星节点: {len(self.satellite_datasets)}")

        elif self.config['data'].get('dataset') == 'network_traffic':
            self.logger.info("开始准备网络流量数据")
            
            # 使用新的生成器创建数据
            self.satellite_datasets = self.data_generator.generate_data(
                total_samples=self.config['data']['total_samples'],
                malicious_ratio=self.config['data'].get('malicious_ratio', 0.3),
                orbit_similarity=self.config['data'].get('orbit_similarity', 0.7),
                position_similarity=self.config['data'].get('position_similarity', 0.8)
            )
            
            # 获取测试数据集
            self.test_dataset = self.data_generator.generate_test_data(
                self.config['data']['test_samples']
            )
            
            # 记录数据分配情况
            for sat_id, dataset in self.satellite_datasets.items():
                self.logger.debug(f"卫星 {sat_id} 数据集大小: {len(dataset)}")
            
            self.logger.info(f"数据生成完成，共{len(self.satellite_datasets)}个卫星节点")
        elif self.config['data'].get('dataset') == 'mnist':
            self.logger.info("开始准备MNIST数据")
            
            # 配置总卫星数
            total_satellites = 66  # 6轨道 × 11卫星
            self.satellite_datasets = {}
            
            # 创建数据生成器
            self.data_generator = MNISTDataGenerator(total_satellites)
            
            # 生成非独立同分布数据
            all_datasets = self.data_generator.generate_non_iid_data(
                dirichlet_alpha=self.config['data']['dirichlet_alpha'],
                mean_samples_per_satellite=self.config['data']['mean_samples_per_satellite']
            )
            
            # 为每个卫星分配数据集
            dataset_idx = 0
            for orbit_num in range(1, 7):  # 6个轨道
                for sat_num in range(1, 12):  # 每轨道11颗卫星
                    sat_id = f"satellite_{orbit_num}-{sat_num}"
                    if dataset_idx < len(all_datasets):
                        self.satellite_datasets[sat_id] = list(all_datasets.values())[dataset_idx]
                        dataset_idx += 1
            
            # 获取测试数据集
            self.test_dataset = self.data_generator.get_test_dataset()
            
            # 记录数据分配情况
            for sat_id, dataset in self.satellite_datasets.items():
                self.logger.debug(f"卫星 {sat_id} 数据集大小: {len(dataset)}")
            
            self.logger.info(f"数据生成完成，共{len(self.satellite_datasets)}个卫星节点")
        
    def setup_clients(self):
        """设置卫星客户端"""
        client_config = ClientConfig(**self.config['client'])
        

        
        # 为每个轨道创建卫星
        for orbit in range(1, 7):  # 6个轨道
            for sat in range(1, 12):  # 每轨道11颗卫星
                sat_id = f"satellite_{orbit}-{sat}"
                # 创建客户端
                if self.config['data'].get('dataset') == 'real_traffic':
                    from fl_core.models.real_traffic_model import RealTrafficModel
                    
                    # 使用与全局模型相同的参数创建新实例
                    model_copy = RealTrafficModel(
                        input_dim=self.config['model']['feature_dim'],
                        hidden_dim=self.config['model'].get('hidden_dim', 64),
                        num_classes=self.config['model'].get('num_classes', 2)
                    )
                    
                    # 加载全局模型参数
                    model_copy.load_state_dict(self.model.state_dict())
                    
                    client = SatelliteClient(
                        sat_id,
                        model_copy,
                        client_config,
                        self.network_manager,
                        self.energy_model
                    )
                elif self.config['data'].get('dataset') == 'network_traffic':
                    from fl_core.models.traffic_model import SimpleTrafficModel
                    model_copy = SimpleTrafficModel(
                        input_dim=10,
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=2
                    )
                    model_copy.load_state_dict(self.model.state_dict())  # 复制参数
                    client = SatelliteClient(
                        sat_id,
                        model_copy,
                        client_config,
                        self.network_manager,
                        self.energy_model
                    )
                elif self.config['data'].get('dataset') == 'mnist':
                    model_copy = MNISTModel()  # 创建新实例
                    model_copy.load_state_dict(self.model.state_dict())  # 复制参数
                    # 对于MNIST，使用预先创建的CNN模型
                    client = SatelliteClient(
                        sat_id,
                        model_copy,  # 创建新的CNN模型实例
                        client_config,
                        self.network_manager,
                        self.energy_model
                    )
                else:
                    # 基础模型
                    base_model = SimpleModel(
                        input_dim=self.config['data'].get('feature_dim', 10),
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=self.config['data'].get('num_classes', 2)
                    )
                    base_model.load_state_dict(self.model.state_dict())  # 复制参数
                    client = SatelliteClient(
                        sat_id,
                        base_model,
                        client_config,
                        self.network_manager,
                        self.energy_model
                    )
                
                # 设置数据集
                if sat_id in self.satellite_datasets:
                    client.set_dataset(self.satellite_datasets[sat_id])
                else:
                    self.logger.warning(f"卫星 {sat_id} 没有对应的数据集")
                    if self.config['data'].get('dataset') == 'mnist':
                        # 为MNIST创建空数据集
                        client.set_dataset(CustomMNISTDataset([]))
                    else:
                        # 原有的空数据集
                        client.set_dataset(self.data_generator.generate_empty_dataset())
                
                self.clients[sat_id] = client

    def setup_ground_stations(self):
        """初始化地面站"""
        for i in range(3):
            ground_station_config = GroundStationConfig(
                bandwidth_limit=1000.0,
                storage_limit=10000.0,
                priority_levels=3,
                batch_size=10,
                aggregation_interval=60.0,
                min_updates=self.config['aggregation']['min_updates'],
                max_staleness=self.config['aggregation']['max_staleness'],
                timeout=self.config['aggregation']['timeout'],
                weighted_average=self.config['aggregation']['weighted_average']
            )
            
            station = GroundStationAggregator(ground_station_config)
            station.responsible_orbits = [i*2, i*2+1]  # 每个地面站负责两个轨道
            
            # 为每个负责的轨道设置权重
            for orbit_id in station.responsible_orbits:
                for sat_num in range(1, 12):  # 每个轨道11颗卫星
                    sat_name = f"satellite_{orbit_id+1}-{sat_num}"
                    station.add_orbit(sat_name, 1.0)
                
            self.ground_stations[f"station_{i}"] = station
            self.logger.info(f"地面站 {i} 初始化完成: 负责轨道={station.responsible_orbits}")
        
    def train(self):
        # 初始化记录列表
        accuracies = []
        losses = []
        # 初始化统计信息
        energy_stats = {
            'training_energy': [],   # 每轮训练能耗
            'communication_energy': [],  # 每轮通信能耗
            'total_energy': []      # 每轮总能耗
        }
        satellite_stats = {
            'training_satellites': [],   # 每轮训练的卫星数
            'receiving_satellites': [],  # 每轮接收参数的卫星数
            'total_active': []          # 每轮总活跃卫星数
        }

        current_time = datetime.now().timestamp()
        self.current_round = 0
        best_accuracy = 0
        rounds_without_improvement = 0
        max_rounds_without_improvement = 3  # 连续3轮没有提升就停止
        min_rounds = 5  # 最少训练轮数
        accuracy_threshold = 95.0  # 提高准确率阈值到95%

        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"=== 开始第 {round_num + 1} 轮训练 === 时间：{datetime.fromtimestamp(current_time)}")
            round_energy = 0

            # 使用线程池并行处理每个地面站的轨道
            orbit_successes = 0
            # 收集所有轨道的统计信息
            round_training_energy = 0
            round_comm_energy = 0
            round_training_sats = set()
            round_receiving_sats = set()
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
                        for client in self.clients.values():
                            if client.train_stats:
                                round_energy += client.train_stats[-1]['summary']['energy_consumption']
                                round_loss += client.train_stats[-1]['summary']['train_loss'][-1]
                                round_stat[client.client_id] = client.train_stats[-1]
                                
                        # energies.append(round_energy)
                        losses.append(round_loss / len(self.clients))  # 平均损失
                        # round_stats.append(round_stat)

                        self.logger.info(f"第 {round_num + 1} 轮全局准确率: {accuracy:.4f}")
                        
                        # 更新最佳准确率和检查提升情况
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            rounds_without_improvement = 0
                            self.logger.info(f"找到更好的模型！新的最佳准确率: {accuracy:.4f}")
                        else:
                            rounds_without_improvement += 1
                            self.logger.info(f"准确率未提升，已经 {rounds_without_improvement} 轮没有改进")

                        # 检查是否满足停止条件
                        if round_num + 1 >= min_rounds:  # 已达到最小轮数
                            if accuracy >= accuracy_threshold:
                                self.logger.info(f"达到目标准确率 {accuracy:.4f}，停止训练")
                                break
                            elif rounds_without_improvement >= max_rounds_without_improvement:
                                self.logger.info(f"连续 {max_rounds_without_improvement} 轮准确率未提升，停止训练")
                                break

                        current_time += self.config['fl']['round_interval']
                        
                    else:
                        self.logger.warning("全局聚合失败")
                else:
                    self.logger.warning(f"只有 {len(station_results)}/{len(self.ground_stations)} 个地面站完成聚合，跳过全局聚合")
            else:
                self.logger.warning("所有轨道训练失败，跳过聚合阶段")

            current_time += self.config['fl']['round_interval']
            
        self.logger.info(f"\n=== 训练结束 ===")
        self.logger.info(f"总轮次: {round_num + 1}")
        self.logger.info(f"最佳准确率: {best_accuracy:.4f}")
        # self.plot_training_results(accuracies, energies)
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'energy_stats': {
                'training_energy': energy_stats['training_energy'],
                'communication_energy': energy_stats['communication_energy'],
                'total_energy': energy_stats['total_energy']
            },
            'satellite_stats': {
                'training_satellites': satellite_stats['training_satellites'],
                'receiving_satellites': satellite_stats['receiving_satellites'],
                'total_active': satellite_stats['total_active']
            }
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

    def collect_stats(self, accuracies, losses, energies, round_stats):
        """收集实验统计信息"""
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'energies': energies,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'total_energy': sum(energies) if energies else 0,
            'avg_round_energy': np.mean(energies) if energies else 0,
            'convergence_round': len(accuracies),
            'training_time': 0,
            'active_satellites_per_round': []
        }
        
        # 计算每轮参与训练的卫星数
        for round_stat in round_stats:
            active_sats = len(round_stat.keys())
            stats['active_satellites_per_round'].append(active_sats)
        
        # 计算训练时间
        for round_stat in round_stats:
            max_time = 0
            for sat_stats in round_stat.values():
                max_time = max(max_time, sat_stats['summary']['compute_time'])
            stats['training_time'] += max_time
        
        return stats

    def plot_training_results(self, accuracies: List[float], energies: List[float]):
        """绘制训练结果图表"""
        import matplotlib.pyplot as plt
        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 准确率变化（保持原样）
        rounds = range(1, len(accuracies) + 1)
        ax1.plot(rounds, accuracies, 'b-', marker='o')
        ax1.set_title('Accuracy over Training Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True)
        
        # 2. 每轨道的平均能耗
        orbit_energies = defaultdict(list)
        for sat_id, client in self.clients.items():
            orbit_num = int(sat_id.split('-')[0].split('_')[1])
            if client.train_stats:
                orbit_energies[orbit_num].append(client.train_stats[-1]['summary']['energy_consumption'])
        
        orbits = sorted(orbit_energies.keys())
        avg_energies = [np.mean(orbit_energies[orbit]) for orbit in orbits]
        ax2.bar(orbits, avg_energies)
        ax2.set_title('Average Energy Consumption per Orbit')
        ax2.set_xlabel('Orbit Number')
        ax2.set_ylabel('Average Energy (Wh)')
        
        # 3. 每轮每颗卫星的能耗分布（箱线图）
        round_satellite_energies = []
        for round_energy in energies:
            satellite_energies = []
            for client in self.clients.values():
                if client.train_stats and len(client.train_stats) > len(round_satellite_energies):
                    satellite_energies.append(client.train_stats[-1]['summary']['energy_consumption'])
            round_satellite_energies.append(satellite_energies)
        
        ax3.boxplot(round_satellite_energies)
        ax3.set_title('Energy Distribution per Round')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Energy Consumption (Wh)')
        
        # 4. 累计总能耗
        cumulative_energy = np.cumsum(energies)
        ax4.plot(rounds, cumulative_energy, 'r-', marker='o')
        ax4.set_title('Cumulative Energy Consumption')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Total Energy (Wh)')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()

    # def _distribute_orbit_params(self, coordinator: str, orbit_satellites: List[str], model_state: Dict, current_time: float):
    #     """
    #     在轨道内传递参数
    #     使用链式传递: 1->2->3->...->11
    #     """
    #     self.logger.info(f"开始轨道内参数传递, 协调者: {coordinator}")
    #     _, coord_num = self._parse_satellite_id(coordinator)
        
    #     # 按序号排序轨道内卫星
    #     sorted_satellites = sorted(orbit_satellites, 
    #                             key=lambda x: int(self._parse_satellite_id(x)[1]))
        
    #     # 从协调者开始，向后传递
    #     current_sat = coordinator
    #     for i in range(len(sorted_satellites)):
    #         next_idx = (sorted_satellites.index(current_sat) + 1) % len(sorted_satellites)
    #         next_sat = sorted_satellites[next_idx]
            
    #         self.logger.info(f"参数传递: {current_sat} -> {next_sat}")
    #         self.clients[next_sat].apply_model_update(model_state)
            
    #         current_sat = next_sat

    # def _distribute_orbit_params(self, coordinator: str, orbit_satellites: List[str], model_state: Dict, current_time: float):
    #     """
    #     在轨道内采用洪泛式传递参数
    #     Args:
    #         coordinator: 协调者卫星ID
    #         orbit_satellites: 轨道内所有卫星
    #         model_state: 模型参数
    #         current_time: 当前时间戳
    #     """
        
        
    #     # 按照序号排序卫星
    #     orbit_num, coord_num = self._parse_satellite_id(coordinator)
        
    #     orbit_prefix = f"[轨道 {orbit_num-1}]"  # 创建轨道前缀
    #     self.logger.info(f"{orbit_prefix} 开始轨道内洪泛式参数传递, 协调者: {coordinator}")
    #     sorted_satellites = sorted(orbit_satellites, 
    #                             key=lambda x: int(self._parse_satellite_id(x)[1]))
        
    #     # 记录已接收参数的卫星
    #     received_params = {coordinator}
    #     # 记录每个卫星的传播时间
    #     distribution_times = {coordinator: current_time}
        
    #     # 从协调者开始，向两个方向传播
    #     max_retries = 3
    #     transmission_interval = 60  # 传输间隔（秒）
        
    #     def propagate_direction(start_idx: int, direction: int):
    #         """向指定方向传播参数"""
    #         current_idx = start_idx
    #         retries = 0
    #         local_time = current_time
            
    #         while retries < max_retries:
    #             next_idx = (current_idx + direction) % len(sorted_satellites)
    #             current_sat = sorted_satellites[current_idx]
    #             next_sat = sorted_satellites[next_idx]
                
    #             # 如果下一个卫星已经收到参数，停止在这个方向的传播
    #             if next_sat in received_params:
    #                 break
                    
    #             try:
    #                 # 等待直到相邻卫星可见
    #                 wait_time = 0
    #                 max_wait = 120
    #                 while not self.network_model._check_visibility(current_sat, next_sat, local_time + wait_time):
    #                     if wait_time >= max_wait:
    #                         break
    #                     wait_time += 10
                    
    #                 if wait_time < max_wait:
    #                     # 传递参数
    #                     self.clients[next_sat].apply_model_update(model_state)
    #                     received_params.add(next_sat)
    #                     distribution_times[next_sat] = local_time + wait_time
    #                     self.logger.info(f"参数传递链: {current_sat} -> {next_sat} 成功")
                        
    #                     # 更新索引和时间
    #                     current_idx = next_idx
    #                     local_time += wait_time + transmission_interval
    #                     retries = 0  # 重置重试次数
    #                 else:
    #                     retries += 1
    #                     local_time += transmission_interval
    #                     self.logger.warning(f"尝试传递参数 {current_sat}->{next_sat} 失败，重试 {retries}/{max_retries}")
                
    #             except Exception as e:
    #                 self.logger.error(f"参数传递出错 {current_sat}->{next_sat}: {str(e)}")
    #                 retries += 1
    #                 local_time += transmission_interval

            
        
    #     # 获取协调者在排序列表中的索引
    #     coord_idx = sorted_satellites.index(coordinator)
        
    #     # 向两个方向传播
    #     propagate_direction(coord_idx, 1)  # 向后传播
    #     propagate_direction(coord_idx, -1)  # 向前传播
        
    #     # 检查传播结果
    #     missing_satellites = set(orbit_satellites) - received_params
    #     if missing_satellites:
    #         self.logger.warning(f"{orbit_prefix} 未收到参数的卫星: {missing_satellites}")
    #     else:
    #         self.logger.info(f"{orbit_prefix} 所有卫星已成功接收参数")
            
    #     # 返回最后一个卫星的传播时间
    #     return max(distribution_times.values()) if distribution_times else current_time

    def _distribute_orbit_params(self, coordinator: str, orbit_satellites: List[str], model_state: Dict, current_time: float):
        """
        在轨道内采用洪泛式传递参数 - 符合FedAvg算法
        Args:
            coordinator: 协调者卫星ID
            orbit_satellites: 轨道内所有卫星
            model_state: 模型参数
            current_time: 当前时间戳
        """
        # 按照序号排序卫星
        orbit_num, coord_num = self._parse_satellite_id(coordinator)
        
        orbit_prefix = f"[轨道 {orbit_num-1}]"  # 创建轨道前缀
        self.logger.info(f"{orbit_prefix} 轮次{self.current_round}: 开始轨道内洪泛式参数传递, 协调者: {coordinator}")
        
        # 检查初始参数示例值
        param_example = list(model_state.values())[0][0][0].item()
        self.logger.info(f"{orbit_prefix} 全局参数示例值: {param_example:.4f}")
        
        # 记录几个卫星在接收全局参数前的本地模型参数
        for sat_id in orbit_satellites[:2]:  # 只检查前两个卫星
            if sat_id in self.clients:
                try:
                    sat_param = list(self.clients[sat_id].model.state_dict().values())[0][0][0].item()
                    self.logger.info(f"参数分发前 {sat_id} 当前参数示例: {sat_param:.4f}")
                except Exception as e:
                    self.logger.error(f"获取卫星参数出错: {str(e)}")
        
        sorted_satellites = sorted(orbit_satellites, 
                                key=lambda x: int(self._parse_satellite_id(x)[1]))
        
        # 记录已接收参数的卫星
        received_params = {coordinator}
        # 记录每个卫星的传播时间
        distribution_times = {coordinator: current_time}
        
        # 从协调者开始，向两个方向传播
        max_retries = 3
        transmission_interval = 60  # 传输间隔（秒）
        
        def propagate_direction(start_idx: int, direction: int):
            """向指定方向传播参数"""
            current_idx = start_idx
            retries = 0
            local_time = current_time
            
            while retries < max_retries:
                next_idx = (current_idx + direction) % len(sorted_satellites)
                current_sat = sorted_satellites[current_idx]
                next_sat = sorted_satellites[next_idx]
                
                # 如果下一个卫星已经收到参数，停止在这个方向的传播
                if next_sat in received_params:
                    break
                    
                try:
                    # 等待直到相邻卫星可见
                    wait_time = 0
                    max_wait = 120
                    while not self.network_model._check_visibility(current_sat, next_sat, local_time + wait_time):
                        if wait_time >= max_wait:
                            break
                        wait_time += 10
                    
                    if wait_time < max_wait:
                        # 传递参数前记录参数
                        if next_sat in self.clients:
                            pre_param = list(self.clients[next_sat].model.state_dict().values())[0][0][0].item()
                        
                        # 传递参数
                        self.clients[next_sat].apply_model_update(model_state)
                        received_params.add(next_sat)
                        distribution_times[next_sat] = local_time + wait_time
                        
                        # 验证参数是否真的改变了
                        if next_sat in self.clients:
                            post_param = list(self.clients[next_sat].model.state_dict().values())[0][0][0].item()
                            self.logger.info(f"参数传递: {current_sat} -> {next_sat} 成功. 参数变化: {pre_param:.4f} -> {post_param:.4f}")
                        
                        # 更新索引和时间
                        current_idx = next_idx
                        local_time += wait_time + transmission_interval
                        retries = 0  # 重置重试次数
                    else:
                        retries += 1
                        local_time += transmission_interval
                        self.logger.warning(f"尝试传递参数 {current_sat}->{next_sat} 失败，重试 {retries}/{max_retries}")
                
                except Exception as e:
                    self.logger.error(f"参数传递出错 {current_sat}->{next_sat}: {str(e)}")
                    retries += 1
                    local_time += transmission_interval
        
        # 获取协调者在排序列表中的索引
        coord_idx = sorted_satellites.index(coordinator)
        
        # 向两个方向传播
        propagate_direction(coord_idx, 1)  # 向后传播
        propagate_direction(coord_idx, -1)  # 向前传播
        
        # 检查传播结果
        missing_satellites = set(orbit_satellites) - received_params
        if missing_satellites:
            self.logger.warning(f"{orbit_prefix} 未收到参数的卫星: {missing_satellites}")
        else:
            self.logger.info(f"{orbit_prefix} 所有卫星已成功接收参数")
            
        # 检查卫星的训练配置，确保优化器和学习率正确设置
        for sat_id in orbit_satellites[:2]:  # 只检查前两个卫星
            if sat_id in self.clients:
                client = self.clients[sat_id]
                if hasattr(client, 'optimizer') and client.optimizer is not None:
                    opt_params = next(iter(client.optimizer.param_groups))
                    self.logger.info(f"卫星 {sat_id} 优化器学习率: {opt_params['lr']}")
        
        # 返回最后一个卫星的传播时间
        return max(distribution_times.values()) if distribution_times else current_time

    def _handle_orbit_training(self, station_id: str, orbit_id: int, current_time: float):
        """
        处理单个轨道的训练过程
        Args:
            station_id: 地面站ID
            orbit_id: 轨道ID
            current_time: 当前时间戳
        Returns:
            bool: 训练过程是否成功完成
        """
        try:
            orbit_prefix = f"[轨道 {orbit_id}]"
            station = self.ground_stations[station_id]
            self.logger.info(f"{orbit_prefix} 开始处理")

            # 记录本轮轨道的统计信息
            orbit_stats = {
                'training_energy': 0,  # 训练能耗
                'communication_energy': 0,  # 通信能耗
                'training_satellites': set(),  # 训练的卫星
                'receiving_satellites': set()  # 接收参数的卫星
            }
            trained_satellites = set()
            
            # 1. 等待并选择可见卫星作为协调者
            coordinator = None
            orbit_satellites = self._get_orbit_satellites(orbit_id)
            max_wait_time = current_time + self.config['fl']['round_interval']
            
            while not coordinator and current_time < max_wait_time:
                for sat_id in orbit_satellites:
                    if self.network_model._check_visibility(station_id, sat_id, current_time):
                        coordinator = sat_id
                        break
                if not coordinator:
                    self.logger.info(f"轨道 {orbit_id} 当前无可见卫星，等待60秒...")
                    current_time += 60
                    self.topology_manager.update_topology(current_time)

            if not coordinator:
                self.logger.warning(f"轨道 {orbit_id} 在指定时间内未找到可见卫星")
                return False

            self.logger.info(f"轨道 {orbit_id} 选择 {coordinator} 作为协调者")

            # 2.3. 分发初始参数给协调者
            model_state = self.model.state_dict()
            self.logger.info(f"\n=== 轨道 {orbit_id} 内参数分发 ===")
            pre_comm_energy = self.energy_model.get_battery_level(coordinator)
            current_time = self._distribute_orbit_params(coordinator, orbit_satellites, model_state, current_time)
            post_comm_energy = self.energy_model.get_battery_level(coordinator)
            # 如果参数分发失败，提前返回
            if not current_time:  # 假设_distribute_orbit_params在失败时返回None
                self.logger.error(f"轨道 {orbit_id} 参数分发失败")
                return False
            orbit_stats['communication_energy'] += (pre_comm_energy - post_comm_energy)
            orbit_stats['receiving_satellites'].update(orbit_satellites)


            # 4. 轨道内训练
            self.logger.info(f"=== 轨道 {orbit_id} 训练 ===")
            trained_satellites = set()
            training_stats = {}  # 记录每个卫星的训练状态

            for sat_id in orbit_satellites:
                model_state = self.clients[sat_id].model.state_dict()
                self.logger.info(f"卫星 {sat_id} 参数示例: {list(model_state.values())[0][0][0].item():.4f}")
                pre_train_energy = self.energy_model.get_battery_level(sat_id)
                stats = self.clients[sat_id].train(self.current_round)
                if stats['summary']['train_loss']:
                    post_train_energy = self.energy_model.get_battery_level(sat_id)
                    orbit_stats['training_energy'] += (pre_train_energy - post_train_energy)
                    orbit_stats['training_satellites'].add(sat_id)
                    trained_satellites.add(sat_id)
                    training_stats[sat_id] = stats
                    self.logger.info(f"轨道 {orbit_id} - {sat_id} 完成训练: "  # 添加轨道ID到训练日志
                            f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                            f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                            f"能耗={stats['summary']['energy_consumption']:.4f}Wh")
                else:
                    self.logger.warning(f"卫星 {sat_id} 训练未产生有效结果")

            # # 添加如下检查
            # self.logger.info(f"=== 轨道 {orbit_id} 训练后、聚合前参数检查 ===")
            # model_params = {}
            # for sat_id in trained_satellites:
            #     try:
            #         # 保存第一个参数的前几个值作为指纹
            #         param = next(iter(self.clients[sat_id].model.parameters()))
            #         model_params[sat_id] = param[:3, :3].detach().numpy().flatten()
            #         self.logger.info(f"卫星 {sat_id} 参数指纹: {model_params[sat_id]}")
            #     except Exception as e:
            #         self.logger.error(f"获取卫星 {sat_id} 参数时出错: {str(e)}")

            # # 比较参数是否相同
            # if len(model_params) >= 2:
            #     param_ids = list(model_params.keys())
            #     same_params = True
            #     for i in range(1, len(param_ids)):
            #         diff = np.abs(model_params[param_ids[0]] - model_params[param_ids[i]]).mean()
            #         if diff > 1e-6:
            #             same_params = False
            #             self.logger.info(f"卫星 {param_ids[0]} 和 {param_ids[i]} 参数不同，差异: {diff:.8f}")
                
            #     if same_params:
            #         self.logger.warning("警告: 所有卫星训练后参数仍然相同!")


            # 创建一个字典存储轨道训练后的模型
            if not hasattr(self, 'orbit_trained_models'):
                self.orbit_trained_models = {}
            
            # 在轨道内聚合前，收集训练后的模型状态
            if trained_satellites:
                # 只选一个卫星的模型来代表整个轨道
                representative_sat = list(trained_satellites)[0]
                self.logger.info(f"使用 {representative_sat} 作为轨道 {orbit_id} 的代表模型")
                
                # 保存该卫星的模型状态
                model_state = {}
                for name, param in self.clients[representative_sat].model.named_parameters():
                    model_state[name] = param.data.clone()
                
                # 存储模型状态
                self.orbit_trained_models[orbit_id] = model_state
            # 5. 轨道内聚合
            min_updates_required = self.config['aggregation']['min_updates']
            self.logger.info(f"需要至少 {min_updates_required} 个卫星更新，当前有 {len(trained_satellites)} 个")

            if len(trained_satellites) >= min_updates_required:
                self.logger.info(f"=== 轨道 {orbit_id} 聚合 ===")
                aggregator = self.intra_orbit_aggregators.get(orbit_id)
                if not aggregator:
                    aggregator = IntraOrbitAggregator(AggregationConfig(**self.config['aggregation']))
                    self.intra_orbit_aggregators[orbit_id] = aggregator

                # 收集更新并聚合
                updates_collected = 0
                for sat_id in trained_satellites:
                    try:
                        model_diff, stats = self.clients[sat_id].get_model_update()
                        if model_diff:
                            self.logger.info(f"收集卫星 {sat_id} 的模型更新")
                            aggregator.receive_update(sat_id, self.current_round, model_diff, current_time)
                            updates_collected += 1
                        else:
                            self.logger.warning(f"卫星 {sat_id} 的模型更新为空")
                    except Exception as e:
                        self.logger.error(f"收集卫星 {sat_id} 更新时出错: {str(e)}")

                self.logger.info(f"成功收集了 {updates_collected} 个卫星的更新")

                orbit_update = aggregator.get_aggregated_update(self.current_round)
                if orbit_update:
                    self.logger.info(f"轨道 {orbit_id} 完成聚合")
                     # 更新所有卫星的模型
                    update_success = 0
                    for sat_id in orbit_satellites:
                        try:
                            self.clients[sat_id].apply_model_update(orbit_update)
                            update_success += 1
                            self.logger.info(f"更新卫星 {sat_id} 的模型参数")
                        except Exception as e:
                            self.logger.error(f"更新卫星 {sat_id} 模型时出错: {str(e)}")

                    self.logger.info(f"成功更新了 {update_success} 个卫星的模型")

                    # 6. 验证并等待可见性窗口
                    visibility_start = current_time
                    best_visibility_time = None
                    max_search_time = 300  # 5分钟搜索窗口

                    # 先搜索一个最佳的可见性时间点
                    for check_time in range(int(visibility_start), int(visibility_start + max_search_time), 30):
                        if self.network_model._check_visibility(station_id, coordinator, check_time):
                            best_visibility_time = check_time
                            break

                    if best_visibility_time is not None:
                        current_time = best_visibility_time
                        self.topology_manager.update_topology(current_time)

                    # 7. 发送模型到地面站
                    try:
                        model_diff, _ = self.clients[coordinator].get_model_update()
                        if model_diff:
                            success = station.receive_orbit_update(
                                str(orbit_id),
                                self.current_round,
                                model_diff,
                                len(trained_satellites)
                            )
                            if success:
                                self.logger.info(f"轨道 {orbit_id} 的模型成功发送给地面站 {station_id}")
                                return True, orbit_stats
                    except Exception as e:
                        self.logger.error(f"发送模型到地面站时出错: {str(e)}")
                else:
                    self.logger.error(f"轨道 {orbit_id} 聚合失败: 无法获取有效的聚合结果")
            else:
                self.logger.warning(f"轨道 {orbit_id} 训练的卫星数量不足: {len(trained_satellites)} < {min_updates_required}")

            return False, orbit_stats

        except Exception as e:
            self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, orbit_stats
        
    def _station_aggregation(self, station_id: str, station) -> bool:
        """
        执行地面站聚合
        Args:
            station_id: 地面站ID
            station: 地面站实例
        Returns:
            bool: 聚合是否成功
        """
        try:
            self.logger.info(f"\n=== 地面站 {station_id} 聚合开始 ===")
            # 检查负责的轨道
            responsible_orbits = station.responsible_orbits
            self.logger.info(f"地面站 {station_id} 负责轨道: {responsible_orbits}")
            
            # 检查收到的更新
            updates = station.pending_updates.get(self.current_round, {})
            self.logger.info(f"收到的轨道更新: {list(updates.keys())}")

            # 获取聚合结果
            aggregated_update = station.get_aggregated_update(self.current_round)
            if aggregated_update:
                self.logger.info(f"地面站 {station_id} 完成聚合")
                # 修改这里：使用 receive_station_update 而不是 receive_orbit_update
                # 将聚合结果传递给全局聚合器
                success = self.global_aggregator.receive_station_update(  # 使用 global_aggregator 而不是 ground_station_aggregator
                    station_id,
                    self.current_round,
                    aggregated_update,
                    {'num_orbits': len(updates)},
                    self.current_round
                )
                
                if success:
                    self.logger.info(f"地面站 {station_id} 的聚合结果已成功发送到全局聚合器")
                    return True
                else:
                    self.logger.error(f"全局聚合器拒绝了地面站 {station_id} 的更新")
                    return False
            else:
                self.logger.warning(f"地面站 {station_id} 聚合失败，可能是因为没有收到足够的更新")
                return False
        except Exception as e:
            self.logger.error(f"地面站 {station_id} 聚合出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

            
    def _get_orbit_satellites(self, orbit_id: int) -> List[str]:
        """获取轨道内的所有卫星"""
        satellites = []
        orbit_num = orbit_id + 1  # 轨道编号从1开始
        # 每个轨道11颗卫星
        for i in range(1, 12):
            sat_name = f"satellite_{orbit_num}-{i}"  # 使用 satellite_X-X 格式
            if sat_name in self.clients:
                satellites.append(sat_name)
        return satellites
    
    def _parse_satellite_id(self, sat_id: str) -> Tuple[int, int]:
        """解析卫星ID获取轨道号和卫星序号"""
        # satellite_1-1 -> (1, 1)
        orbit_num, sat_num = sat_id.split('_')[1].split('-')
        return int(orbit_num), int(sat_num)

    def _get_orbit_number(self, satellite_id: str) -> int:
        """从卫星ID获取轨道编号"""
        return int(satellite_id.split('_')[1]) - 1
    
    def _select_orbit_coordinator(self, orbit_id: int, station_id: str, current_time: float) -> Optional[str]:
        """
        选择轨道协调者
        根据与地面站的可见性选择协调者
        """
        orbit_satellites = self._get_orbit_satellites(orbit_id)
        if not orbit_satellites:
            self.logger.warning(f"轨道 {orbit_id} 没有找到卫星")
            return None
            
        # 检查卫星与地面站的可见性
        visible_satellites = []
        for sat_id in orbit_satellites:
            is_visible = self.network_model._check_visibility(station_id, sat_id, current_time)
            self.logger.debug(f"检查可见性: {station_id} -> {sat_id}: {is_visible}")
            if is_visible:
                visible_satellites.append(sat_id)
                
        if visible_satellites:
            selected = visible_satellites[0]
            self.logger.info(f"轨道 {orbit_id} 选择 {selected} 作为协调者")
            return selected
            
        self.logger.warning(f"轨道 {orbit_id} 没有可见卫星可作为协调者")
        return None

    def _perform_intra_orbit_aggregation(self, round_num: int):
        """执行轨道内聚合"""
        # 根据拓扑分组进行聚合
        for group_id, group in self.topology_manager.groups.items():
            if group_id not in self.intra_orbit_aggregators:
                self.intra_orbit_aggregators[group_id] = IntraOrbitAggregator(
                    AggregationConfig(**self.config['aggregation'])
                )
                
            aggregator = self.intra_orbit_aggregators[group_id]
            
            # 收集组内更新
            for client_id in group.members:
                if client_id in self.clients:
                    model_diff, stats = self.clients[client_id].get_model_update()
                    aggregator.receive_update(
                        client_id,
                        round_num,
                        model_diff,
                        datetime.now().timestamp()
                    )
                    
            # 获取聚合结果
            aggregated_update = aggregator.get_aggregated_update(round_num)
            if aggregated_update:
                self.logger.info(f"轨道组{group_id}完成聚合")
                
    def _perform_ground_station_aggregation(self, round_num: int):
        """执行地面站聚合"""
        # 获取每个轨道组的聚合结果
        for group_id, aggregator in self.intra_orbit_aggregators.items():
            result = aggregator.get_aggregated_update(round_num)
            if result:
                self.ground_station_aggregator.receive_orbit_update(
                    group_id,
                    round_num,
                    result,
                    len(self.topology_manager.groups[group_id].members)
                )
                
        # 广播全局更新
            global_update = self.ground_station_aggregator.get_aggregated_update(round_num)
            if global_update:
                self.logger.info("完成全局聚合，更新所有客户端")
                for client in self.clients.values():
                    client.apply_model_update(global_update)
                    
    # def _perform_global_aggregation(self, round_num: int) -> bool:
    #     """
    #     执行全局聚合
    #     Args:
    #         round_num: 当前轮次
    #     Returns:
    #         bool: 聚合是否成功
    #     """
    #     try:
    #         self.logger.info("\n=== 全局聚合阶段 ===")
            
    #         # 检查每个地面站的状态
    #         station_updates = {}
    #         for station_id, station in self.ground_stations.items():
    #             updates = station.pending_updates.get(round_num, {})
    #             self.logger.info(f"地面站 {station_id} 收到的更新数: {len(updates)}")
    #             if updates:
    #                 station_updates[station_id] = len(updates)
            
    #         if not station_updates:
    #             self.logger.warning("没有地面站提供更新")
    #             return False
                
    #         self.logger.info(f"参与全局聚合的地面站数量: {len(station_updates)}")
            
    #         # 获取全局聚合结果
    #         global_update = self.global_aggregator.get_aggregated_update(round_num)
            
    #         if global_update:
    #             self.logger.info(f"完成第 {round_num + 1} 轮全局聚合")
                
    #             # 验证全局更新是否有效
    #             for name, param in global_update.items():
    #                 if torch.isnan(param).any() or torch.isinf(param).any():
    #                     self.logger.error(f"全局更新参数 {name} 包含无效值")
    #                     return False
                
    #             # 更新所有卫星的模型
    #             update_success = 0
    #             for client in self.clients.values():
    #                 try:
    #                     client.apply_model_update(global_update)
    #                     update_success += 1
    #                 except Exception as e:
    #                     self.logger.error(f"更新客户端 {client.client_id} 时出错: {str(e)}")
                
    #             self.logger.info(f"成功更新了 {update_success}/{len(self.clients)} 个卫星的模型")
                
    #             # 评估全局模型
    #             accuracy = self.evaluate()
    #             self.logger.info(f"全局聚合后测试准确率: {accuracy:.4f}")
    #             return True
    #         else:
    #             self.logger.warning(f"全局聚合失败: 无法获取有效的聚合结果")
    #             # 输出每个地面站的聚合状态
    #             for station_id, station in self.ground_stations.items():
    #                 updates = station.pending_updates.get(round_num, {})
    #                 self.logger.info(f"地面站 {station_id}: {len(updates)} 个待处理更新")
    #             return False
                
    #     except Exception as e:
    #         self.logger.error(f"全局聚合出错: {str(e)}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
    #         return False

    def _perform_global_aggregation(self, round_num: int) -> bool:
        try:
            orbit_accuracies = self.evaluate_orbit_models()
            self.logger.info("\n=== 全局聚合阶段 ===")
            global_update = self.global_aggregator.get_aggregated_update(round_num)
            
            if global_update:
                self.logger.info(f"完成第 {round_num + 1} 轮全局聚合")
                
                # 检查全局更新的参数
                for name, param in global_update.items():
                    if torch.isnan(param).any():
                        self.logger.error(f"Parameter {name} contains NaN values")
                        return False
                    self.logger.debug(f"Parameter {name} mean: {param.mean():.4f}")
                
                # 关键修改：保留批量归一化层的统计数据
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        global_update[name] = param
                
                # 更新所有卫星的模型
                update_success = 0
                for client in self.clients.values():
                    try:
                        client.apply_model_update(global_update)
                        update_success += 1
                    except Exception as e:
                        self.logger.error(f"更新客户端 {client.client_id} 时出错: {str(e)}")
                for client_id, client in self.clients.items():
                    if client_id.startswith("satellite_1-"):  # 只检查第一轨道的几个卫星
                        if client_id in ["satellite_1-1", "satellite_1-5", "satellite_1-11"]:  # 选择几个关键位置
                            self.logger.info(f"更新后 {client_id} 参数示例: {list(client.model.state_dict().values())[0][0][0].item():.4f}")
                
                # 更新评估用的模型
                self.model.load_state_dict(global_update)
                
                self.logger.info(f"成功更新了 {update_success}/{len(self.clients)} 个卫星的模型")
                
                # 评估全局模型
                global_accuracy = self.evaluate()
                self.logger.info(f"第 {round_num + 1} 轮全局准确率: {global_accuracy:.4f}")
                # 对比轨道模型和全局模型
                self.logger.info("\n=== 轨道模型 vs 全局模型性能对比 ===")
                for orbit_id, accuracy in orbit_accuracies.items():
                    diff = accuracy - global_accuracy
                    self.logger.info(f"轨道 {orbit_id}: {accuracy:.2f}% (与全局差异: {diff:+.2f}%)")
                return True
            else:
                self.logger.warning("全局聚合失败")
                return False
                
        except Exception as e:
            self.logger.error(f"全局聚合出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        

    # def evaluate(self) -> float:
    #     """评估全局模型性能"""
    #     self.model.eval()
    #     correct = 0
    #     total = 0
        
    #     # 使用第一个客户端的模型进行评估
    #     test_model = next(iter(self.clients.values())).model
        
    #     with torch.no_grad():
    #         for features, labels in torch.utils.data.DataLoader(
    #             self.test_dataset,
    #             batch_size=100
    #         ):
    #             outputs = test_model(features)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
                
    #     accuracy = correct / total
    #     return accuracy

    # def evaluate(self) -> float:
    #     """评估全局模型性能"""
    #     self.model.eval()  # 确保模型在评估模式
    #     test_loss = 0
    #     correct = 0
        
    #     test_loader = DataLoader(
    #         self.test_dataset, 
    #         batch_size=1000,
    #         shuffle=False
    #     )
        
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             output = self.model(data)
                
    #             # 使用交叉熵损失（确保模型输出是原始分数）
    #             test_loss += F.cross_entropy(output, target, reduction='sum').item()
                
    #             # 或者如果模型已经输出log-softmax，则使用nll_loss
    #             # test_loss += F.nll_loss(output, target, reduction='sum').item()
                
    #             pred = output.argmax(dim=1)  # 获取预测结果
    #             correct += pred.eq(target).sum().item()
        
    #     test_loss /= len(test_loader.dataset)
    #     accuracy = 100. * correct / len(test_loader.dataset)
        
    #     self.logger.info(
    #         f'测试结果:'
    #         f'\n总样本数: {len(test_loader.dataset)}'
    #         f'\n正确预测数: {correct}'
    #         f'\n平均损失: {test_loss:.4f}'
    #         f'\n准确率: {accuracy:.2f}%'
    #     )
        
    #     return accuracy

    def evaluate(self) -> float:
        """评估全局模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1000,
            shuffle=False
        )
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        self.logger.info(
            f'测试结果:'
            f'\n总样本数: {len(test_loader.dataset)}'
            f'\n正确预测数: {correct}'
            f'\n平均损失: {test_loss:.4f}'
            f'\n准确率: {accuracy:.2f}%'
        )
        
        return accuracy

    # 添加新的辅助方法来获取轨道模型和评估模型
    def _get_orbit_model(self, orbit: int):
        """获取指定轨道的模型"""
        try:
            # 选择轨道中的最后一个卫星(应该包含了轨道内聚合的结果)
            sat_id = f"satellite_{orbit}-{self.config['fl']['satellites_per_orbit']}"
            if sat_id in self.clients:
                # 创建一个新模型并加载参数
                if self.config['data'].get('dataset') == 'network_traffic':
                    from data_simulator.network_traffic_generator import SimpleTrafficModel
                    orbit_model = SimpleTrafficModel(
                        input_dim=10,
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=2
                    )
                else:
                    # 其他模型类型
                    orbit_model = type(self.model)(*self.model.__init__args__, **self.model.__init__kwargs__)
                
                # 加载卫星的模型参数
                orbit_model.load_state_dict(self.clients[sat_id].model.state_dict())
                return orbit_model
            return None
        except Exception as e:
            self.logger.error(f"获取轨道 {orbit} 模型时出错: {str(e)}")
            return None

    def _evaluate_model(self, model, dataset) -> float:
        """评估特定模型在数据集上的准确率"""
        model.eval()
        correct = 0
        total = 0
        
        test_loader = DataLoader(
            dataset, 
            batch_size=1000,
            shuffle=False
        )
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        return 100. * correct / total
        
    def run(self):
        """运行基准实验"""
        self.logger.info("开始基线实验")
        
        # 准备数据
        self.prepare_data()
        
        # 设置客户端
        self.setup_clients()
        
        # 执行训练并获取统计信息
        stats = self.train()
        
        self.logger.info("实验完成")
        
        # 返回统计信息供后续比较
        return stats
        
    def evaluate_orbit_models(self) -> Dict[int, float]:
        """评估每个轨道的训练模型性能"""
        self.logger.info("\n=== 各轨道训练后模型单独评估 ===")
        
        results = {}
        
        if not hasattr(self, 'orbit_trained_models') or not self.orbit_trained_models:
            self.logger.warning("没有找到保存的轨道模型")
            return results
        
        for orbit_id, model_state in self.orbit_trained_models.items():
            try:
                # 创建一个新模型实例
                if self.config['data'].get('dataset') == 'network_traffic':
                    from data_simulator.network_traffic_generator import SimpleTrafficModel
                    orbit_model = SimpleTrafficModel(
                        input_dim=10,
                        hidden_dim=self.config['model'].get('hidden_dim', 20),
                        num_classes=2
                    )
                else:
                    # 其他模型类型
                    orbit_model = type(self.model)(*self.model.__init__args__, **self.model.__init__kwargs__)
                
                 # 关键修改：先获取完整的模型状态以保留批量归一化层的统计数据
                complete_state_dict = orbit_model.state_dict()
                
                # 合并保存的参数与初始状态字典
                for name, param in model_state.items():
                    if name in complete_state_dict:
                        complete_state_dict[name] = param
                # 加载保存的参数
                orbit_model.load_state_dict(complete_state_dict)
                
                # 评估模型
                orbit_model.eval()
                correct = 0
                total = 0
                
                test_loader = DataLoader(
                    self.test_dataset, 
                    batch_size=1000,
                    shuffle=False
                )
                
                with torch.no_grad():
                    for data, target in test_loader:
                        output = orbit_model(data)
                        pred = output.argmax(dim=1)
                        total += target.size(0)
                        correct += pred.eq(target).sum().item()
                
                accuracy = 100. * correct / total
                results[orbit_id] = accuracy
                self.logger.info(f"轨道 {orbit_id} 训练后模型准确率: {accuracy:.2f}%")
                
            except Exception as e:
                self.logger.error(f"评估轨道 {orbit_id} 模型时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results


def main():
    # 运行基线实验
    experiment = BaselineExperiment()

    experiment.run()
    
if __name__ == "__main__":
    main()