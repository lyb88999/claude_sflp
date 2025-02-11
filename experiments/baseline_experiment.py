from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from simulation.network_manager import NetworkManager
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import yaml
import logging
from pathlib import Path
import numpy as np

from simulation.network_model import SatelliteNetwork
from simulation.comm_scheduler import CommunicationScheduler
from simulation.energy_model import EnergyModel
from simulation.topology_manager import TopologyManager
from data_simulator.non_iid_generator import NonIIDGenerator
from fl_core.client.satellite_client import SatelliteClient, ClientConfig
from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
from fl_core.aggregation.ground_station import GroundStationAggregator

class SimpleModel(nn.Module):
    """简单的神经网络模型"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    

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
        
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 定义两个不同的处理器
        # 1. 文件处理器 - 记录详细日志
        file_handler = logging.FileHandler(
            f"logs/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录详细信息
        
        # 2. 控制台处理器 - 只显示简要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台只显示重要信息
        
        # 为不同处理器设置不同的格式
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(message)s'  # 控制台输出简化，只显示消息
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 配置logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
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
        self.data_generator = NonIIDGenerator(
            num_satellites=self.config['fl']['num_satellites'],
            feature_dim=self.config['data']['feature_dim'],
            num_classes=self.config['data']['num_classes']
        )
        
        # 初始化卫星客户端
        self.clients = {}
        self.model = SimpleModel(
            input_dim=self.config['data']['feature_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_classes=self.config['data']['num_classes']
        )
        
        # 初始化聚合器
        self.intra_orbit_aggregators = {}
        self.ground_station_aggregator = GroundStationAggregator(self.config['aggregation'])

        # 初始化地面站
        self.ground_stations = {}
        self.setup_ground_stations()
        self.logger.info("=== 初始化地面站 ===")
        for station_id, station in self.ground_stations.items():
            self.logger.info(f"地面站 {station_id} 负责轨道: {station.responsible_orbits}")
        
    def prepare_data(self):
        """准备训练数据"""
        # 生成训练数据
        self.satellite_datasets = self.data_generator.generate_data(
            total_samples=self.config['data']['total_samples'],
            dirichlet_alpha=self.config['data']['dirichlet_alpha'],
            mean_samples_per_satellite=self.config['data']['mean_samples_per_satellite']
        )
        
        # 生成测试数据
        self.test_dataset = self.data_generator.generate_test_data(
            self.config['data']['test_samples']
        )
        
        self.logger.info(f"数据生成完成，共{len(self.satellite_datasets)}个卫星节点")
        
    def setup_clients(self):
        """设置卫星客户端"""
        client_config = ClientConfig(**self.config['client'])
        
        for sat_id, dataset in self.satellite_datasets.items():
            # 创建客户端
            client = SatelliteClient(
                sat_id,
                self.model,
                client_config,
                self.network_manager,  # 使用网络管理器
                self.energy_model
            )
            client.set_dataset(dataset)
            self.clients[sat_id] = client
            
        self.logger.info(f"客户端设置完成，共{len(self.clients)}个客户端")

    def setup_ground_stations(self):
        """初始化地面站"""
        for i in range(3):
            config = self.config['aggregation'].copy()
            config['station_id'] = f"station_{i}"
            station = GroundStationAggregator(config)
            
            # 设置负责的轨道
            responsible_orbits = [i*2, i*2+1]
            station.responsible_orbits = responsible_orbits
            
            # 为每个负责的轨道设置权重
            for orbit_id in responsible_orbits:
                sat_name = f"satellite_{orbit_id + 1}"  # 使用 satellite_1 格式
                station.add_orbit(sat_name, 1.0)
                
            self.ground_stations[f"station_{i}"] = station
            self.logger.info(f"地面站 {i} 初始化完成: 负责卫星 {[f'satellite_{j+1}' for j in responsible_orbits]}")
        
    def train(self):
        """执行训练过程"""
        current_time = datetime.now().timestamp()

        for round_num in range(self.config['fl']['num_rounds']):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"开始第 {round_num + 1} 轮训练")
            self.logger.info(f"{'='*50}")

            # 1. 地面站独立处理各自负责的轨道
            self.logger.info("\n=== 阶段1: 地面站分发初始参数 ===")
            
            # 记录每个轨道的状态
            orbit_status = defaultdict(lambda: {
                'coordinator': None,           # 协调者卫星
                'params_distributed': False,   # 是否已分发参数
                'training_completed': False,   # 轨道内训练是否完成
                'orbit_aggregated': False,     # 轨道内聚合是否完成
                'model_sent_to_station': False # 是否已将模型发送给地面站
            })

            # 每个地面站处理其负责的轨道
            for station_id, station in self.ground_stations.items():
                self.logger.info(f"\n处理地面站 {station_id}:")
                
                # 处理每个负责的轨道
                for orbit_id in station.responsible_orbits:
                    self.logger.info(f"处理轨道 {orbit_id}:")
                    
                    # 等待直到找到可见的卫星作为协调者
                    coordinator = None
                    while not coordinator:
                        orbit_satellites = self._get_orbit_satellites(orbit_id)
                        for sat_id in orbit_satellites:
                            if self.network_model.check_visibility(station_id, sat_id, current_time):
                                coordinator = sat_id
                                break
                        if not coordinator:
                            self.logger.info(f"轨道 {orbit_id} 当前无可见卫星，等待60秒...")
                            current_time += 60
                            self.topology_manager.update_topology(current_time)

                    self.logger.info(f"轨道 {orbit_id} 选择 {coordinator} 作为协调者")
                    orbit_status[orbit_id]['coordinator'] = coordinator

                    # 分发初始参数给协调者
                    model_state = self.model.state_dict()
                    self.clients[coordinator].apply_model_update(model_state)
                    orbit_status[orbit_id]['params_distributed'] = True
                    self.logger.info(f"成功将参数分发给协调者 {coordinator}")

                    # 2. 协调者向轨道内其他卫星分发参数并开始训练
                    self.logger.info(f"\n=== 阶段2: 轨道 {orbit_id} 内参数分发 ===")
                    orbit_satellites = self._get_orbit_satellites(orbit_id)
                    for sat_id in orbit_satellites:
                        if sat_id != coordinator:
                            # 等待直到卫星可见
                            while not self.topology_manager.check_visibility(coordinator, sat_id, current_time):
                                current_time += 60
                                self.topology_manager.update_topology(current_time)
                            
                            # 分发参数
                            self.clients[sat_id].apply_model_update(model_state)
                            self.logger.info(f"协调者 {coordinator} 将参数分发给 {sat_id}")

                    # 3. 轨道内训练
                    self.logger.info(f"\n=== 阶段3: 轨道 {orbit_id} 训练 ===")
                    trained_satellites = set()
                    for sat_id in orbit_satellites:
                        stats = self.clients[sat_id].train(round_num)
                        trained_satellites.add(sat_id)
                        if stats['summary']['train_loss']:
                            self.logger.info(f"卫星 {sat_id} 完成训练: "
                                        f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                        f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                                        f"能耗={stats['summary']['energy_consumption']:.4f}Wh")
                        else:
                            self.logger.warning(f"卫星 {sat_id} 训练未产生有效结果")
                    
                    orbit_status[orbit_id]['training_completed'] = True

                    # 4. 轨道内聚合
                    self.logger.info(f"\n=== 阶段4: 轨道 {orbit_id} 聚合 ===")
                    if len(trained_satellites) == len(orbit_satellites):
                        aggregator = self.intra_orbit_aggregators.get(orbit_id)
                        if not aggregator:
                            aggregator = IntraOrbitAggregator(AggregationConfig(**self.config['aggregation']))
                            self.intra_orbit_aggregators[orbit_id] = aggregator

                        # 收集更新并聚合
                        for sat_id in orbit_satellites:
                            model_diff, _ = self.clients[sat_id].get_model_update()
                            self.logger.info(f"收集卫星 {sat_id} 的模型更新")
                            aggregator.receive_update(sat_id, round_num, model_diff, current_time)

                        orbit_update = aggregator.get_aggregated_update(round_num)
                        if orbit_update:
                            self.logger.info(f"轨道 {orbit_id} 完成聚合")
                            # 更新所有卫星的模型
                            for sat_id in orbit_satellites:
                                self.clients[sat_id].apply_model_update(orbit_update)
                                self.logger.info(f"更新卫星 {sat_id} 的模型参数")
                            orbit_status[orbit_id]['orbit_aggregated'] = True

                            # 5. 协调者将聚合后的模型发送给地面站
                            self.logger.info(f"\n=== 阶段5: 轨道 {orbit_id} 发送模型到地面站 ===")
                            while not self.network_model.check_visibility(station_id, coordinator, current_time):
                                current_time += 60
                                self.topology_manager.update_topology(current_time)

                            model_diff, _ = self.clients[coordinator].get_model_update()
                            success = station.receive_orbit_update(
                                str(orbit_id),
                                round_num,
                                model_diff,
                                len(orbit_satellites)
                            )
                            if success:
                                orbit_status[orbit_id]['model_sent_to_station'] = True
                                self.logger.info(f"轨道 {orbit_id} 的模型已发送给地面站 {station_id}")
                            else:
                                self.logger.warning(f"轨道 {orbit_id} 向地面站 {station_id} 发送模型失败")

            # 6. 地面站聚合
            self.logger.info("\n=== 阶段6: 地面站聚合 ===")
            station_aggregated = []
            for station_id, station in self.ground_stations.items():
                responsible_orbits = station.responsible_orbits
                received_orbits = [orbit_id for orbit_id in responsible_orbits 
                                if orbit_status[orbit_id]['model_sent_to_station']]
                
                self.logger.info(f"地面站 {station_id} 已收到轨道: {received_orbits}")
                
                if len(received_orbits) == len(responsible_orbits):
                    aggregated_update = station.get_aggregated_update(round_num)
                    if aggregated_update:
                        self.logger.info(f"地面站 {station_id} 完成聚合")
                        self.ground_station_aggregator.receive_station_update(
                            station_id,
                            round_num,
                            aggregated_update,
                            station.get_aggregation_stats()
                        )
                        station_aggregated.append(station_id)

            # 7. 全局聚合
            if len(station_aggregated) == len(self.ground_stations):
                self.logger.info("\n=== 阶段7: 全局聚合 ===")
                global_update = self.ground_station_aggregator.get_aggregated_update(round_num)
                if global_update:
                    self.logger.info(f"完成第 {round_num + 1} 轮全局聚合")
                    # 更新所有卫星的模型
                    for client in self.clients.values():
                        client.apply_model_update(global_update)
                    
                    accuracy = self.evaluate()
                    self.logger.info(f"全局聚合后测试准确率: {accuracy:.4f}")
                else:
                    self.logger.warning("全局聚合失败")
            else:
                self.logger.warning(f"只有部分地面站完成聚合: {station_aggregated}")

            # 更新时间
            current_time += self.config['fl']['round_interval']

            
    def _get_orbit_satellites(self, orbit_id: int) -> List[str]:
        """获取轨道内的所有卫星"""
        satellites = []
        # satellite_1 格式
        sat_num = orbit_id + 1
        sat_name = f"satellite_{sat_num}"
        if sat_name in self.clients:
            satellites.append(sat_name)
        return satellites

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
            is_visible = self.network_model.check_visibility(station_id, sat_id, current_time)
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
                    
    def evaluate(self) -> float:
        """评估全局模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        # 使用第一个客户端的模型进行评估
        test_model = next(iter(self.clients.values())).model
        
        with torch.no_grad():
            for features, labels in torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=100
            ):
                outputs = test_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        return accuracy
        
    def run(self):
        """运行实验"""
        self.logger.info("开始基线实验")
        
        # 准备数据
        self.prepare_data()
        
        # 设置客户端
        self.setup_clients()
        
        # 执行训练
        self.train()
        
        self.logger.info("实验完成")
        
def main():
    # 运行基线实验
    experiment = BaselineExperiment()
    experiment.run()
    
if __name__ == "__main__":
    main()