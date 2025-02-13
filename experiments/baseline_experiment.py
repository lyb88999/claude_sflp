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

from simulation.network_model import SatelliteNetwork
from simulation.comm_scheduler import CommunicationScheduler
from simulation.energy_model import EnergyModel
from simulation.topology_manager import TopologyManager
from data_simulator.non_iid_generator import NonIIDGenerator
from fl_core.client.satellite_client import SatelliteClient, ClientConfig
from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
from fl_core.aggregation.ground_station import GroundStationAggregator, GroundStationConfig

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
        
    def prepare_data(self):
        """准备训练数据"""
        # 生成训练数据，为66个卫星分配数据
        total_satellites = 66  # 6轨道 × 11卫星
        self.satellite_datasets = {}
        
        # 生成基础数据
        all_datasets = self.data_generator.generate_data(
            total_samples=self.config['data']['total_samples'],
            dirichlet_alpha=self.config['data']['dirichlet_alpha'],
            mean_samples_per_satellite=self.config['data']['mean_samples_per_satellite'],
            num_satellites=total_satellites  # 确保只生成66个数据集
        )
        
        # 为每个卫星分配数据集
        dataset_idx = 0
        for orbit_num in range(1, 7):  # 6个轨道
            for sat_num in range(1, 12):  # 每轨道11颗卫星
                sat_id = f"satellite_{orbit_num}-{sat_num}"
                if dataset_idx < len(all_datasets):
                    self.satellite_datasets[sat_id] = list(all_datasets.values())[dataset_idx]
                    dataset_idx += 1
        
        # 生成测试数据
        self.test_dataset = self.data_generator.generate_test_data(
            self.config['data']['test_samples']
        )
        
        # 记录数据分配情况
        for sat_id, dataset in self.satellite_datasets.items():
            self.logger.debug(f"卫星 {sat_id} 数据集大小: {len(dataset)}")
        
        self.logger.info(f"数据生成完成，共{len(self.satellite_datasets)}个卫星节点")
        
    def setup_clients(self):
        """设置卫星客户端"""
        client_config = ClientConfig(**self.config['client'])
        
        # 基础模型定义
        base_model = SimpleModel(
            input_dim=self.config['data']['feature_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_classes=self.config['data']['num_classes']
        )
        
        # 保存基础模型参数
        base_state_dict = base_model.state_dict()
        
        # 为每个轨道创建卫星
        for orbit in range(1, 7):  # 6个轨道
            for sat in range(1, 12):  # 每轨道11颗卫星
                sat_id = f"satellite_{orbit}-{sat}"
                
                # 创建新的模型实例
                model = SimpleModel(
                    input_dim=self.config['data']['feature_dim'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    num_classes=self.config['data']['num_classes']
                )
                model.load_state_dict(base_state_dict)
                
                # 创建客户端
                client = SatelliteClient(
                    sat_id,
                    model,
                    client_config,
                    self.network_manager,
                    self.energy_model
                )
                
                # 设置数据集
                if sat_id in self.satellite_datasets:
                    client.set_dataset(self.satellite_datasets[sat_id])
                else:
                    self.logger.warning(f"卫星 {sat_id} 没有对应的数据集")
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
        """执行训练过程"""
        current_time = datetime.now().timestamp()
        self.current_round = 0

        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"\n=== 开始第 {round_num + 1} 轮训练 ===")

            # 使用线程池并行处理每个地面站的轨道
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
                        success = future.result()
                        if success:
                            self.logger.info(f"轨道 {orbit_id} 成功完成训练并发送模型到地面站 {station_id}")
                        else:
                            self.logger.warning(f"轨道 {orbit_id} 训练或发送模型失败")
                    except Exception as e:
                        self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")

            # 地面站聚合
            self.logger.info("\n=== 地面站聚合阶段 ===")
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
                self._perform_global_aggregation(round_num)

            current_time += self.config['fl']['round_interval']

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

    def _distribute_orbit_params(self, coordinator: str, orbit_satellites: List[str], model_state: Dict, current_time: float):
        """
        在轨道内采用洪泛式传递参数
        Args:
            coordinator: 协调者卫星ID
            orbit_satellites: 轨道内所有卫星
            model_state: 模型参数
            current_time: 当前时间戳
        """
        self.logger.info(f"\n开始轨道内洪泛式参数传递, 协调者: {coordinator}")
        
        # 按照序号排序卫星
        orbit_num, coord_num = self._parse_satellite_id(coordinator)
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
                        # 传递参数
                        self.clients[next_sat].apply_model_update(model_state)
                        received_params.add(next_sat)
                        distribution_times[next_sat] = local_time + wait_time
                        self.logger.info(f"参数传递链: {current_sat} -> {next_sat} 成功")
                        
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
            self.logger.warning(f"以下卫星未收到参数: {missing_satellites}")
        else:
            self.logger.info("所有卫星已成功接收参数")
            
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
            station = self.ground_stations[station_id]
            self.logger.info(f"\n处理轨道 {orbit_id}:")
            
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
            # self.clients[coordinator].apply_model_update(model_state)
            # self.logger.info(f"成功将参数分发给协调者 {coordinator}")
            self.logger.info(f"\n=== 轨道 {orbit_id} 内参数分发 ===")
            current_time = self._distribute_orbit_params(coordinator, orbit_satellites, model_state, current_time)
            # 如果参数分发失败，提前返回
            if not current_time:  # 假设_distribute_orbit_params在失败时返回None
                self.logger.error(f"轨道 {orbit_id} 参数分发失败")
                return False

            # # 3. 协调者向轨道内其他卫星分发参数
            # self.logger.info(f"\n=== 轨道 {orbit_id} 内参数分发 ===")
            # orbit_satellites = self._get_orbit_satellites(orbit_id)
            # current_time = self._distribute_orbit_params(coordinator, orbit_satellites, model_state, current_time)



            # 4. 轨道内训练
            self.logger.info(f"\n=== 轨道 {orbit_id} 训练 ===")
            trained_satellites = set()
            training_stats = {}  # 记录每个卫星的训练状态

            for sat_id in orbit_satellites:
                stats = self.clients[sat_id].train(self.current_round)
                if stats['summary']['train_loss']:
                    trained_satellites.add(sat_id)
                    training_stats[sat_id] = stats
                    self.logger.info(f"卫星 {sat_id} 完成训练: "
                                f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                                f"能耗={stats['summary']['energy_consumption']:.4f}Wh")
                else:
                    self.logger.warning(f"卫星 {sat_id} 训练未产生有效结果")

            # 5. 轨道内聚合
            min_updates_required = self.config['aggregation']['min_updates']
            self.logger.info(f"需要至少 {min_updates_required} 个卫星更新，当前有 {len(trained_satellites)} 个")

            if len(trained_satellites) >= min_updates_required:
                self.logger.info(f"\n=== 轨道 {orbit_id} 聚合 ===")
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
                                return True
                    except Exception as e:
                        self.logger.error(f"发送模型到地面站时出错: {str(e)}")
                else:
                    self.logger.error(f"轨道 {orbit_id} 聚合失败: 无法获取有效的聚合结果")
            else:
                self.logger.warning(f"轨道 {orbit_id} 训练的卫星数量不足: {len(trained_satellites)} < {min_updates_required}")

            return False

        except Exception as e:
            self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
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
            self.logger.info("\n=== 全局聚合阶段 ===")
            
            # 获取全局聚合结果
            global_update = self.global_aggregator.get_aggregated_update(round_num)
            if global_update:
                self.logger.info(f"完成第 {round_num + 1} 轮全局聚合")
                
                # 更新所有卫星的模型
                update_success = 0
                for client in self.clients.values():
                    try:
                        client.apply_model_update(global_update)
                        update_success += 1
                    except Exception as e:
                        self.logger.error(f"更新客户端 {client.client_id} 时出错: {str(e)}")
                
                self.logger.info(f"成功更新了 {update_success}/{len(self.clients)} 个卫星的模型")
                
                # 评估全局模型
                accuracy = self.evaluate()
                self.logger.info(f"全局聚合后测试准确率: {accuracy:.4f}")
                return True
            else:
                self.logger.warning("全局聚合失败")
                return False
                
        except Exception as e:
            self.logger.error(f"全局聚合出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        

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