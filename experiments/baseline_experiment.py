from simulation.network_manager import NetworkManager
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import yaml
import logging
from pathlib import Path

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
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
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
        
    def train(self):
        """执行训练过程"""
        current_time = datetime.now().timestamp()
        
        for round_num in range(self.config['fl']['num_rounds']):
            self.logger.info(f"开始第{round_num + 1}轮训练")
            
            # 更新网络拓扑
            self.topology_manager.update_topology(current_time)
            
            # 本地训练
            for client_id, client in self.clients.items():
                self.logger.info(f"客户端{client_id}开始训练")
                stats = client.train(round_num)
                self.logger.info(f"客户端{client_id}训练完成: {stats}")
                
            # 轨道内聚合
            self._perform_intra_orbit_aggregation(round_num)
            
            # 地面站聚合
            self._perform_ground_station_aggregation(round_num)
            
            # 评估模型
            accuracy = self.evaluate()
            self.logger.info(f"第{round_num + 1}轮训练完成，测试准确率: {accuracy:.4f}")
            
            # 更新时间
            current_time += self.config['fl']['round_interval']
            
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