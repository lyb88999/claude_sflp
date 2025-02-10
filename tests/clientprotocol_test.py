import unittest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from fl_core.client.satellite_client import SatelliteClient, ClientConfig
from fl_core.protocols.delay_simulator import DelaySimulator, LinkState
from simulation.network_model import SatelliteNetwork

# 简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

# 模拟数据集
class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 模拟网络管理器
class MockNetworkManager:
    def __init__(self):
        self.connected = True
        self.has_priority = False
        
    def is_connected(self):
        return self.connected
        
    def has_priority_task(self):
        return self.has_priority

# 模拟能源管理器
class MockEnergyManager:
    def __init__(self):
        self.energy_level = 1000.0
        
    def can_consume(self, amount):
        return self.energy_level >= amount
        
    def consume_energy(self, amount):
        self.energy_level -= amount
        
    def get_energy_level(self):
        return self.energy_level
        
    def has_minimum_energy(self):
        return self.energy_level > 100.0

class TestSatelliteClient(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.config = ClientConfig()
        self.network_manager = MockNetworkManager()
        self.energy_manager = MockEnergyManager()
        self.client = SatelliteClient(
            "test_client",
            self.model,
            self.config,
            self.network_manager,
            self.energy_manager
        )
        self.dataset = MockDataset()
        
    def test_client_initialization(self):
        """测试客户端初始化"""
        self.assertEqual(self.client.client_id, "test_client")
        self.assertIsNotNone(self.client.optimizer)
        self.assertFalse(self.client.is_training)
        
    def test_local_training(self):
        """测试本地训练"""
        self.client.set_dataset(self.dataset)
        stats = self.client.train(round_number=1)
        
        self.assertIn('train_loss', stats)
        self.assertIn('train_accuracy', stats)
        self.assertIn('energy_consumption', stats)
        self.assertIn('compute_time', stats)
        
    def test_energy_constraint(self):
        """测试能量约束"""
        self.client.set_dataset(self.dataset)
        self.energy_manager.energy_level = 0.1  # 设置极低能量
        
        stats = self.client.train(round_number=1)
        self.assertEqual(stats['energy_consumption'], 0.0)
        
    def test_model_update(self):
        """测试模型更新"""
        self.client.set_dataset(self.dataset)
        self.client.train(round_number=1)
        
        model_diff, stats = self.client.get_model_update()
        self.assertIsInstance(model_diff, dict)
        self.assertTrue(all(isinstance(param, torch.Tensor) 
                          for param in model_diff.values()))
                          
class TestDelaySimulator(unittest.TestCase):
    def setUp(self):
        # 创建包含两颗卫星的TLE数据
        tle_data = """Iridium 1              
1 24101U 96020A   23001.00000000  .00000000  00000-0  00000-0 0  9999
2 24101  86.4000   0.0000 0001000   0.0000   0.0000 14.34920000    0
Iridium 2              
1 24102U 96020B   23001.00000000  .00000000  00000-0  00000-0 0  9999
2 24102  86.4000  60.0000 0001000   0.0000  60.0000 14.34920000    0"""
        
        with open('test_tle.txt', 'w') as f:
            f.write(tle_data)
            
        self.network_model = SatelliteNetwork('test_tle.txt')
        self.simulator = DelaySimulator(self.network_model)
        
        # 设置测试链路
        self.link_state = LinkState(
            bandwidth=100.0,
            base_delay=5.0,
            jitter=1.0,
            packet_loss=0.01,
            queue_size=100
        )
        self.simulator.set_link_state("Iridium 1", "Iridium 2", self.link_state)
        
    def test_delay_simulation(self):
        """测试延迟模拟"""
        packet_id = "test_packet"
        size = 1000  # bytes
        
        delivery_time = self.simulator.schedule_transmission(
            "Iridium 1", "Iridium 2", packet_id, size
        )
        
        self.assertIsNotNone(delivery_time)
        self.assertGreater(delivery_time, self.simulator.current_time)
        
    def test_packet_loss(self):
        """测试丢包"""
        # 设置100%丢包率
        self.link_state.packet_loss = 1.0
        self.simulator.set_link_state("Iridium 1", "Iridium 2", self.link_state)
        
        delivery_time = self.simulator.schedule_transmission(
            "Iridium 1", "Iridium 2", "test_packet", 1000
        )
        
        self.assertIsNone(delivery_time)
        
    def test_queue_overflow(self):
        """测试队列溢出"""
        # 设置小队列和大数据包
        self.link_state.queue_size = 1
        self.simulator.set_link_state("Iridium 1", "Iridium 2", self.link_state)
        
        # 清空现有队列
        self.simulator.packet_queues[("Iridium 1", "Iridium 2")] = []
        
        # 发送第一个数据包（应该成功）
        first_delivery = self.simulator.schedule_transmission(
            "Iridium 1", "Iridium 2", "packet_0", 1000
        )
        self.assertIsNotNone(first_delivery)
        
        # 确保第一个数据包在队列中
        queue = self.simulator.packet_queues[("Iridium 1", "Iridium 2")]
        self.assertEqual(len(queue), 1, "Queue should contain one packet")
        
        # 发送第二个数据包（应该失败，因为队列已满）
        second_delivery = self.simulator.schedule_transmission(
            "Iridium 1", "Iridium 2", "packet_1", 1000
        )
        self.assertIsNone(second_delivery, "Queue overflow should result in None")
        
        # 验证队列大小没有改变
        self.assertEqual(len(queue), 1, "Queue size should remain 1 after overflow")
                
    def test_bandwidth_limit(self):
        """测试带宽限制"""
        start_time = self.simulator.current_time
        
        # 发送大数据包
        delivery_time = self.simulator.schedule_transmission(
            "Iridium 1", "Iridium 2", "large_packet", 1_000_000  # 1MB
        )
        
        # 验证传输时间符合带宽限制
        transmission_time = delivery_time - start_time
        expected_min_time = (1_000_000 * 8) / (self.link_state.bandwidth * 1e6)
        self.assertGreaterEqual(transmission_time, expected_min_time)

if __name__ == '__main__':
    unittest.main()