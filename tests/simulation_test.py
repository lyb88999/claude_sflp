import unittest
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
import yaml

from simulation.network_model import SatelliteNetwork
from simulation.comm_scheduler import CommunicationScheduler, CommunicationTask
from simulation.energy_model import EnergyModel
from simulation.topology_manager import TopologyManager

class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前执行一次"""
        cls._generate_test_data()
    
    @classmethod
    def _generate_test_data(cls):
        """生成测试用的TLE数据"""
        # 生成TLE数据
        tle_template = """Iridium {number}              
1 2412{num} U 97030{num} {epoch} .00000000  00000-0  00000-0 0  9999
2 2412{num} {inc} {raan} 0001000   0.0000 {mean}  1.00270000    00"""
        
        epoch = datetime.now().strftime('%y%j.00000000')  # 当前时间
        tle_data = ""
        
        # 生成6颗测试卫星的TLE
        for i in range(6):
            inc = 86.4  # 倾角
            raan = (i * 60) % 360  # 升交点赤经
            mean_anomaly = (i * 60) % 360  # 平近点角
            
            tle_data += tle_template.format(
                number=i+1,
                num=str(i+1).zfill(2),
                epoch=epoch,
                inc=f"{inc:8.4f}",
                raan=f"{raan:8.4f}",
                mean=f"{mean_anomaly:8.4f}"
            ) + "\n"
            
        # 保存TLE数据
        with open('test_tle.txt', 'w') as f:
            f.write(tle_data)
            
        # 创建能源配置
        energy_config = {
            'default': {
                'battery_capacity': 1000.0,
                'solar_panel_area': 2.5,
                'solar_efficiency': 0.3,
                'cpu_power': 15.0,
                'radio_power_idle': 2.0,
                'radio_power_tx': 20.0,
                'radio_power_rx': 10.0
            }
        }
        with open('test_energy_config.yaml', 'w') as f:
            yaml.dump(energy_config, f)
    
    def setUp(self):
        """每个测试方法开始前执行"""
        # 初始化组件
        self.network_model = SatelliteNetwork('test_tle.txt')
        self.comm_scheduler = CommunicationScheduler(self.network_model)
        self.energy_model = EnergyModel(self.network_model, 'test_energy_config.yaml')
        self.topology_manager = TopologyManager(
            self.network_model,
            self.comm_scheduler,
            self.energy_model
        )

    def test_network_model(self):
        """测试网络模型"""
        print("\n=== 测试网络模型 ===")
        current_time = datetime.now()
        
        # 测试位置计算
        sat_name = "Iridium 1"
        position = self.network_model.compute_position(
            sat_name, 
            current_time.timestamp()
        )
        print(f"卫星 {sat_name} 的位置(ECEF):", position)
        self.assertEqual(len(position), 3)  # 确保返回三维坐标
        
        # 测试可见性
        sat1, sat2 = "Iridium 1", "Iridium 2"
        is_visible = self.network_model.check_visibility(
            sat1, sat2,
            current_time.timestamp()
        )
        print(f"卫星 {sat1} 和 {sat2} 是否可见:", is_visible)
        self.assertIsInstance(is_visible, bool)
        
        # 测试多普勒频移
        doppler = self.network_model.compute_doppler_shift(
            sat1, sat2,
            current_time.timestamp(),
            2.4e9  # 2.4GHz
        )
        print(f"多普勒频移: {doppler:.2f} Hz")
        self.assertIsInstance(doppler, float)

    def test_comm_scheduler(self):
        """测试通信调度器"""
        print("\n=== 测试通信调度器 ===")
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=1)
        
        # 测试窗口预测
        sat1, sat2 = "Iridium 1", "Iridium 2"
        windows = self.comm_scheduler.predict_communication_windows(
            current_time.timestamp(),
            end_time.timestamp(),
            (sat1, sat2)
        )
        print(f"预测到的通信窗口数量: {len(windows)}")
        if windows:
            window = windows[0]
            print(f"第一个窗口: {window}")
        
        # 测试任务调度
        task = CommunicationTask(
            task_id="test_task",
            source=sat1,
            target=sat2,
            data_size=100.0,  # 100MB
            priority=5,
            deadline=end_time.timestamp()
        )
        self.comm_scheduler.add_task(task)
        
        schedule = self.comm_scheduler.schedule_tasks(
            current_time.timestamp(),
            3600  # 1小时调度范围
        )
        print("任务调度结果:", schedule)
        self.assertIsInstance(schedule, dict)

    def test_energy_model(self):
        """测试能源模型"""
        print("\n=== 测试能源模型 ===")
        current_time = datetime.now()
        sat_name = "Iridium 1"
        
        # 初始化电池
        self.energy_model.initialize_battery(sat_name)
        initial_level = self.energy_model.get_battery_level(sat_name)
        print(f"初始电池电量: {initial_level:.2f} Wh")
        self.assertGreater(initial_level, 0)
        
        # 测试太阳能发电
        solar_power = self.energy_model.calculate_solar_power(
            sat_name,
            current_time.timestamp()
        )
        print(f"当前太阳能发电功率: {solar_power:.2f} W")
        self.assertGreaterEqual(solar_power, 0)
        
        # 测试能量消耗计算
        transmission_energy = self.energy_model.calculate_transmission_energy(
            sat_name,
            100.0,  # 100MB数据
            50.0    # 50Mbps带宽
        )
        print(f"传输100MB数据需要的能量: {transmission_energy:.2f} Wh")
        self.assertGreater(transmission_energy, 0)

    def test_topology_manager(self):
        """测试拓扑管理器"""
        print("\n=== 测试拓扑管理器 ===")
        current_time = datetime.now()
        
        # 更新拓扑
        self.topology_manager.update_topology(
            current_time.timestamp(),
            300  # 5分钟窗口
        )
        
        # 检查分组情况
        for group_id, group in self.topology_manager.groups.items():
            print(f"\n分组 {group_id}:")
            print(f"组长: {group.leader}")
            print(f"成员: {group.members}")
            print(f"平均连接性: {group.avg_connectivity:.2f}")
            
        # 测试路由功能
        sat1, sat2 = "Iridium 1", "Iridium 4"
        next_hop = self.topology_manager.get_next_hop(sat1, sat2)
        print(f"\n从 {sat1} 到 {sat2} 的下一跳: {next_hop}")
        
        # 优化拓扑
        self.topology_manager.optimize_topology()
        print("\n拓扑优化完成")

    def test_visualization(self):
        """测试网络可视化"""
        print("\n=== 生成网络可视化 ===")
        current_time = datetime.now()
        
        # 获取所有卫星位置
        positions = {}
        for sat_name in self.network_model.satellites.keys():
            pos = self.network_model.compute_position(
                sat_name,
                current_time.timestamp()
            )
            positions[sat_name] = pos
        
        self.assertGreater(len(positions), 0)
        print("成功获取卫星位置数据")

if __name__ == '__main__':
    unittest.main()