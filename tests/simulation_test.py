import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
import yaml

# 导入我们实现的模块
from simulation.network_model import SatelliteNetwork
from simulation.comm_scheduler import CommunicationScheduler, CommunicationTask
from simulation.energy_model import EnergyModel
from simulation.topology_manager import TopologyManager

class SimulationTester:
    def __init__(self):
        """初始化测试环境"""
        # 生成测试用的TLE数据
        self.tle_data = self._generate_test_tle()
        with open('test_tle.txt', 'w') as f:
            f.write(self.tle_data)
            
        # 创建测试配置
        self.energy_config = {
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
            yaml.dump(self.energy_config, f)
            
        # 初始化组件
        self.network_model = SatelliteNetwork('test_tle.txt')
        self.comm_scheduler = CommunicationScheduler(self.network_model)
        self.energy_model = EnergyModel(self.network_model, 'test_energy_config.yaml')
        self.topology_manager = TopologyManager(
            self.network_model,
            self.comm_scheduler,
            self.energy_model
        )
        
    def _generate_test_tle(self) -> str:
        """生成测试用的TLE数据（简化的铱星星座）"""
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
            
        return tle_data
        
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
        
        # 测试可见性
        sat1, sat2 = "Iridium 1", "Iridium 2"
        is_visible = self.network_model.check_visibility(
            sat1, sat2,
            current_time.timestamp()
        )
        print(f"卫星 {sat1} 和 {sat2} 是否可见:", is_visible)
        
        # 测试多普勒频移
        doppler = self.network_model.compute_doppler_shift(
            sat1, sat2,
            current_time.timestamp(),
            2.4e9  # 2.4GHz
        )
        print(f"多普勒频移: {doppler:.2f} Hz")
        
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
        
    def test_energy_model(self):
        """测试能源模型"""
        print("\n=== 测试能源模型 ===")
        current_time = datetime.now()
        sat_name = "Iridium 1"
        
        # 初始化电池
        self.energy_model.initialize_battery(sat_name)
        initial_level = self.energy_model.get_battery_level(sat_name)
        print(f"初始电池电量: {initial_level:.2f} Wh")
        
        # 测试太阳能发电
        solar_power = self.energy_model.calculate_solar_power(
            sat_name,
            current_time.timestamp()
        )
        print(f"当前太阳能发电功率: {solar_power:.2f} W")
        
        # 测试能量消耗计算
        transmission_energy = self.energy_model.calculate_transmission_energy(
            sat_name,
            100.0,  # 100MB数据
            50.0    # 50Mbps带宽
        )
        print(f"传输100MB数据需要的能量: {transmission_energy:.2f} Wh")
        
        # 更新电池电量
        self.energy_model.update_battery_level(
            sat_name,
            current_time.timestamp(),
            (current_time + timedelta(minutes=30)).timestamp(),
            transmission_energy
        )
        new_level = self.energy_model.get_battery_level(sat_name)
        print(f"更新后的电池电量: {new_level:.2f} Wh")
        
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
        
    def visualize_network(self):
        """可视化网络状态"""
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
            
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地球
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        earth_radius = 6371.0
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # 绘制卫星和连接
        for sat_name, pos in positions.items():
            ax.scatter(pos[0], pos[1], pos[2], c='red', marker='o')
            ax.text(pos[0], pos[1], pos[2], sat_name, fontsize=8)
            
        # 绘制卫星间连接
        for edge in self.topology_manager.topology_graph.edges():
            sat1, sat2 = edge
            pos1 = positions[sat1]
            pos2 = positions[sat2]
            ax.plot([pos1[0], pos2[0]], 
                   [pos1[1], pos2[1]], 
                   [pos1[2], pos2[2]], 
                   'g-', alpha=0.5)
            
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Satellite Network Topology')
        
        plt.savefig('network_visualization.png')
        print("网络可视化已保存为 'network_visualization.png'")
        
def main():
    """运行所有测试"""
    tester = SimulationTester()
    
    # 运行各组件测试
    tester.test_network_model()
    tester.test_comm_scheduler()
    tester.test_energy_model()
    tester.test_topology_manager()
    
    # 生成可视化
    tester.visualize_network()

if __name__ == "__main__":
    main()