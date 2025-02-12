import logging
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from skyfield.api import load
from skyfield.api import utc

@dataclass
class SatelliteEnergyConfig:
    """卫星能源配置"""
    battery_capacity: float = 1000.0    # 电池容量 (Wh)
    solar_panel_area: float = 2.5       # 太阳能板面积 (m²)
    solar_efficiency: float = 0.3       # 太阳能转换效率
    cpu_power: float = 5.0             # CPU功耗 (W) - 降低功耗
    radio_power_idle: float = 1.0      # 通信模块空闲功耗 (W)
    radio_power_tx: float = 10.0       # 通信模块发送功耗 (W)
    radio_power_rx: float = 5.0        # 通信模块接收功耗 (W)

class EnergyModel:
    def __init__(self, network_model, config_file: str):
        """
        初始化能源模型
        Args:
            network_model: 卫星网络模型实例
            config_file: 能源配置文件路径
        """
        self.network_model = network_model
        self.configs = self._load_configs(config_file)
        self.battery_levels = {}  # 电池电量状态
        self.energy_usage = {}    # 能量使用记录
        self.solar_intensity = 1361.0  # 太阳常数 (W/m²)
        self.current_satellite = None  # 添加当前卫星属性
        
        # 初始化天文数据
        self.ts = load.timescale()
        self.sun = load('de421.bsp')['sun']
        self.earth = load('de421.bsp')['earth']

        self.logger = logging.getLogger(__name__)
        
    def _load_configs(self, config_file: str) -> Dict[str, SatelliteEnergyConfig]:
        """加载卫星能源配置"""
        # 这里应该实现配置文件的解析
        # 为演示使用硬编码的默认值
        default_config = SatelliteEnergyConfig(
            battery_capacity=1000.0,    # 1000Wh
            solar_panel_area=2.5,       # 2.5m²
            solar_efficiency=0.3,       # 30%效率
            cpu_power=15.0,            # 15W
            radio_power_idle=2.0,      # 2W
            radio_power_tx=20.0,       # 20W
            radio_power_rx=10.0        # 10W
        )
        return {'default': default_config}
        
    def get_satellite_config(self, sat_name: str) -> SatelliteEnergyConfig:
        """获取卫星配置"""
        return self.configs.get(sat_name, self.configs['default'])
        
    def initialize_battery(self, satellite: str, initial_level: float = 0.8):
        """初始化卫星电池电量"""
        config = self.get_satellite_config(satellite)
        self.battery_levels[satellite] = config.battery_capacity * initial_level
        self.logger.info(f"已初始化卫星 {satellite} 电池电量: {self.battery_levels[satellite]:.2f} Wh")
        
    def calculate_solar_power(self, sat_name: str, time: float) -> float:
        """
        计算太阳能发电功率
        Args:
            sat_name: 卫星名称
            time: 时间戳
        Returns:
            太阳能发电功率 (W)
        """
        try:
            config = self.get_satellite_config(sat_name)
            
            # 确保时间戳是有效的
            if not isinstance(time, (int, float)):
                raise ValueError(f"Invalid timestamp: {time}")
                
            # 获取卫星位置
            sat_pos = self.network_model.compute_position(sat_name, time)
            
            # 验证位置计算结果
            if np.any(np.isnan(sat_pos)):
                print(f"警告: 卫星 {sat_name} 位置计算结果包含 NaN 值")
                return 0.0
                
            dt = datetime.fromtimestamp(time, tz=utc)
            t = self.ts.from_datetime(dt)
            
            # 计算太阳方向
            sun_pos = (self.sun - self.earth).at(t).position.km
            
            # 计算卫星-太阳矢量
            sat_sun_vector = sun_pos - sat_pos
            sat_sun_vector = sat_sun_vector / np.linalg.norm(sat_sun_vector)
            
            # 检查地球遮挡
            earth_radius = 6371.0  # km
            sat_distance = np.linalg.norm(sat_pos)
            earth_angle = np.arcsin(earth_radius / sat_distance)
            sun_angle = np.arccos(np.dot(-sat_pos/sat_distance, sat_sun_vector))
            
            if sun_angle < earth_angle:
                return 0.0  # 卫星在地球阴影中
                
            # 假设太阳能板始终朝向太阳
            # 实际应考虑卫星姿态和太阳能板方向
            incident_power = self.solar_intensity * config.solar_panel_area
            
            return incident_power * config.solar_efficiency
            
        except ValueError as e:
            print(f"计算太阳能功率时出现值错误: {str(e)}")
            return 0.0
        except Exception as e:
            print(f"计算太阳能功率时出现未知错误: {str(e)}")
            return 0.0
        
    def calculate_transmission_energy(self, 
                                   sat_name: str,
                                   data_size: float,
                                   bandwidth: float) -> float:
        """
        计算数据传输所需能量
        Args:
            sat_name: 卫星名称
            data_size: 数据量 (MB)
            bandwidth: 带宽 (Mbps)
        Returns:
            能量消耗 (Wh)
        """
        config = self.get_satellite_config(sat_name)
        
        # 计算传输时间(小时)
        transmission_time = data_size / (bandwidth * 3600)  # MB / (Mbps * s/h)
        
        # 计算能量消耗
        energy = config.radio_power_tx * transmission_time
        
        return energy
        
    def calculate_computation_energy(self,
                                  sat_name: str,
                                  computation_cycles: int) -> float:
        """
        计算计算任务能量消耗
        Args:
            sat_name: 卫星名称
            computation_cycles: 计算循环数
        Returns:
            能量消耗 (Wh)
        """
        config = self.get_satellite_config(sat_name)
        
        # 假设每个循环1ns
        computation_time = computation_cycles * 1e-9 / 3600  # 转换为小时
        
        return config.cpu_power * computation_time
        
    def update_battery_level(self, 
                           sat_name: str,
                           time_start: float,
                           time_end: float,
                           energy_consumption: float = 0.0):
        """
        更新电池电量
        Args:
            sat_name: 卫星名称
            time_start: 开始时间戳
            time_end: 结束时间戳
            energy_consumption: 额外能量消耗 (Wh)
        """
        if sat_name not in self.battery_levels:
            self.initialize_battery(sat_name)
            
        config = self.get_satellite_config(sat_name)
        duration = (time_end - time_start) / 3600  # 转换为小时
        
        # 计算基础能耗
        base_consumption = (config.cpu_power + config.radio_power_idle) * duration
        
        # 计算太阳能充电
        # 使用多个采样点计算平均太阳能
        num_samples = max(int(duration * 60), 1)  # 至少1个采样点
        times = np.linspace(time_start, time_end, num_samples)
        solar_power = np.mean([self.calculate_solar_power(sat_name, t) for t in times])
        solar_energy = solar_power * duration
        
        # 更新电池电量
        energy_delta = solar_energy - base_consumption - energy_consumption
        new_level = self.battery_levels[sat_name] + energy_delta
        
        # 确保电池电量在有效范围内
        self.battery_levels[sat_name] = min(max(0.0, new_level), config.battery_capacity)
        
        # 记录能量使用
        self.energy_usage[sat_name].append({
            'time_start': time_start,
            'time_end': time_end,
            'solar_energy': solar_energy,
            'base_consumption': base_consumption,
            'task_consumption': energy_consumption,
            'battery_level': self.battery_levels[sat_name]
        })
        
    def get_battery_level(self, satellite: str) -> float:
        """获取电池电量"""
        if satellite not in self.battery_levels:
            self.initialize_battery(satellite)
        return self.battery_levels[satellite]
        
    def get_transmission_capacity(self, sat_name: str) -> float:
        """
        获取卫星当前的传输能力系数 (0-1)
        用于通信调度决策
        """
        config = self.get_satellite_config(sat_name)
        battery_level = self.get_battery_level(sat_name)
        
        # 当电量低于20%时开始限制传输能力
        if battery_level < 0.2 * config.battery_capacity:
            return battery_level / (0.2 * config.battery_capacity)
        return 1.0
        
    def can_schedule_task(self, 
                         sat_name: str,
                         energy_required: float,
                         margin: float = 0.1) -> bool:
        """
        检查是否可以调度新任务
        Args:
            sat_name: 卫星名称
            energy_required: 所需能量 (Wh)
            margin: 安全边际 (占总容量比例)
        Returns:
            bool: 是否可以调度
        """
        config = self.get_satellite_config(sat_name)
        battery_level = self.get_battery_level(sat_name)
        
        # 保留最小电量
        min_level = config.battery_capacity * margin
        
        return (battery_level - energy_required) >= min_level
        
    def get_energy_statistics(self, sat_name: str) -> Dict:
        """获取能量使用统计"""
        if sat_name not in self.energy_usage:
            return {}
            
        usage = self.energy_usage[sat_name]
        total_solar = sum(record['solar_energy'] for record in usage)
        total_base = sum(record['base_consumption'] for record in usage)
        total_task = sum(record['task_consumption'] for record in usage)
        
        return {
            'total_solar_energy': total_solar,
            'total_base_consumption': total_base,
            'total_task_consumption': total_task,
            'current_battery_level': self.get_battery_level(sat_name)
        }
    
    def can_consume(self, satellite_id: str, amount: float) -> bool:
        """
        检查是否有足够的能量消耗
        Args:
            satellite_id: 卫星ID
            amount: 需要消耗的能量(Wh)
        """
        if amount <= 0:
            return True
            
        # 降低能量检查的阈值
        min_level = 0.1  # 电池最低电量阈值降低到10%
        
        # 确保卫星有电池电量记录
        if satellite_id not in self.battery_levels:
            self.initialize_battery(satellite_id)
            
        current_level = self.battery_levels.get(satellite_id, 0)
        return current_level - amount >= current_level * min_level
        
    def has_minimum_energy(self, satellite: str) -> bool:
        """检查是否有最小运行能量"""
        if satellite not in self.battery_levels:
            self.initialize_battery(satellite)
            
        config = self.get_satellite_config(satellite)
        min_level = config.battery_capacity * 0.2  # 20%的电池容量作为最小能量
        current_level = self.battery_levels[satellite]
        
        has_energy = current_level >= min_level
        if not has_energy:
            print(f"卫星 {satellite} 能量不足: {current_level:.2f} Wh < {min_level:.2f} Wh")
        
        return has_energy
    
    def consume_energy(self, satellite_id: str, amount: float):
        """
        消耗能量
        Args:
            satellite_id: 卫星ID
            amount: 消耗的能量(Wh)
        """
        if satellite_id not in self.battery_levels:
            self.initialize_battery(satellite_id)
            
        self.battery_levels[satellite_id] -= amount