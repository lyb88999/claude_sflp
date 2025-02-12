import logging
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import load, EarthSatellite, utc, wgs84
from astropy import units as u
from astropy.coordinates import CartesianRepresentation

class SatelliteNetwork:
    def __init__(self, tle_file: str):
        """
        初始化卫星网络模型
        Args:
            tle_file: TLE数据文件路径
        """
        self.ts = load.timescale()
        self.satellites = self._load_satellites(tle_file)
        self.positions_cache = {}  # 位置缓存

        # 添加logger
        self.logger = logging.getLogger(__name__)

        # 添加地面站位置 (经度,纬度,高度)
        self.ground_stations = {
            "station_0": (0.0, 0.0, 0.0),      # 经度0度赤道
            "station_1": (120.0, 0.0, 0.0),    # 经度120度赤道
            "station_2": (-120.0, 0.0, 0.0),   # 经度-120度赤道
        }
        
    def _load_satellites(self, tle_file: str) -> Dict[str, EarthSatellite]:
        """
        从TLE文件加载卫星数据
        """
        satellites = {}
        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                raise ValueError("TLE file is empty")
                
            print(f"读取到 {len(lines)} 行TLE数据")
            
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                try:
                    name = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    print(f"正在加载卫星: {name}")
                    print(f"Line 1: {line1}")
                    print(f"Line 2: {line2}")
                    
                    satellite = EarthSatellite(line1, line2, name, self.ts)
                    satellites[name] = satellite
                    
                except Exception as e:
                    print(f"加载卫星 {name} 时出错: {str(e)}")
                    continue
                    
            if not satellites:
                raise ValueError("No valid satellites loaded from TLE file")
                
        except Exception as e:
            print(f"加载TLE文件时出错: {str(e)}")
            raise
            
        return satellites
    
    def compute_position(self, sat_name: str, time: float) -> np.ndarray:
        """计算卫星位置"""
        try:
            # 统一使用 satellite_X-X 格式
            # 如果是 Iridium X-X 格式，转换为 satellite_X-X
            if sat_name.startswith('Iridium'):
                parts = sat_name.split()[1].split('-')
                sat_name = f"satellite_{parts[0]}-{parts[1]}"
            
            orbit_num, sat_num = self._parse_satellite_id(sat_name)
            
            # 基本轨道参数
            orbit_radius = 7155.0  # 轨道半径(km)
            inclination = np.radians(86.4)  # 轨道倾角
            
            # 计算轨道平面的升交点赤经（每个轨道平面间隔60度）
            raan = np.radians((orbit_num - 1) * 60)
            
            # 计算卫星在轨道上的位置（11颗卫星均匀分布）
            # 每颗卫星间隔 360/11 度
            phase_angle = 360.0 / 11  # 相位角
            
            # 根据卫星序号计算真近点角
            true_anomaly = np.radians((sat_num - 1) * phase_angle)
            
            # 计算轨道平面中的位置
            x_orbit = orbit_radius * np.cos(true_anomaly)
            y_orbit = orbit_radius * np.sin(true_anomaly)
            
            # 考虑时间对位置的影响
            # 轨道周期约为100分钟
            orbit_period = 100 * 60  # 秒
            time_fraction = (time % orbit_period) / orbit_period
            rotation = 2 * np.pi * time_fraction
            
            # 添加时间旋转
            x_rotated = x_orbit * np.cos(rotation) - y_orbit * np.sin(rotation)
            y_rotated = x_orbit * np.sin(rotation) + y_orbit * np.cos(rotation)
            
            # 1. 绕z轴旋转(升交点赤经)
            x1 = x_rotated * np.cos(raan) - y_rotated * np.sin(raan)
            y1 = x_rotated * np.sin(raan) + y_rotated * np.cos(raan)
            z1 = 0
            
            # 2. 绕x轴旋转(轨道倾角)
            x2 = x1
            y2 = y1 * np.cos(inclination)
            z2 = y1 * np.sin(inclination)
            
            position = np.array([x2, y2, z2])
            
            # 添加日志以便调试
            self.logger.debug(f"计算卫星 {sat_name} 位置:")
            self.logger.debug(f"  轨道: {orbit_num}, 序号: {sat_num}")
            self.logger.debug(f"  位置: [{x2:.2f}, {y2:.2f}, {z2:.2f}]")
            
            return position
            
        except Exception as e:
            self.logger.error(f"计算卫星 {sat_name} 位置时出错: {str(e)}")
            return np.array([0, 0, 0])

    
    # def check_visibility(self, src: str, dst: str, time: float) -> bool:
    #     """
    #     检查源节点和目标节点之间是否可见
    #     Args:
    #         src: 源节点ID(可以是卫星或地面站)
    #         dst: 目标节点ID(可以是卫星或地面站)
    #         time: 时间戳
    #     Returns:
    #         bool: 是否可见
    #     """
    #     try:
    #         # 转换卫星ID格式
    #         def convert_sat_id(sat_id: str) -> str:
    #             if sat_id.startswith('satellite_'):
    #                 num = sat_id.split('_')[1]
    #                 return f"Iridium {num}"
    #             return sat_id

    #         # 如果包含地面站
    #         if src.startswith('station_') or dst.startswith('station_'):
    #             station_id = src if src.startswith('station_') else dst
    #             sat_id = dst if dst.startswith('satellite_') else src
    #             return self.check_ground_station_visibility(station_id, convert_sat_id(sat_id), time)
                
    #         # 卫星间可见性检查
    #         src_id = convert_sat_id(src)
    #         dst_id = convert_sat_id(dst)
            
    #         try:
    #             sat1_num = int(src_id.split()[-1])
    #             sat2_num = int(dst_id.split()[-1])
                
    #             # 计算轨道平面差异
    #             plane_diff = abs(sat1_num - sat2_num)
    #             if plane_diff > 3:  # 如果相差超过半个星座，取补值
    #                 plane_diff = 6 - plane_diff
                    
    #             # 检查是否为相邻轨道平面或同一轨道平面
    #             if plane_diff > 1:
    #                 return False  # 非相邻轨道平面不可见
                    
    #             pos1 = self.compute_position(src, time)
    #             pos2 = self.compute_position(dst, time)
                
    #             # 计算距离
    #             distance = float(np.linalg.norm(pos2 - pos1))  # 转换为Python float
                
    #             # 根据轨道平面关系设置阈值
    #             if plane_diff == 1:  # 相邻轨道平面
    #                 is_visible = distance <= 7500.0  # 稍大于标称距离7155km
    #             else:  # 同一轨道平面
    #                 is_visible = distance <= 4000.0
                    
    #             return bool(is_visible)  # 显式转换为Python布尔值
                
    #         except Exception as e:
    #             self.logger.error(f"可见性检查时出错 ({src}-{dst}): {str(e)}")
    #             return False
                
    #     except Exception as e:
    #         self.logger.error(f"可见性检查主方法出错: {str(e)}")
    #         return False

    def _check_visibility(self, src: str, dst: str, time: float) -> bool:
        """检查源节点和目标节点之间的可见性"""
        try:
            # 如果包含地面站
            if src.startswith('station_') or dst.startswith('station_'):
                station_id = src if src.startswith('station_') else dst
                sat_id = dst if dst.startswith('satellite_') else src
                # 地面站与卫星的可见性判断
                return self.check_ground_station_visibility(station_id, sat_id, time)
                
            # 卫星间可见性检查 - 只考虑同轨道内相邻卫星
            src_orbit, src_num = self._parse_satellite_id(src)
            dst_orbit, dst_num = self._parse_satellite_id(dst)
            
            # 只有同轨道的卫星才可能可见
            if src_orbit != dst_orbit:
                return False
                
            # 检查是否相邻或首尾相连
            sat_diff = abs(src_num - dst_num)
            return sat_diff == 1 or sat_diff == 10  # 11颗卫星时，1和11是相连的
            
        except Exception as e:
            self.logger.error(f"可见性检查出错 ({src}-{dst}): {str(e)}")
            return False
        
    def _parse_satellite_id(self, sat_id: str) -> Tuple[int, int]:
        """
        解析卫星ID获取轨道号和卫星序号
        Args:
            sat_id: 卫星ID (格式: Iridium 1-1 或 satellite_1-1)
        Returns:
            Tuple[int, int]: (轨道号, 卫星序号)
        """
        try:
            if sat_id.startswith('Iridium'):
                # Iridium 1-1 格式
                parts = sat_id.split()[-1].split('-')
            elif sat_id.startswith('satellite_'):
                # satellite_1-1 格式
                parts = sat_id.split('_')[1].split('-')
            else:
                return (0, 0)
                
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            return (0, 0)
            
        except Exception as e:
            self.logger.error(f"解析卫星ID出错 ({sat_id}): {str(e)}")
            return (0, 0)
        
    def compute_doppler_shift(self, sat1: str, sat2: str, 
                            time: float, frequency: float) -> float:
        """
        计算多普勒频移
        Args:
            sat1: 发送卫星名称
            sat2: 接收卫星名称
            time: 时间戳
            frequency: 载波频率(Hz)
        Returns:
            频移量(Hz)
        """
        dt = 0.1  # 时间差分间隔(s)
        
        # 计算t和t+dt时刻的位置
        pos1_t = self.compute_position(sat1, time)
        pos2_t = self.compute_position(sat2, time)
        pos1_dt = self.compute_position(sat1, time + dt)
        pos2_dt = self.compute_position(sat2, time + dt)
        
        # 计算相对速度
        vel1 = (pos1_dt - pos1_t) / dt
        vel2 = (pos2_dt - pos2_t) / dt
        rel_vel = vel2 - vel1
        
        # 计算视线方向
        los = pos2_t - pos1_t
        los_unit = los / np.linalg.norm(los)
        
        # 计算径向速度
        radial_vel = np.dot(rel_vel, los_unit)
        
        # 计算多普勒频移
        c = 299792.458  # 光速(km/s)
        doppler_shift = frequency * radial_vel / c
        
        return doppler_shift

    def get_orbit_plane(self, sat_name: str) -> np.ndarray:
        """
        计算卫星轨道平面的法向量
        Args:
            sat_name: 卫星名称
        Returns:
            轨道平面法向量
        """
        # 计算三个时间点的位置
        t0 = self.ts.now()
        positions = []
        for dt in [0, 10, 20]:  # 取20分钟内的三个点
            t = self.ts.from_datetime(t0.utc_datetime() + timedelta(minutes=dt))
            geocentric = self.satellites[sat_name].at(t)
            positions.append(geocentric.position.km)
            
        # 使用叉积计算轨道平面法向量
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)
    
    def check_ground_station_visibility(self, station_id: str, sat_id: str, time: float, min_elevation: float = 5.0) -> bool:
        """
        检查地面站和卫星之间是否可见
        Args:
            station_id: 地面站ID
            sat_id: 卫星ID
            time: 时间戳
            min_elevation: 最小仰角(度)
        Returns:
            bool: 是否可见
        """
        try:
            # 获取地面站位置（示例位置，根据实际情况设置）
            station_positions = {
                'station_0': (0.0, 0.0, 0.0),      # 赤道0度
                'station_1': (0.0, 120.0, 0.0),    # 赤道120度
                'station_2': (0.0, -120.0, 0.0)    # 赤道-120度
            }
            
            # 获取卫星轨道和编号
            orbit_num, sat_num = self._parse_satellite_id(sat_id)
            
            # 检查地面站是否负责该轨道
            station_num = int(station_id.split('_')[1])
            responsible_orbits = [station_num * 2, station_num * 2 + 1]
            if (orbit_num - 1) not in responsible_orbits:
                return False
                
            # 放宽可见性条件
            min_elevation = 0.0  # 降低最小仰角要求
            max_range = 6000.0  # 增加最大可见距离(km)
            
            # 获取卫星位置
            sat_pos = self.compute_position(sat_id, time)
            station_pos = self._geodetic_to_ecef(*station_positions[station_id])
            
            # 计算距离和仰角
            range_vector = sat_pos - station_pos
            distance = np.linalg.norm(range_vector)
            
            if distance > max_range:
                return False
                
            # 计算仰角
            up_vector = station_pos / np.linalg.norm(station_pos)
            elevation = np.degrees(np.arcsin(np.dot(range_vector/distance, up_vector)))
            
            self.logger.debug(f"地面站 {station_id} -> 卫星 {sat_id}:")
            self.logger.debug(f"  距离: {distance:.2f}km")
            self.logger.debug(f"  仰角: {elevation:.2f}度")
            
            return elevation >= min_elevation and distance <= max_range
            
        except Exception as e:
            self.logger.error(f"地面站可见性检查出错: {str(e)}")
            return False

    def _geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        将大地坐标（经纬度）转换为ECEF坐标
        Args:
            lat: 纬度(度)
            lon: 经度(度)
            alt: 高度(km)
        Returns:
            np.ndarray: ECEF坐标(x, y, z)
        """
        # WGS84椭球体参数
        a = 6378.137  # 长半轴(km)
        e2 = 0.006694379990141  # 第一偏心率平方
        
        # 转换为弧度
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # 计算卯酉圈曲率半径
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # 计算ECEF坐标
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        
        return np.array([x, y, z])