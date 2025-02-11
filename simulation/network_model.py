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
        """
        计算指定时间的卫星位置
        Args:
            sat_name: 卫星名称
            time: 时间戳(UTC)
        Returns:
            ECEF坐标系中的位置向量(x, y, z)
        """
        if sat_name not in self.satellites:
            raise ValueError(f"卫星 {sat_name} 未找到")
            
        # 获取卫星编号(1-6)
        sat_num = int(sat_name.split()[-1])
        
        # 铱星星座参数
        orbit_radius = 7155.0  # 轨道半径(km)
        inclination = np.radians(86.4)  # 轨道倾角
        raan = np.radians((sat_num - 1) * 60)  # 升交点赤经，每个轨道平面间隔60度
        
        # 计算轨道周期
        mu = 3.986004418e5  # 地球引力常数(km³/s²)
        orbit_period = 2 * np.pi * np.sqrt(orbit_radius**3 / mu)  # 轨道周期(秒)
        
        # 计算当前时刻的真近点角
        mean_motion = 2 * np.pi / orbit_period  # 平均角速度(rad/s)
        mean_anomaly = mean_motion * (time % orbit_period)
        true_anomaly = mean_anomaly  # 简化处理，假设圆轨道
        
        # 计算轨道平面中的位置
        x_orbit = orbit_radius * np.cos(true_anomaly)
        y_orbit = orbit_radius * np.sin(true_anomaly)
        
        # 执行坐标变换
        # 1. 绕z轴旋转(升交点赤经)
        x1 = x_orbit * np.cos(raan) - y_orbit * np.sin(raan)
        y1 = x_orbit * np.sin(raan) + y_orbit * np.cos(raan)
        z1 = 0
        
        # 2. 绕x轴旋转(轨道倾角)
        x2 = x1
        y2 = y1 * np.cos(inclination)
        z2 = y1 * np.sin(inclination)
        
        position = np.array([x2, y2, z2])
        return position

    
    def check_visibility(self, src: str, dst: str, time: float) -> bool:
        """
        检查源节点和目标节点之间是否可见
        Args:
            src: 源节点ID(可以是卫星或地面站)
            dst: 目标节点ID(可以是卫星或地面站)
            time: 时间戳
        Returns:
            bool: 是否可见
        """
        try:
            # 转换卫星ID格式
            def convert_sat_id(sat_id: str) -> str:
                if sat_id.startswith('satellite_'):
                    num = sat_id.split('_')[1]
                    return f"Iridium {num}"
                return sat_id

            # 如果包含地面站
            if src.startswith('station_') or dst.startswith('station_'):
                station_id = src if src.startswith('station_') else dst
                sat_id = dst if dst.startswith('satellite_') else src
                return self.check_ground_station_visibility(station_id, convert_sat_id(sat_id), time)
                
            # 卫星间可见性检查
            src_id = convert_sat_id(src)
            dst_id = convert_sat_id(dst)
            
            try:
                sat1_num = int(src_id.split()[-1])
                sat2_num = int(dst_id.split()[-1])
                
                # 计算轨道平面差异
                plane_diff = abs(sat1_num - sat2_num)
                if plane_diff > 3:  # 如果相差超过半个星座，取补值
                    plane_diff = 6 - plane_diff
                    
                # 检查是否为相邻轨道平面或同一轨道平面
                if plane_diff > 1:
                    return False  # 非相邻轨道平面不可见
                    
                pos1 = self.compute_position(src, time)
                pos2 = self.compute_position(dst, time)
                
                # 计算距离
                distance = float(np.linalg.norm(pos2 - pos1))  # 转换为Python float
                
                # 根据轨道平面关系设置阈值
                if plane_diff == 1:  # 相邻轨道平面
                    is_visible = distance <= 7500.0  # 稍大于标称距离7155km
                else:  # 同一轨道平面
                    is_visible = distance <= 4000.0
                    
                return bool(is_visible)  # 显式转换为Python布尔值
                
            except Exception as e:
                self.logger.error(f"可见性检查时出错 ({src}-{dst}): {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"可见性检查主方法出错: {str(e)}")
            return False
        
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
    
    def check_ground_station_visibility(self, station_id: str, sat_id: str, 
                                      time: float, min_elevation: float = 10.0) -> bool:
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
            if station_id not in self.ground_stations or sat_id not in self.satellites:
                return False

            # 转换时间戳为datetime对象
            dt = datetime.fromtimestamp(time, tz=timezone.utc)
            t = self.ts.from_datetime(dt)

            # 获取地面站位置
            lat, lon, height = self.ground_stations[station_id]
            station = wgs84.latlon(lat, lon, height)
            
            # 获取卫星位置
            satellite = self.satellites[sat_id]
            
            # 计算卫星相对于地面站的方位角和仰角
            difference = satellite - station
            topocentric = difference.at(t)
            alt, az, distance = topocentric.altaz()
            
            # 检查是否满足最小仰角要求
            is_visible = alt.degrees >= min_elevation
            
            if is_visible:
                self.logger.debug(f"地面站 {station_id} 可见卫星 {sat_id}, "
                               f"仰角: {alt.degrees:.2f}°")
            
            return is_visible
            
        except Exception as e:
            self.logger.error(f"检查地面站可见性时出错: {str(e)}")
            return False