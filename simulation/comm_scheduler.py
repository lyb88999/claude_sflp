from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from queue import PriorityQueue
import heapq

@dataclass
class CommunicationWindow:
    """通信窗口类"""
    start_time: float  # 开始时间
    end_time: float    # 结束时间
    source: str        # 源卫星
    target: str        # 目标卫星
    quality: float     # 链路质量（0-1）
    bandwidth: float   # 可用带宽（Mbps）

@dataclass
class CommunicationTask:
    """通信任务类"""
    task_id: str
    source: str
    target: str
    data_size: float  # MB
    priority: int     # 1-10，10最高
    deadline: float   # 截止时间戳
    
class CommunicationScheduler:
    def __init__(self, network_model, energy_model=None):
        """
        初始化通信调度器
        Args:
            network_model: 卫星网络模型实例
            energy_model: 能源模型实例（可选）
        """
        if network_model is None:
            raise ValueError("Network model cannot be None")
            
        self.network_model = network_model
        self.energy_model = energy_model
        self.windows_cache = {}  # 窗口缓存
        self.current_schedule = {}  # 当前调度计划
        self.task_queue = PriorityQueue()  # 任务优先级队列
        
        print(f"通信调度器初始化完成，已加载 {len(self.network_model.satellites)} 颗卫星")
        
    def predict_communication_windows(self, 
                                   start_time: float,
                                   end_time: float,
                                   sat_pair: Tuple[str, str]) -> List[CommunicationWindow]:
        """
        预测指定时间段内的通信窗口
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            sat_pair: (源卫星，目标卫星)元组
        Returns:
            通信窗口列表
        """
        cache_key = (start_time, end_time, sat_pair)
        if cache_key in self.windows_cache:
            return self.windows_cache[cache_key]
            
        windows = []
        current_time = start_time
        step = 60  # 60秒采样间隔
        
        window_start = None
        prev_visible = False
        
        while current_time <= end_time:
            is_visible = self.network_model.check_visibility(
                sat_pair[0], sat_pair[1], current_time
            )
            
            # 检测窗口开始
            if is_visible and not prev_visible:
                window_start = current_time
                
            # 检测窗口结束
            elif not is_visible and prev_visible and window_start is not None:
                quality = self._evaluate_link_quality(
                    sat_pair[0], sat_pair[1], 
                    window_start, current_time
                )
                bandwidth = self._estimate_bandwidth(
                    sat_pair[0], sat_pair[1],
                    window_start, current_time
                )
                
                windows.append(CommunicationWindow(
                    start_time=window_start,
                    end_time=current_time,
                    source=sat_pair[0],
                    target=sat_pair[1],
                    quality=quality,
                    bandwidth=bandwidth
                ))
                window_start = None
                
            prev_visible = is_visible
            current_time += step
            
        # 处理最后一个窗口
        if window_start is not None:
            quality = self._evaluate_link_quality(
                sat_pair[0], sat_pair[1],
                window_start, current_time
            )
            bandwidth = self._estimate_bandwidth(
                sat_pair[0], sat_pair[1],
                window_start, current_time
            )
            windows.append(CommunicationWindow(
                start_time=window_start,
                end_time=current_time,
                source=sat_pair[0],
                target=sat_pair[1],
                quality=quality,
                bandwidth=bandwidth
            ))
            
        self.windows_cache[cache_key] = windows
        return windows
        
    def _evaluate_link_quality(self, sat1: str, sat2: str, 
                             start_time: float, end_time: float) -> float:
        """评估链路质量"""
        # 采样点数
        num_samples = 5
        times = np.linspace(start_time, end_time, num_samples)
        
        # 计算平均距离和多普勒频移
        distances = []
        doppler_shifts = []
        freq = 2.4e9  # 2.4GHz载波
        
        for t in times:
            pos1 = self.network_model.compute_position(sat1, t)
            pos2 = self.network_model.compute_position(sat2, t)
            distance = np.linalg.norm(pos1 - pos2)
            doppler = self.network_model.compute_doppler_shift(sat1, sat2, t, freq)
            
            distances.append(distance)
            doppler_shifts.append(abs(doppler))
            
        # 标准化指标
        avg_distance = np.mean(distances)
        max_doppler = max(doppler_shifts)
        
        # 距离衰减权重
        distance_quality = np.exp(-avg_distance / 1000)  # 1000km特征距离
        
        # 多普勒影响权重
        doppler_quality = np.exp(-max_doppler / 1000)  # 1kHz特征频移
        
        # 综合评分
        quality = 0.7 * distance_quality + 0.3 * doppler_quality
        return min(max(quality, 0), 1)  # 限制在[0,1]范围
        
    def _estimate_bandwidth(self, sat1: str, sat2: str,
                          start_time: float, end_time: float) -> float:
        """估计可用带宽"""
        # 基础带宽（Mbps）
        base_bandwidth = 100.0
        
        # 获取链路质量
        quality = self._evaluate_link_quality(sat1, sat2, start_time, end_time)
        
        # 考虑能量约束
        if self.energy_model:
            energy_factor = self.energy_model.get_transmission_capacity(sat1)
            return base_bandwidth * quality * energy_factor
        
        return base_bandwidth * quality
        
    def add_task(self, task: CommunicationTask):
        """添加通信任务到队列"""
        # 优先级反转（优先级队列默认最小优先）
        self.task_queue.put((-task.priority, task.deadline, task))
        
    def schedule_tasks(self, current_time: float, 
                      horizon: float = 3600) -> Dict[str, List[Tuple[float, float]]]:
        """
        调度任务
        Args:
            current_time: 当前时间戳
            horizon: 调度时间范围（秒）
        Returns:
            任务调度方案，格式：{task_id: [(start_time, end_time), ...]}
        """
        schedule = {}
        available_windows = {}
        
        # 获取所有待调度任务
        tasks = []
        while not self.task_queue.empty():
            _, _, task = self.task_queue.get()
            if task.deadline > current_time:
                tasks.append(task)
                
        # 按优先级排序
        tasks.sort(key=lambda x: (-x.priority, x.deadline))
        
        for task in tasks:
            # 获取可用通信窗口
            if (task.source, task.target) not in available_windows:
                windows = self.predict_communication_windows(
                    current_time,
                    current_time + horizon,
                    (task.source, task.target)
                )
                available_windows[(task.source, task.target)] = windows
            
            windows = available_windows[(task.source, task.target)]
            task_schedule = self._schedule_single_task(task, windows, current_time)
            
            if task_schedule:
                schedule[task.task_id] = task_schedule
            else:
                # 任务无法在截止时间前完成，重新加入队列
                self.task_queue.put((-task.priority, task.deadline, task))
                
        return schedule
        
    def _schedule_single_task(self, task: CommunicationTask,
                            windows: List[CommunicationWindow],
                            current_time: float) -> Optional[List[Tuple[float, float]]]:
        """为单个任务分配通信窗口"""
        remaining_data = task.data_size
        schedule = []
        
        for window in windows:
            if window.start_time >= task.deadline:
                break
                
            if window.start_time < current_time:
                continue
                
            # 计算此窗口可传输的数据量
            duration = window.end_time - window.start_time
            transferable = window.bandwidth * duration  # MB
            
            if transferable >= remaining_data:
                # 计算实际需要的传输时间
                actual_duration = remaining_data / window.bandwidth
                schedule.append((
                    window.start_time,
                    window.start_time + actual_duration
                ))
                return schedule
            else:
                schedule.append((window.start_time, window.end_time))
                remaining_data -= transferable
                
        # 如果无法完成全部传输，返回None
        if remaining_data > 0:
            return None
            
        return schedule