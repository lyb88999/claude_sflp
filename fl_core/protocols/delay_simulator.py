from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import heapq

@dataclass
class LinkState:
    """链路状态"""
    bandwidth: float  # Mbps
    base_delay: float  # ms
    jitter: float  # ms
    packet_loss: float  # 0-1
    queue_size: int  # packets
    
class QueuedPacket:
    """排队的数据包"""
    def __init__(self, packet_id: str, size: int, arrival_time: float,
                 expected_delivery: float):
        self.packet_id = packet_id
        self.size = size
        self.arrival_time = arrival_time
        self.expected_delivery = expected_delivery
        
    def __lt__(self, other):
        return self.expected_delivery < other.expected_delivery

class DelaySimulator:
    def __init__(self, network_model):
        """
        初始化延迟模拟器
        Args:
            network_model: 卫星网络模型
        """
        self.network_model = network_model
        self.link_states = {}  # (source, target) -> LinkState
        self.packet_queues = {}  # (source, target) -> List[QueuedPacket]
        self.delivery_heap = []  # 优先队列，按预期传输时间排序
        self.current_time = 0.0  # 当前模拟时间
        
    def set_link_state(self, source: str, target: str, state: LinkState):
        """
        设置链路状态
        Args:
            source: 源节点
            target: 目标节点
            state: 链路状态
        """
        link_key = (source, target)
        self.link_states[link_key] = state
        # 初始化或重置队列
        self.packet_queues[link_key] = []
        # 清理相关的传输堆
        self.delivery_heap = [(t, p, k) for t, p, k in self.delivery_heap 
                            if k != link_key]
        heapq.heapify(self.delivery_heap)
        
    # 在 DelaySimulator 类中修改 schedule_transmission 方法
    def schedule_transmission(self, source: str, target: str, 
                        packet_id: str, size: int) -> Optional[float]:
        """
        调度数据包传输
        Args:
            source: 源节点
            target: 目标节点
            packet_id: 数据包ID
            size: 数据包大小(bytes)
        Returns:
            预期到达时间，如果传输失败则返回None
        """
        # 获取当前链路状态
        link_key = (source, target)
        if link_key not in self.link_states:
            return None
            
        state = self.link_states[link_key]
        queue = self.packet_queues[link_key]
        
        # 检查队列是否已满
        if len(queue) >= state.queue_size:
            print(f"Queue overflow: {len(queue)} >= {state.queue_size}")
            return None
            
        # 检查是否丢包
        if np.random.random() < state.packet_loss:
            return None
            
        # 计算传播延迟
        distance = self._calculate_distance(source, target)
        propagation_delay = distance / 299792.458  # 光速传播(ms)
        
        # 计算传输延迟
        transmission_delay = (size * 8) / (state.bandwidth * 1e6)  # ms
        
        # 计算处理延迟（包括抖动）
        processing_delay = state.base_delay + np.random.normal(0, state.jitter)
        
        # 计算排队延迟
        queuing_delay = self._calculate_queuing_delay(queue, size, state.bandwidth)
        
        # 计算总延迟
        total_delay = (propagation_delay + transmission_delay + 
                    processing_delay + queuing_delay)
                    
        # 计算预期到达时间
        arrival_time = self.current_time
        expected_delivery = arrival_time + total_delay
        
        # 创建并添加数据包到队列
        packet = QueuedPacket(packet_id, size, arrival_time, expected_delivery)
        queue.append(packet)
        heapq.heappush(self.delivery_heap, (expected_delivery, packet, link_key))
        
        return expected_delivery
        
    def update_time(self, new_time: float):
        """
        更新模拟时间并处理已传输的数据包
        Args:
            new_time: 新的时间戳
        Returns:
            已传输的数据包列表
        """
        self.current_time = new_time
        delivered_packets = []
        
        # 处理所有已经到达的数据包
        while (self.delivery_heap and 
               self.delivery_heap[0][0] <= self.current_time):
            delivery_time, packet, link_key = heapq.heappop(self.delivery_heap)
            queue = self.packet_queues[link_key]
            
            # 从队列中移除数据包
            if packet in queue:
                queue.remove(packet)
                delivered_packets.append((packet.packet_id, 
                                       link_key[0], link_key[1], 
                                       delivery_time))
                
        return delivered_packets
        
    def get_current_delays(self) -> Dict[Tuple[str, str], float]:
        """获取当前所有链路的预期延迟"""
        delays = {}
        for link_key, state in self.link_states.items():
            source, target = link_key
            # 计算基础延迟
            distance = self._calculate_distance(source, target)
            propagation_delay = distance / 299792.458
            base_delay = state.base_delay
            
            # 考虑当前队列状态
            queue = self.packet_queues[link_key]
            queuing_delay = 0.0
            if queue:
                last_packet = queue[-1]
                queuing_delay = last_packet.expected_delivery - self.current_time
                
            total_delay = propagation_delay + base_delay + queuing_delay
            delays[link_key] = total_delay
            
        return delays
        
    def _calculate_distance(self, source: str, target: str) -> float:
        """计算两个节点之间的距离(km)"""
        pos1 = self.network_model.compute_position(source, self.current_time)
        pos2 = self.network_model.compute_position(target, self.current_time)
        return np.linalg.norm(pos2 - pos1)
        
    def _calculate_queuing_delay(self, queue: List[QueuedPacket],
                               packet_size: int,
                               bandwidth: float) -> float:
        """计算排队延迟(ms)"""
        if not queue:
            return 0.0
            
        # 计算队列中所有数据包的传输时间
        total_size = sum(p.size for p in queue)
        transmission_time = (total_size * 8) / (bandwidth * 1e6)
        
        # 如果队列不为空，新数据包需要等待前面的数据包传输完成
        last_delivery = queue[-1].expected_delivery
        queuing_delay = max(0, last_delivery - self.current_time)
        
        return queuing_delay + transmission_time
        
    def get_queue_status(self) -> Dict[Tuple[str, str], Dict]:
        """获取所有链路的队列状态"""
        status = {}
        for link_key, queue in self.packet_queues.items():
            state = self.link_states[link_key]
            status[link_key] = {
                'queue_length': len(queue),
                'queue_capacity': state.queue_size,
                'utilization': len(queue) / state.queue_size if state.queue_size > 0 else 0,
                'pending_bytes': sum(p.size for p in queue)
            }
        return status
        
    def estimate_throughput(self, source: str, target: str,
                          window: float = 1000.0) -> float:
        """
        估计链路吞吐量(Mbps)
        Args:
            source: 源节点
            target: 目标节点
            window: 时间窗口(ms)
        Returns:
            估计的吞吐量
        """
        link_key = (source, target)
        if link_key not in self.link_states:
            return 0.0
            
        queue = self.packet_queues[link_key]
        state = self.link_states[link_key]
        
        # 计算时间窗口内的数据量
        window_start = self.current_time - window
        window_packets = [p for p in queue 
                         if p.arrival_time >= window_start]
                         
        total_bytes = sum(p.size for p in window_packets)
        throughput = (total_bytes * 8) / (window * 1000)  # 转换为Mbps
        
        # 考虑丢包率
        throughput *= (1 - state.packet_loss)
        
        return min(throughput, state.bandwidth)