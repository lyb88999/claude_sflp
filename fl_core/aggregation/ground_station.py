import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import heapq

@dataclass
class GroundStationConfig:
    """地面站配置"""
    bandwidth_limit: float = 1000.0  # Mbps
    storage_limit: float = 10000.0  # MB
    priority_levels: int = 3
    batch_size: int = 10  # 批处理大小
    aggregation_interval: float = 60.0  # 聚合间隔(秒)
    min_updates: int = 2  # 最小更新数量
    max_staleness: float = 300.0  # 最大容忍延迟(秒)
    timeout: float = 600.0  # 聚合超时时间(秒)
    weighted_average: bool = True  # 是否使用加权平均

@dataclass
class OrbitUpdate:
    """轨道更新信息"""
    orbit_id: str
    round_number: int
    model_update: Dict[str, torch.Tensor]
    timestamp: float
    priority: int
    size: float  # MB
    num_clients: int

class GroundStationAggregator:
    def __init__(self, config: GroundStationConfig):
        """
        初始化地面站聚合器
        Args:
            config: 地面站配置
        """
        if isinstance(config, dict):
            config = GroundStationConfig(**config)
        self.config = config
        self.orbit_weights = {}  # orbit_id -> weight
        self.pending_updates = defaultdict(dict)  # round -> {orbit_id: OrbitUpdate}
        self.update_queue = []  # 优先级队列
        self.aggregation_state = {}  # round -> state
        self.bandwidth_usage = defaultdict(float)  # timestamp -> usage
        self.storage_usage = 0.0
        self.last_aggregation_time = datetime.now().timestamp()
        self.responsible_orbits = []  # 负责的轨道
        self.logger = logging.getLogger(__name__)
        
    def add_orbit(self, orbit_id: str, weight: float = 1.0):
        """添加轨道"""
        self.orbit_weights[orbit_id] = weight
        
    def remove_orbit(self, orbit_id: str):
        """移除轨道"""
        self.orbit_weights.pop(orbit_id, None)
        
    def receive_orbit_update(self, orbit_id: str, round_number: int,
                   model_update: Dict[str, torch.Tensor],
                   num_clients: int, priority: int = 1) -> bool:
        """接收轨道更新"""
        if orbit_id not in [str(x) for x in self.responsible_orbits]:
            self.logger.warning(f"轨道 {orbit_id} 不在负责范围内")
            return False
            
        # 计算更新大小（MB）
        size = sum(param.nelement() * param.element_size() 
                for param in model_update.values()) / (1024 * 1024)
                
        # 检查存储容量
        if self.storage_usage + size > self.config.storage_limit:
            self.logger.warning(f"存储不足: 当前{self.storage_usage:.2f}MB + 需要{size:.2f}MB > 限制{self.config.storage_limit}MB")
            return False
            
        # 检查带宽使用
        current_bandwidth = self._get_current_bandwidth_usage()
        required_bandwidth = size * 8  # 转换为Mbits
        if current_bandwidth + required_bandwidth > self.config.bandwidth_limit:
            self.logger.warning(f"带宽不足: 当前{current_bandwidth:.2f}Mbps + 需要{required_bandwidth:.2f}Mbps > 限制{self.config.bandwidth_limit}Mbps")
            return False
            
        # 更新资源使用
        self.storage_usage += size
        self._update_bandwidth_usage(size)
        
        # 创建更新对象
        update = OrbitUpdate(
            orbit_id=orbit_id,
            round_number=round_number,
            model_update=model_update,
            timestamp=datetime.now().timestamp(),
            priority=priority,
            size=size,
            num_clients=num_clients
        )
        
        # 存储更新
        self.pending_updates[round_number][orbit_id] = update
        self.logger.info(f"成功存储轨道 {orbit_id} 的更新，待处理更新数: {len(self.pending_updates[round_number])}")

        heapq.heappush(self.update_queue, (-priority, update.timestamp, update))
        
        return True
        
    def _process_pending_updates(self):
        """处理待处理的更新"""
        current_time = datetime.now().timestamp()
        
        # 检查是否需要进行聚合
        if (current_time - self.last_aggregation_time < 
            self.config.aggregation_interval):
            return
            
        # 按批次处理更新
        processed_size = 0
        while self.update_queue:
            # 检查带宽限制
            if self._get_current_bandwidth_usage() >= self.config.bandwidth_limit:
                break
                
            # 获取最高优先级的更新
            _, _, update = heapq.heappop(self.update_queue)
            
            # 检查批次大小限制
            if processed_size + update.size > self.config.batch_size * 1024:
                break
                
            # 存储更新
            self.pending_updates[update.round_number][update.orbit_id] = update
            processed_size += update.size
            
            # 更新带宽使用
            self._update_bandwidth_usage(update.size)
            
            # 检查是否可以聚合
            if self._should_aggregate(update.round_number):
                self._aggregate_round(update.round_number)
                
        self.last_aggregation_time = current_time
        
    def _get_current_bandwidth_usage(self) -> float:
        """获取当前带宽使用"""
        current_time = datetime.now().timestamp()
        # 清理过期的带宽记录
        self.bandwidth_usage = defaultdict(float,
            {t: usage for t, usage in self.bandwidth_usage.items()
             if current_time - t <= 1.0})  # 只保留最近1秒的记录
             
        return sum(self.bandwidth_usage.values())
        
    def _update_bandwidth_usage(self, size: float):
        """更新带宽使用记录"""
        current_time = datetime.now().timestamp()
        self.bandwidth_usage[current_time] += size * 8  # 转换为Mbits
        
    def _should_aggregate(self, round_number: int) -> bool:
        """检查是否应该进行聚合"""
        updates = self.pending_updates[round_number]
        if not updates:
            return False
            
        # 检查是否所有轨道都已更新
        return len(updates) == len(self.orbit_weights)
        
    def _aggregate_round(self, round_number: int):
        """
        聚合指定轮次的更新
        Args:
            round_number: 轮次
        """
        updates = self.pending_updates[round_number]
        
        # 计算权重
        weights = {}
        total_weight = 0.0
        
        for orbit_id, update in updates.items():
            # 基于客户端数量和轨道权重的加权
            weight = self.orbit_weights[orbit_id] * update.num_clients
            weights[orbit_id] = weight
            total_weight += weight
            
        # 归一化权重
        for orbit_id in weights:
            weights[orbit_id] /= total_weight
            
        # 聚合更新
        aggregated_update = {}
        for param_name in next(iter(updates.values())).model_update.keys():
            weighted_sum = None
            
            for orbit_id, update in updates.items():
                weighted_update = (update.model_update[param_name] * 
                                 weights[orbit_id])
                
                if weighted_sum is None:
                    weighted_sum = weighted_update
                else:
                    weighted_sum += weighted_update
                    
            aggregated_update[param_name] = weighted_sum
            
        # 更新聚合状态
        self.aggregation_state[round_number] = {
            'completed': True,
            'result': aggregated_update,
            'participants': list(updates.keys()),
            'weights': weights,
            'timestamp': datetime.now().timestamp(),
            'total_clients': sum(update.num_clients for update in updates.values())
        }
        
        # 更新存储使用
        for update in updates.values():
            self.storage_usage -= update.size
            
        # 清理已完成的更新
        self.pending_updates.pop(round_number, None)
        
    # def get_aggregated_update(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
    #     """获取聚合后的更新"""
    #     self.logger.info(f"尝试获取轮次 {round_number} 的聚合结果")

    #     if round_number not in self.pending_updates:
    #         self.logger.warning(f"轮次 {round_number} 没有待处理的更新")
    #         return None
            
    #     updates = self.pending_updates[round_number]
    #     self.logger.info(f"当前轮次 {round_number} 有 {len(updates)} 个待处理更新")
        
    #     if not updates:
    #         self.logger.warning("没有可用的更新")
    #         return None

    #     try:
    #         # 聚合更新
    #         aggregated_update = {}
    #         update_weights = {}
    #         total_clients = 0

    #         # 为每个轨道分配权重
    #         for orbit_id, update in updates.items():
    #             update_weights[orbit_id] = update.num_clients
    #             total_clients += update.num_clients

    #         self.logger.info(f"总计 {total_clients} 个客户端参与更新")

    #         # 归一化权重
    #         for orbit_id in update_weights:
    #             update_weights[orbit_id] /= total_clients

    #         # 获取所有参数名
    #         first_update = next(iter(updates.values())).model_update
    #         param_names = first_update.keys()

    #         # 聚合每个参数
    #         for param_name in param_names:
    #             weighted_sum = None
    #             for orbit_id, update in updates.items():
    #                 weight = update_weights[orbit_id]
    #                 param = update.model_update[param_name]
                    
    #                 if weighted_sum is None:
    #                     weighted_sum = param * weight
    #                 else:
    #                     weighted_sum += param * weight

    #             aggregated_update[param_name] = weighted_sum

    #         # 清理已处理的更新
    #         self.pending_updates.pop(round_number, None)
    #         self.logger.info("聚合完成")
    #         return aggregated_update

    #     except Exception as e:
    #         self.logger.error(f"聚合过程出错: {str(e)}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
    #         return None

    def get_aggregated_update(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """获取聚合后的更新"""
        self.logger.info(f"尝试获取轮次 {round_number} 的聚合结果")
        
        if round_number not in self.pending_updates:
            self.logger.warning(f"轮次 {round_number} 没有待处理的更新")
            return None
            
        updates = self.pending_updates[round_number]
        self.logger.info(f"当前轮次 {round_number} 有 {len(updates)} 个待处理更新")
        
        try:
            if len(updates) < self.config.min_updates:
                self.logger.warning(f"更新数量不足: {len(updates)} < {self.config.min_updates}")
                return None
                
            # 详细记录每个更新的状态
            for update_id, update in updates.items():
                self.logger.info(f"更新 {update_id}: {update.size:.2f}MB, "
                               f"{update.num_clients} 个客户端")
                
            # 聚合参数
            first_update = next(iter(updates.values())).model_update
            aggregated_update = {}
            
            for param_name, param in first_update.items():
                # 初始化为零张量
                aggregated_param = torch.zeros_like(param)
                total_weight = 0.0
                
                for update in updates.values():
                    if param_name not in update.model_update:
                        self.logger.error(f"参数 {param_name} 在某些更新中缺失")
                        continue
                        
                    weight = update.num_clients
                    total_weight += weight
                    aggregated_param += update.model_update[param_name] * weight
                
                if total_weight > 0:
                    aggregated_update[param_name] = aggregated_param / total_weight
                else:
                    self.logger.error(f"参数 {param_name} 的总权重为0")
                    return None
            
            # 验证聚合结果
            for name, param in aggregated_update.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self.logger.error(f"聚合结果 {name} 包含无效值")
                    return None
            
            # 清理已处理的更新
            self.pending_updates.pop(round_number, None)
            self.logger.info("聚合成功完成")
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"聚合过程出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def get_aggregation_stats(self) -> Dict:
        """获取聚合统计信息"""
        current_time = datetime.now().timestamp()
        
        return {
            'bandwidth_usage': self._get_current_bandwidth_usage(),
            'storage_usage': self.storage_usage,
            'pending_updates': len(self.update_queue),
            'completed_rounds': len(self.aggregation_state),
            'last_aggregation_time': current_time - self.last_aggregation_time,
            'active_orbits': len(self.orbit_weights)
        }
        
    def clear_round(self, round_number: int):
        """清理指定轮次的数据"""
        # 更新存储使用
        if round_number in self.pending_updates:
            for update in self.pending_updates[round_number].values():
                self.storage_usage -= update.size
                
        self.pending_updates.pop(round_number, None)
        self.aggregation_state.pop(round_number, None)