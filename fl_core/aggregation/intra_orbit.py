from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

@dataclass
class AggregationConfig:
    """聚合配置"""
    min_updates: int = 2  # 最小更新数量
    max_staleness: float = 300.0  # 最大容忍延迟(秒)
    timeout: float = 600.0  # 聚合超时时间(秒)
    weighted_average: bool = True  # 是否使用加权平均
    timeout_strategy: str = 'wait'  # 超时策略: 'wait' 或 'proceed'

class IntraOrbitAggregator:
    def __init__(self, config: AggregationConfig):
        """
        初始化轨道内聚合器
        Args:
            config: 聚合配置
        """
        self.config = config
        self.pending_updates = defaultdict(dict)  # round -> {client_id: update}
        self.update_timestamps = defaultdict(dict)  # round -> {client_id: timestamp}
        self.client_weights = {}  # client_id -> weight
        self.aggregation_state = {}  # round -> state
        
    def add_client(self, client_id: str, weight: float = 1.0):
        """
        添加客户端
        Args:
            client_id: 客户端ID
            weight: 客户端权重
        """
        self.client_weights[client_id] = weight
        
    def remove_client(self, client_id: str):
        """移除客户端"""
        self.client_weights.pop(client_id, None)
        
    def receive_update(self, client_id: str, round_number: int,
                      model_update: Dict[str, torch.Tensor],
                      timestamp: float) -> bool:
        """
        接收客户端更新
        Args:
            client_id: 客户端ID
            round_number: 轮次
            model_update: 模型更新
            timestamp: 时间戳
        Returns:
            是否成功接收更新
        """
        if client_id not in self.client_weights:
            return False
            
        # 检查更新是否过期
        current_time = datetime.now().timestamp()
        if current_time - timestamp > self.config.max_staleness:
            return False
            
        # 存储更新
        self.pending_updates[round_number][client_id] = model_update
        self.update_timestamps[round_number][client_id] = timestamp
        
        # 检查是否可以进行聚合
        if self._should_aggregate(round_number):
            self._aggregate_round(round_number)
            
        return True
        
    def get_aggregated_update(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取聚合后的更新
        Args:
            round_number: 轮次
        Returns:
            聚合后的模型更新
        """
        if round_number not in self.aggregation_state:
            return None
            
        state = self.aggregation_state[round_number]
        if not state.get('completed', False):
            return None
            
        return state.get('result')
        
    def _should_aggregate(self, round_number: int) -> bool:
        """检查是否应该进行聚合"""
        updates = self.pending_updates[round_number]
        if len(updates) < self.config.min_updates:
            return False
            
        current_time = datetime.now().timestamp()
        earliest_update = min(self.update_timestamps[round_number].values())
        
        # 如果达到最小更新数量就进行聚合
        if len(updates) >= self.config.min_updates:
            return True
            
        # 检查是否超时
        if current_time - earliest_update > self.config.timeout:
            return True
            
        return False
        
    def _aggregate_round(self, round_number: int):
        """
        聚合指定轮次的更新
        Args:
            round_number: 轮次
        """
        updates = self.pending_updates[round_number]
        timestamps = self.update_timestamps[round_number]
        
        # 计算权重
        weights = {}
        total_weight = 0.0
        
        for client_id in updates.keys():
            if self.config.weighted_average:
                # 基于数据量的权重
                base_weight = self.client_weights[client_id]
                
                # 考虑时间衰减
                staleness = datetime.now().timestamp() - timestamps[client_id]
                time_factor = np.exp(-staleness / self.config.max_staleness)
                
                weight = base_weight * time_factor
            else:
                weight = 1.0
                
            weights[client_id] = weight
            total_weight += weight
            
        # 归一化权重
        for client_id in weights:
            weights[client_id] /= total_weight
            
        # 聚合更新
        aggregated_update = {}
        for param_name in next(iter(updates.values())).keys():
            weighted_sum = None
            
            for client_id, update in updates.items():
                weighted_update = update[param_name] * weights[client_id]
                
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
            'timestamp': datetime.now().timestamp()
        }
        
        # 清理已完成的更新
        self.pending_updates.pop(round_number, None)
        self.update_timestamps.pop(round_number, None)
        
    def get_aggregation_stats(self, round_number: int) -> Dict:
        """
        获取聚合统计信息
        Args:
            round_number: 轮次
        Returns:
            聚合统计信息
        """
        if round_number not in self.aggregation_state:
            return {}
            
        state = self.aggregation_state[round_number]
        current_time = datetime.now().timestamp()
        
        return {
            'completed': state['completed'],
            'num_participants': len(state['participants']),
            'total_clients': len(self.client_weights),
            'aggregation_time': state.get('timestamp', current_time) - 
                              min(self.update_timestamps.get(round_number, {}).values()),
            'weights': state['weights']
        }
        
    def clear_round(self, round_number: int):
        """清理指定轮次的数据"""
        self.pending_updates.pop(round_number, None)
        self.update_timestamps.pop(round_number, None)
        self.aggregation_state.pop(round_number, None)