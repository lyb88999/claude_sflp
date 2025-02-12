import logging
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
        self.logger = logging.getLogger(__name__)  # 添加logger

        
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
        """接收客户端更新"""
        try:
            # 验证更新内容
            if not model_update:
                self.logger.warning(f"客户端 {client_id} 提供了空的更新")
                return False
                
            # 检查参数是否为tensor
            for param_name, param in model_update.items():
                if not isinstance(param, torch.Tensor):
                    self.logger.error(f"客户端 {client_id} 的参数 {param_name} 不是tensor")
                    return False
                    
            # 存储更新
            self.pending_updates[round_number][client_id] = {
                name: param.clone().detach() 
                for name, param in model_update.items()
            }
            self.update_timestamps[round_number][client_id] = timestamp
            
            self.logger.info(f"成功接收客户端 {client_id} 的更新，轮次 {round_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"接收更新时出错: {str(e)}")
            return False
        
    def get_aggregated_update(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """获取聚合后的更新"""
        self.logger.info(f"尝试获取轮次 {round_number} 的聚合结果")
        
        if round_number not in self.pending_updates:
            self.logger.warning(f"轮次 {round_number} 没有待处理的更新")
            return None
            
        updates = self.pending_updates[round_number]
        if len(updates) < self.config.min_updates:
            self.logger.warning(f"更新数量不足: {len(updates)} < {self.config.min_updates}")
            return None
            
        try:
            # 计算权重
            weights = {}
            total_weight = 0.0
            
            for client_id in updates.keys():
                if self.config.weighted_average:
                    # 基于时间戳的权重计算
                    current_time = datetime.now().timestamp()
                    staleness = current_time - self.update_timestamps[round_number][client_id]
                    time_factor = np.exp(-staleness / self.config.max_staleness)
                    weight = time_factor
                else:
                    weight = 1.0
                    
                weights[client_id] = weight
                total_weight += weight
            
            # 归一化权重
            if total_weight > 0:
                for client_id in weights:
                    weights[client_id] /= total_weight
            else:
                self.logger.error("权重总和为0，无法进行聚合")
                return None
                
            # 聚合更新
            aggregated_update = {}
            self.logger.info(f"开始聚合 {len(updates)} 个更新")
            
            # 获取参数名列表
            param_names = next(iter(updates.values())).keys()
            
            for param_name in param_names:
                weighted_sum = None
                param_updates_available = True
                
                for client_id, update in updates.items():
                    if param_name not in update:
                        self.logger.error(f"客户端 {client_id} 缺少参数 {param_name}")
                        param_updates_available = False
                        break
                        
                    try:
                        weighted_update = update[param_name] * weights[client_id]
                        if weighted_sum is None:
                            weighted_sum = weighted_update
                        else:
                            weighted_sum += weighted_update
                    except Exception as e:
                        self.logger.error(f"处理参数 {param_name} 时出错: {str(e)}")
                        param_updates_available = False
                        break
                
                if not param_updates_available:
                    self.logger.error("参数聚合失败")
                    return None
                    
                aggregated_update[param_name] = weighted_sum
            
            # 清理已完成的更新
            self.pending_updates.pop(round_number, None)
            self.update_timestamps.pop(round_number, None)
            
            self.logger.info("聚合完成")
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"聚合过程出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
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