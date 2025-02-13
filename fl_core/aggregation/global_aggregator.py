import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import hashlib
import json

@dataclass
class GlobalConfig:
    """全局聚合配置"""
    min_ground_stations: int = 2  # 最小地面站数量
    consistency_threshold: float = 0.8  # 一致性阈值
    max_version_diff: int = 2  # 最大版本差异
    aggregation_timeout: float = 1800.0  # 聚合超时时间(秒)
    validation_required: bool = True  # 是否需要验证

@dataclass
class ModelVersion:
    """模型版本信息"""
    version: int
    parameters: Dict[str, torch.Tensor]
    hash_value: str
    timestamp: float
    metrics: Dict[str, float]

class GlobalAggregator:
    def __init__(self, config: GlobalConfig):
        """
        初始化全局聚合器
        Args:
            config: 全局配置
        """
        self.config = config
        self.ground_stations = {}  # station_id -> weight
        self.pending_updates = defaultdict(dict)  # round -> {station_id: update}
        self.model_versions = []  # 版本历史
        self.current_version = 0
        self.validation_results = defaultdict(dict)  # round -> {station_id: metrics}
        self.logger = logging.getLogger(__name__)

        
    def add_ground_station(self, station_id: str, weight: float = 1.0):
        """添加地面站"""
        self.ground_stations[station_id] = weight
        
    def remove_ground_station(self, station_id: str):
        """移除地面站"""
        self.ground_stations.pop(station_id, None)
        

    def receive_station_update(self, station_id: str, round_number: int,
                             model_update: Dict[str, torch.Tensor],
                             metrics: Dict[str, float],
                             base_version: int) -> bool:
        """
        接收地面站更新
        """
        try:
            self.logger.info(f"全局聚合器接收到地面站 {station_id} 的更新")
            
            if station_id not in self.ground_stations:
                self.logger.warning(f"未知地面站: {station_id}")
                return False
                
            # 存储更新
            self.pending_updates[round_number][station_id] = {
                'update': {k: v.clone() for k, v in model_update.items()},
                'metrics': metrics,
                'base_version': base_version,
                'timestamp': datetime.now().timestamp()
            }
            
            self.logger.info(f"成功存储地面站 {station_id} 的更新，当前轮次 {round_number} 共有 {len(self.pending_updates[round_number])} 个更新")
            
            # 检查是否可以进行聚合
            if self._should_aggregate(round_number):
                self._aggregate_round(round_number)
                
            return True
            
        except Exception as e:
            self.logger.error(f"接收地面站更新时出错: {str(e)}")
            return False
        
    def submit_validation_result(self, station_id: str, round_number: int,
                          metrics: Dict[str, float]) -> bool:
        """
        提交验证结果
        Args:
            station_id: 地面站ID
            round_number: 轮次
            metrics: 验证指标
        Returns:
            是否成功提交
        """
        if station_id not in self.ground_stations:
            return False
            
        self.validation_results[round_number][station_id] = metrics
        
        # 如果所有必需的验证结果都已收到，尝试进行聚合
        if len(self.validation_results[round_number]) >= self.config.min_ground_stations:
            if round_number in self.pending_updates and self._check_consistency(round_number):
                self._aggregate_round(round_number)
                
        return True
        
    def _should_aggregate(self, round_number: int) -> bool:
        """检查是否应该进行聚合"""
        updates = self.pending_updates[round_number]
        if len(updates) < self.config.min_ground_stations:
            return False
            
        # 检查是否超时
        current_time = datetime.now().timestamp()
        earliest_update = min(u['timestamp'] for u in updates.values())
        if current_time - earliest_update > self.config.aggregation_timeout:
            return True
            
        # 检查是否所有地面站都已更新
        return len(updates) == len(self.ground_stations)
        
    def _aggregate_round(self, round_number: int):
        """
        聚合指定轮次的更新
        Args:
            round_number: 轮次
        """
        updates = self.pending_updates[round_number]
        
        # 检查一致性
        if not self._check_consistency(round_number):
            print(f"Round {round_number} consistency check failed")
            return False  # 增加返回值表示聚合是否成功
            
        # 计算权重
        weights = {}
        total_weight = 0.0
        
        for station_id in updates.keys():
            # 基于性能指标调整权重
            base_weight = self.ground_stations[station_id]
            metrics_factor = self._calculate_metrics_factor(updates[station_id]['metrics'])
            weight = base_weight * metrics_factor
            
            weights[station_id] = weight
            total_weight += weight
            
        # 归一化权重
        for station_id in weights:
            weights[station_id] /= total_weight
            
        # 聚合更新
        aggregated_update = {}
        for param_name in next(iter(updates.values()))['update'].keys():
            weighted_sum = None
            
            for station_id, update in updates.items():
                weighted_update = update['update'][param_name] * weights[station_id]
                
                if weighted_sum is None:
                    weighted_sum = weighted_update
                else:
                    weighted_sum += weighted_update
                    
            aggregated_update[param_name] = weighted_sum
            
        # 创建新版本
        self._create_new_version(aggregated_update, round_number)
        return True  # 聚合成功
        
    def _check_consistency(self, round_number: int) -> bool:
        """检查更新的一致性"""
        updates = self.pending_updates[round_number]
        validation_results = self.validation_results.get(round_number, {})

        # 基本检查
        if len(updates) < self.config.min_ground_stations:
            print(f"Not enough stations for aggregation: {len(updates)} < {self.config.min_ground_stations}")
            return False

        # 检查基础版本一致性
        base_versions = set(u['base_version'] for u in updates.values())
        if len(base_versions) > 1:
            print(f"Inconsistent base versions detected: {base_versions}")
            return False

        # 如果不需要验证，检查到这里就可以返回True
        if not self.config.validation_required:
            return True

        # 验证结果检查
        if len(validation_results) < self.config.min_ground_stations:
            print(f"Not enough validation results: {len(validation_results)} < {self.config.min_ground_stations}")
            return False

        # 计算并检查每个指标的差异
        for metric_name in next(iter(validation_results.values())).keys():
            values = [result[metric_name] for result in validation_results.values()]
            max_val = max(values)
            min_val = min(values)
            diff = max_val - min_val
            avg = sum(values) / len(values)
            variance = sum((x - avg) ** 2 for x in values) / len(values)

            print(f"Metric {metric_name}:")
            print(f"  Values: {values}")
            print(f"  Max difference: {diff:.4f}")
            print(f"  Variance: {variance:.4f}")
            print(f"  Threshold: {1 - self.config.consistency_threshold:.4f}")

            # 检查最大差异
            if diff > (1 - self.config.consistency_threshold):
                print(f"Consistency check failed: difference {diff:.4f} exceeds threshold {1 - self.config.consistency_threshold:.4f}")
                return False

            # 检查方差
            max_variance = (1 - self.config.consistency_threshold) ** 2
            if variance > max_variance:
                print(f"Consistency check failed: variance {variance:.4f} exceeds threshold {max_variance:.4f}")
                return False

        return True
        
    def _calculate_metrics_factor(self, metrics: Dict[str, float]) -> float:
        """计算性能指标因子"""
        # 这里可以根据具体需求设计更复杂的计算方法
        if 'accuracy' in metrics:
            return max(0.1, metrics['accuracy'])
        return 1.0
        
    def _create_new_version(self, parameters: Dict[str, torch.Tensor],
                           round_number: int):
        """创建新的模型版本"""
        # 计算模型哈希值
        hash_input = []
        for name, param in parameters.items():
            hash_input.append(f"{name}:{param.mean().item():.4f}")
        hash_string = json.dumps(hash_input, sort_keys=True)
        hash_value = hashlib.sha256(hash_string.encode()).hexdigest()
        
        # 创建新版本
        version = ModelVersion(
            version=self.current_version + 1,
            parameters=parameters,
            hash_value=hash_value,
            timestamp=datetime.now().timestamp(),
            metrics=self._aggregate_metrics(round_number)
        )
        
        # 更新版本信息
        self.model_versions.append(version)
        self.current_version += 1
        
        # 只保留最近的版本
        if len(self.model_versions) > self.config.max_version_diff:
            self.model_versions.pop(0)
            
        # 清理已完成的更新和验证结果
        self.pending_updates.pop(round_number, None)
        self.validation_results.pop(round_number, None)
            
    def _aggregate_metrics(self, round_number: int) -> Dict[str, float]:
        """聚合性能指标"""
        metrics = defaultdict(list)
        
        # 收集所有验证结果
        for result in self.validation_results.get(round_number, {}).values():
            for metric, value in result.items():
                metrics[metric].append(value)
                
        # 计算平均值
        return {metric: np.mean(values) for metric, values in metrics.items()}
        
    def get_current_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取当前模型参数"""
        if not self.model_versions:
            return None
        return self.model_versions[-1].parameters
        
    def get_version_info(self, version: int) -> Optional[ModelVersion]:
        """获取指定版本的信息"""
        for v in self.model_versions:
            if v.version == version:
                return v
        return None
        
    def get_aggregation_stats(self) -> Dict:
        """获取聚合统计信息"""
        current_time = datetime.now().timestamp()
        
        return {
            'current_version': self.current_version,
            'num_ground_stations': len(self.ground_stations),
            'pending_rounds': len(self.pending_updates),
            'latest_update': self.model_versions[-1].timestamp if self.model_versions else 0,
            'version_history': len(self.model_versions)
        }
    

    def get_aggregated_update(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取全局聚合更新
        Args:
            round_number: 轮次
        Returns:
            聚合后的模型更新
        """
        try:
            self.logger.info(f"尝试获取轮次 {round_number} 的全局聚合结果")
            
            if round_number not in self.pending_updates:
                self.logger.warning(f"轮次 {round_number} 没有待处理的更新")
                return None
                
            updates = self.pending_updates[round_number]
            self.logger.info(f"当前轮次有 {len(updates)} 个地面站更新")
            
            # 检查是否有足够的地面站更新
            if len(updates) < self.config.min_ground_stations:
                self.logger.warning(f"地面站更新数量不足: {len(updates)} < {self.config.min_ground_stations}")
                return None
                
            # 计算权重
            total_orbits = sum(update['metrics']['num_orbits'] for update in updates.values())
            weights = {
                station_id: update['metrics']['num_orbits'] / total_orbits 
                for station_id, update in updates.items()
            }
            
            # 聚合更新
            aggregated_update = {}
            first_update = next(iter(updates.values()))['update']
            
            for param_name in first_update.keys():
                weighted_sum = None
                
                for station_id, update in updates.items():
                    if param_name not in update['update']:
                        continue
                        
                    param = update['update'][param_name] * weights[station_id]
                    if weighted_sum is None:
                        weighted_sum = param
                    else:
                        weighted_sum += param
                        
                if weighted_sum is not None:
                    aggregated_update[param_name] = weighted_sum
                    
            # 验证聚合结果
            for name, param in aggregated_update.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self.logger.error(f"参数 {name} 包含无效值")
                    return None
                    
            self.logger.info("全局聚合成功完成")
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"全局聚合过程出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None