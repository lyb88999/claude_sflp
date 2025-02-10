from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ClientConfig:
    """卫星客户端配置"""
    batch_size: int = 32
    local_epochs: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9
    compute_capacity: float = 1.0  # 计算能力系数
    storage_capacity: float = 1000.0  # 存储容量(MB)

class SatelliteClient:
    def __init__(self, 
                 client_id: str,
                 model: nn.Module,
                 config: ClientConfig,
                 network_manager,
                 energy_manager):
        """
        初始化卫星客户端
        Args:
            client_id: 客户端ID
            model: 初始模型
            config: 客户端配置
            network_manager: 网络管理器实例
            energy_manager: 能源管理器实例
        """
        self.client_id = client_id
        self.model = model
        self.config = config
        self.network_manager = network_manager
        self.energy_manager = energy_manager
        
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.train_stats = []
        self.is_training = False
        self.current_round = 0
        
        # 初始化优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )
        
    def set_dataset(self, dataset: Dataset):
        """设置本地数据集"""
        self.dataset = dataset
        
    def train(self, round_number: int) -> Dict:
        """
        执行本地训练
        Args:
            round_number: 当前轮次
        Returns:
            训练统计信息
        """
        if not self.dataset:
            raise ValueError("Dataset not set")
            
        if len(self.dataset) == 0:
            print(f"Client {self.client_id}: 数据集为空，跳过训练")
            return self._get_empty_stats()
            
        self.current_round = round_number
        self.is_training = True
        
        # 检查能量状态
        if not self._check_energy_available():
            print(f"Client {self.client_id}: 能量不足，跳过训练")
            return self._get_empty_stats()
            
        # 创建数据加载器
        train_loader = DataLoader(
            self.dataset,
            batch_size=min(self.config.batch_size, len(self.dataset)),
            shuffle=True
        )
        
        if len(train_loader) == 0:
            print(f"Client {self.client_id}: 没有可训练的批次，跳过训练")
            return self._get_empty_stats()
            
        # 训练统计
        stats = {
            'train_loss': [],
            'train_accuracy': [],
            'energy_consumption': 0.0,
            'compute_time': 0.0,
            'total_samples': len(self.dataset)
        }
        
        # 训练循环
        self.model.train()
        start_time = datetime.now()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 检查是否需要中断训练
                if self._should_interrupt_training():
                    break
                    
                # 计算能耗
                batch_energy = self._estimate_batch_energy()
                if not self.energy_manager.can_consume(batch_energy):
                    print(f"Client {self.client_id}: 能量不足，中断训练")
                    break
                    
                # 前向传播
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新统计信息
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 记录能耗
                self.energy_manager.consume_energy(self.client_id, batch_energy)
                stats['energy_consumption'] += batch_energy
                
            # 记录每轮统计信息
            if total > 0:  # 确保有数据被处理
                avg_loss = epoch_loss / len(train_loader)
                accuracy = 100.0 * correct / total
                stats['train_loss'].append(avg_loss)
                stats['train_accuracy'].append(accuracy)
            
        # 记录计算时间
        stats['compute_time'] = (datetime.now() - start_time).total_seconds()
        
        self.is_training = False
        self.train_stats.append(stats)
        return stats
        
    def get_model_update(self) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        获取模型更新
        Returns:
            (模型参数差异, 训练统计信息)
        """
        if not self.train_stats:
            return {}, self._get_empty_stats()
            
        model_diff = {}
        for name, param in self.model.named_parameters():
            model_diff[name] = param.data.clone()
            
        return model_diff, self.train_stats[-1]
        
    def apply_model_update(self, model_update: Dict[str, torch.Tensor]):
        """
        应用模型更新
        Args:
            model_update: 模型参数更新
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in model_update:
                    param.copy_(model_update[name])
                    
    def evaluate(self, test_data: Dataset) -> Dict:
        """
        评估模型性能
        Args:
            test_data: 测试数据集
        Returns:
            评估结果
        """
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size)
        
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += nn.functional.cross_entropy(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'test_samples': total
        }
        
    def _check_energy_available(self) -> bool:
        """检查是否有足够的能量进行训练"""
        # 估算整体训练所需能量
        estimated_energy = self._estimate_training_energy()
        return self.energy_manager.can_consume(estimated_energy)
        
        
    def _estimate_training_energy(self) -> float:
        """估算完整训练过程的能量消耗"""
        if not self.dataset:
            return 0.0
            
        # 基于数据集大小和训练轮数估算
        num_batches = len(self.dataset) // self.config.batch_size
        energy_per_batch = self._estimate_batch_energy()
        total_energy = (energy_per_batch * num_batches * 
                       self.config.local_epochs)
                       
        # 考虑计算能力的影响
        return total_energy / self.config.compute_capacity
        
    def _estimate_batch_energy(self) -> float:
        """估算处理一个批次数据的能量消耗"""
        # 基于实际硬件功耗进行估算
        compute_time = 0.1  # 假设每批次计算时间为0.1秒
        
        # 计算能耗(Wh)
        compute_energy = (self.config.compute_capacity * 15.0 * compute_time) / 3600  # 15W的CPU功耗
        
        # 考虑通信能耗（假设每批次需要发送模型更新）
        comm_overhead = 0.001  # 1mWh的通信开销
        
        return compute_energy + comm_overhead
        
    def _should_interrupt_training(self) -> bool:
        """检查是否需要中断训练"""
        # 检查网络状态
        if not self.network_manager.is_connected():
            return True
            
        # 检查能量状态
        if not self.energy_manager.has_minimum_energy(self.client_id):
            return True
            
        # 检查是否有高优先级任务
        if self.network_manager.has_priority_task():
            return True
            
        return False
        
    def _get_empty_stats(self) -> Dict:
        """返回空的训练统计信息"""
        return {
            'train_loss': [],
            'train_accuracy': [],
            'energy_consumption': 0.0,
            'compute_time': 0.0,
            'total_samples': 0
        }
        
    def get_status(self) -> Dict:
        """获取客户端状态信息"""
        return {
            'client_id': self.client_id,
            'is_training': self.is_training,
            'current_round': self.current_round,
            'dataset_size': len(self.dataset) if self.dataset else 0,
            'energy_level': self.energy_manager.get_energy_level(),
            'network_connected': self.network_manager.is_connected(),
            'compute_capacity': self.config.compute_capacity
        }