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
        estimated_energy = self._estimate_training_energy()
        if not self.energy_manager.can_consume(estimated_energy):
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
            'batch_losses': [],  # 记录每个batch的loss
            'energy_consumption': 0.0,
            'compute_time': 0.0,
            'total_samples': len(self.dataset),
            'processed_samples': 0,
            'completed_epochs': 0
        }
        
        # 训练循环
        self.model.train()
        start_time = datetime.now()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 检查是否需要中断训练
                if self._should_interrupt_training():
                    print(f"Client {self.client_id}: 训练被中断")
                    break
                
                # 计算批次能耗
                batch_energy = self._estimate_batch_energy()
                if not self.energy_manager.can_consume(batch_energy):
                    print(f"Client {self.client_id}: 能量不足，中断训练")
                    break
                
                try:
                    # 前向传播
                    output = self.model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # 计算准确率
                    _, predicted = output.max(1)
                    batch_total = target.size(0)
                    batch_correct = predicted.eq(target).sum().item()
                    
                    # 更新统计信息
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    epoch_correct += batch_correct
                    epoch_total += batch_total
                    
                    # 记录batch统计
                    stats['batch_losses'].append(batch_loss)
                    stats['processed_samples'] += batch_total
                    # 在batch训练后添加验证
                    model_stats = self._verify_model_update()
                    stats['model_updates'] = model_stats
                    
                    # 记录能耗
                    self.energy_manager.consume_energy(self.client_id, batch_energy)
                    stats['energy_consumption'] += batch_energy
                    
                    print(f"Client {self.client_id}: Epoch {epoch+1}/{self.config.local_epochs}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {batch_loss:.4f}, "
                        f"Accuracy: {100.0 * batch_correct / batch_total:.2f}%")
                    
                except Exception as e:
                    print(f"Client {self.client_id}: 训练过程出错: {str(e)}")
                    continue
            
            # 记录每轮统计信息
            if epoch_total > 0:
                avg_loss = epoch_loss / len(train_loader)
                accuracy = 100.0 * epoch_correct / epoch_total
                stats['train_loss'].append(avg_loss)
                stats['train_accuracy'].append(accuracy)
                stats['completed_epochs'] += 1
                
                print(f"Client {self.client_id}: Epoch {epoch+1} 完成, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.2f}%")
        
        # 记录计算时间
        stats['compute_time'] = (datetime.now() - start_time).total_seconds()
        
        self.is_training = False
        self.train_stats.append(stats)
        
        # 打印训练总结
        print(f"\nClient {self.client_id} 训练完成:")
        print(f"处理样本数: {stats['processed_samples']}/{stats['total_samples']}")
        print(f"完成轮次: {stats['completed_epochs']}/{self.config.local_epochs}")
        print(f"最终loss: {stats['train_loss'][-1] if stats['train_loss'] else 'N/A'}")
        print(f"最终准确率: {stats['train_accuracy'][-1] if stats['train_accuracy'] else 'N/A'}%")
        print(f"能量消耗: {stats['energy_consumption']:.4f} Wh")
        print(f"计算时间: {stats['compute_time']:.4f} s\n")
        
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
            
        num_batches = len(self.dataset) // self.config.batch_size
        if len(self.dataset) % self.config.batch_size > 0:
            num_batches += 1
            
        # 每个batch的能耗
        batch_energy = self._estimate_batch_energy()
        
        # 总能耗
        total_energy = batch_energy * num_batches * self.config.local_epochs
        
        # 考虑计算能力的影响
        return total_energy / self.config.compute_capacity
        
    def _estimate_batch_energy(self) -> float:
        """估算处理一个批次数据的能量消耗"""
        # 基于批次大小估算计算时间（秒）
        compute_time = 0.05 + 0.001 * self.config.batch_size  # 基础时间 + 每样本处理时间
        
        # CPU能耗(Wh)
        cpu_energy = (self.config.compute_capacity * 15.0 * compute_time) / 3600
        
        # 通信能耗（基于模型大小）
        model_size_mb = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        comm_time = model_size_mb / 50.0  # 假设50Mbps的传输速率
        comm_energy = (20.0 * comm_time) / 3600  # 20W的传输功率
        
        # 总能耗
        total_energy = cpu_energy + comm_energy
        
        return total_energy
        
    def _should_interrupt_training(self) -> bool:
        """检查是否需要中断训练"""
        # 检查网络状态
        if not self.network_manager.is_connected():
            print(f"Client {self.client_id}: 网络连接丢失")
            return True
            
        # 检查能量状态
        if not self.energy_manager.has_minimum_energy(self.client_id):
            print(f"Client {self.client_id}: 能量低于最小阈值")
            return True
            
        # 检查是否有高优先级任务
        if self.network_manager.has_priority_task():
            print(f"Client {self.client_id}: 存在高优先级任务")
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
    
    def _verify_model_update(self) -> Dict[str, float]:
        """验证模型是否有更新"""
        verification = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                verification[f"{name}_grad_norm"] = grad_norm
                verification[f"{name}_param_norm"] = param_norm
                
        return verification
        
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