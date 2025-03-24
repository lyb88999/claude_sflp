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
        """
        self.client_id = client_id
        
        # 创建模型的深度复制
        if hasattr(model, '__init__args__') and hasattr(model, '__init__kwargs__'):
            # 使用保存的初始化参数创建新实例
            self.model = type(model)(*model.__init__args__, **model.__init__kwargs__)
            # 复制参数
            self.model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
        else:
            # 无法获取初始化参数，直接使用传入的模型
            self.logger.warning(f"Client {client_id}: 无法深度复制模型，使用直接引用")
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
        # 1. 初始检查
        if not self._check_training_prerequisites():
            return self._get_empty_stats()
        
        # 2. 准备训练
        train_loader = self._prepare_data_loader()
        if not train_loader:
            return self._get_empty_stats()
        
        # 3. 初始化训练统计
        stats = self._init_training_stats()
        
        # 4. 执行训练
        self.model.train()
        start_time = datetime.now()
        
        for epoch in range(self.config.local_epochs):
            epoch_stats = self._train_one_epoch(epoch, train_loader, stats)
            if not epoch_stats['completed']:
                break
        
        # 5. 完成训练
        stats = self._finalize_training_stats(stats, start_time)
        
        # 6. 记录结果
        self._log_training_results(stats)
        
        return stats

    def _check_training_prerequisites(self) -> bool:
        """检查训练前提条件"""
        if not self.dataset:
            print(f"Client {self.client_id}: 数据集未设置")
            return False
            
        if len(self.dataset) == 0:
            print(f"Client {self.client_id}: 数据集为空")
            return False
            
        estimated_energy = self._estimate_training_energy()
        # 修改这里：添加 client_id 参数
        if not self.energy_manager.can_consume(self.client_id, estimated_energy):
            print(f"Client {self.client_id}: 能量不足，跳过训练")
            return False
            
        return True

    def _prepare_data_loader(self) -> Optional[DataLoader]:
        """准备数据加载器"""
        train_loader = DataLoader(
            self.dataset,
            batch_size=min(self.config.batch_size, len(self.dataset)),
            shuffle=True
        )
        
        if len(train_loader) == 0:
            print(f"Client {self.client_id}: 没有可训练的批次，跳过训练")
            return None
            
        return train_loader

    def _init_training_stats(self) -> Dict:
        """初始化训练统计信息"""
        return {
            'summary': {
                'train_loss': [],
                'train_accuracy': [],
                'energy_consumption': 0.0,
                'compute_time': 0.0,
                'completed_epochs': 0
            },
            'details': {
                'batch_losses': [],
                'total_samples': len(self.dataset),
                'processed_samples': 0,
                'model_updates': None
            }
        }

    def _train_one_epoch(self, epoch: int, train_loader: DataLoader, stats: Dict) -> Dict:
        """训练一个epoch"""
        epoch_stats = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'completed': True
        }
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 批次能量检查
            batch_energy = self._estimate_batch_energy()
            # 修改这里：添加 client_id 参数
            if not self.energy_manager.can_consume(self.client_id, batch_energy):
                print(f"Client {self.client_id}: 能量不足，中断训练")
                epoch_stats['completed'] = False
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
                epoch_stats['loss'] += batch_loss
                epoch_stats['correct'] += batch_correct
                epoch_stats['total'] += batch_total
                
                # 记录batch统计
                stats['details']['batch_losses'].append(batch_loss)
                stats['details']['processed_samples'] += batch_total
                
                # 记录能耗，同样修改这里
                self.energy_manager.consume_energy(self.client_id, batch_energy)
                stats['summary']['energy_consumption'] += batch_energy
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Client {self.client_id}: Epoch {epoch+1}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}")
                    
            except Exception as e:
                print(f"Client {self.client_id}: 训练过程出错: {str(e)}")
                continue
        
        # 计算epoch统计信息
        if epoch_stats['total'] > 0 and epoch_stats['completed']:
            avg_loss = epoch_stats['loss'] / len(train_loader)
            accuracy = 100.0 * epoch_stats['correct'] / epoch_stats['total']
            stats['summary']['train_loss'].append(avg_loss)
            stats['summary']['train_accuracy'].append(accuracy)
            stats['summary']['completed_epochs'] += 1
        
        return epoch_stats

    def _train_one_batch(self, data: torch.Tensor, target: torch.Tensor, 
                     batch_energy: float) -> Optional[Dict]:
        """训练一个batch"""
        try:
            # 确保数据是独立的副本
            data = data.clone().detach()
            target = target.clone().detach()
            
            # 清除之前的梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            with torch.no_grad():
                _, predicted = output.max(1)
                correct = predicted.eq(target).sum().item()
                
            return {
                'loss': loss.item(),
                'correct': correct,
                'total': target.size(0)
            }
            
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: 训练过程出错: {str(e)}")
            return None

    def _finalize_training_stats(self, stats: Dict, start_time: datetime) -> Dict:
        """完成训练统计"""
        # 更新训练时间
        compute_time = (datetime.now() - start_time).total_seconds()
        stats['summary']['compute_time'] = compute_time
        
        # 更新最终指标
        if stats['summary']['train_loss']:
            stats['summary'].update({
                'final_loss': stats['summary']['train_loss'][-1],
                'final_accuracy': stats['summary']['train_accuracy'][-1]
            })
        
        # 更新训练状态
        self.is_training = False
        self.train_stats.append(stats)
        
        return stats

    def _log_training_results(self, stats: Dict):
        """记录训练结果"""
        print(f"\nClient {self.client_id} 训练完成: "
            f"轮次: {stats['summary']['completed_epochs']}/{self.config.local_epochs} | "
            f"Loss: {stats['summary']['final_loss']:.4f} | "
            f"Acc: {stats['summary']['final_accuracy']:.2f}% | "
            f"能耗: {stats['summary']['energy_consumption']:.4f}Wh | "
            f"耗时: {stats['summary']['compute_time']:.3f}s")
        
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
            model_diff[name] = param.data.clone().detach()
            
        return model_diff, self.train_stats[-1]
        
    def apply_model_update(self, model_update: Dict[str, torch.Tensor]):
        model_size_mb = sum(p.nelement() * p.element_size() for p in model_update.values()) / (1024 * 1024)
        energy_consumption = 0.01 * model_size_mb  # 例如每MB消耗0.001Wh
        """应用模型更新"""
        with torch.no_grad():
            # 创建参数的深度复制
            new_state_dict = {}
            for name, param in model_update.items():
                new_state_dict[name] = param.clone().detach()
            
            # 检查参数匹配
            current_state = self.model.state_dict()
            for name in new_state_dict:
                if name not in current_state:
                    print(f"警告: 参数 {name} 不在模型中")

            # 保留批量归一化层的统计数据
            for name, param in current_state.items():
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    if name not in new_state_dict:
                        new_state_dict[name] = param
            
            # 更新模型参数
            self.model.load_state_dict(new_state_dict)
            
            # 重新初始化优化器
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum
            )
            
            # 添加调试信息
            first_param = next(iter(new_state_dict.values()))
            print(f"卫星 {self.client_id} 应用更新: 第一个参数示例值 {first_param.flatten()[0].item():.4f}")
            self.energy_manager.consume_energy(self.client_id, energy_consumption)


                    
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
                
        # 调整能量估算参数
        base_computation_energy = 0.001  # 每个batch的基础计算能耗(Wh)
        communication_energy = 0.0005   # 每个batch的通信能耗(Wh)
        
        # 每个batch的总能耗
        batch_energy = base_computation_energy + communication_energy
        
        # 总能耗
        total_energy = batch_energy * num_batches * self.config.local_epochs
        
        return total_energy
        
    # def _estimate_batch_energy(self) -> float:
    #     """估算处理一个批次数据的能量消耗"""
    #     # 基础计算能耗
    #     base_computation_energy = 0.001  # Wh
        
    #     # 通信能耗（基于模型大小）
    #     model_size_mb = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    #     communication_energy = 0.0005 * model_size_mb  # Wh
        
    #     return base_computation_energy + communication_energy
        

    def _estimate_batch_energy(self) -> float:
        base_computation_energy = 0.001
        
        # 使用客户端ID生成变化因子
        # 使用客户端ID生成变化因子
        client_id_num = int(self.client_id.split('_')[1].split('-')[1]) if '-' in self.client_id else 0
        variation_factor = 0.8 + 0.4 * (client_id_num % 11) / 11
        
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        communication_energy = 0.0005 * model_size_mb
        
        return (base_computation_energy * variation_factor) + communication_energy

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
    
    def _get_empty_stats(self) -> Dict:
        """返回空的训练统计信息"""
        return {
            'summary': {
                'train_loss': [],
                'train_accuracy': [],
                'energy_consumption': 0.0,
                'compute_time': 0.0,
                'completed_epochs': 0
            },
            'details': {
                'batch_losses': [],
                'total_samples': 0,
                'processed_samples': 0,
                'model_updates': None
            }
        }