from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fl_core.client.satellite_client import SatelliteClient, ClientConfig

class FedProxClient(SatelliteClient):
    def __init__(self, client_id: str, model: nn.Module, config: ClientConfig,
                 network_manager, energy_manager, mu: float = 0.01, device=None):
        """
        初始化 FedProx 客户端
        
        Args:
            client_id: 客户端ID
            model: 模型实例
            config: 客户端配置
            network_manager: 网络管理器
            energy_manager: 能源管理器
            mu: 接近性参数，控制正则化强度
            device: 计算设备
        """
        super().__init__(client_id, model, config, network_manager, energy_manager, device)
        self.mu = mu  # 接近性参数
        self.global_model_params = {}  # 存储全局模型参数
        print(f"FedProx Client {client_id} 初始化完成，μ={mu}")
        
    def apply_model_update(self, model_update: Dict[str, torch.Tensor]):
        """重写应用模型更新方法，保存全局模型参数"""
        # 保存全局模型参数用于接近性正则化
        self.global_model_params = {
            name: param.clone().detach().to(self.device) for name, param in model_update.items()
        }
        # 调用父类方法更新模型
        super().apply_model_update(model_update)
        print(f"FedProx Client {self.client_id} 应用模型更新，保存全局参数用于接近性正则化")
    
    def _train_one_epoch(self, epoch: int, train_loader: DataLoader, stats: Dict) -> Dict:
        """覆盖训练方法，添加接近性正则化"""
        epoch_stats = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'completed': True
        }
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 批次能量检查
            batch_energy = self._estimate_batch_energy()
            if not self.energy_manager.can_consume(self.client_id, batch_energy):
                print(f"Client {self.client_id}: 能量不足，中断训练")
                epoch_stats['completed'] = False
                break
            
            try:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                task_loss = nn.functional.cross_entropy(output, target)
                
                # 添加接近性正则化项
                proximal_term = 0.0
                if self.global_model_params:
                    for name, param in self.model.named_parameters():
                        if name in self.global_model_params:
                            proximal_term += torch.sum((param - self.global_model_params[name])**2)
                    
                # 计算总损失
                loss = task_loss + (self.mu / 2) * proximal_term
                
                # 记录接近性项的值，用于后续分析
                self.last_proximal_term = proximal_term.item()
                
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
                
                # 记录能耗
                self.energy_manager.consume_energy(self.client_id, batch_energy)
                stats['summary']['energy_consumption'] += batch_energy
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"FedProx Client {self.client_id}: Epoch {epoch+1}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Task Loss: {task_loss.item():.4f}, "
                        f"Proximal Term: {proximal_term.item():.4f}, "
                        f"Total Loss: {batch_loss:.4f}")
                    
            except Exception as e:
                print(f"Client {self.client_id}: 训练过程出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # 计算epoch统计信息
        if epoch_stats['total'] > 0 and epoch_stats['completed']:
            avg_loss = epoch_stats['loss'] / len(train_loader)
            accuracy = 100.0 * epoch_stats['correct'] / epoch_stats['total']
            stats['summary']['train_loss'].append(avg_loss)
            stats['summary']['train_accuracy'].append(accuracy)
            stats['summary']['completed_epochs'] += 1
        
        return epoch_stats