from experiments.baseline_experiment import BaselineExperiment
import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict

class Generator(nn.Module):
    """SDA-FL中使用的生成器"""
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """SDA-FL中使用的判别器"""
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

class SDAFLExperiment(BaselineExperiment):
    """SDA-FL实验类"""
    def __init__(self, config_path="configs/sda_fl_config.yaml"):
        super().__init__(config_path)
        
        # 额外的SDA-FL配置
        self.sda_fl_config = self.config.get('sda_fl', {})
        self.noise_dim = self.sda_fl_config.get('noise_dim', 100)
        self.num_synthetic_samples = self.sda_fl_config.get('num_synthetic_samples', 1000)
        self.dp_epsilon = self.sda_fl_config.get('dp_epsilon', 1.0)
        self.dp_delta = self.sda_fl_config.get('dp_delta', 1e-5)
        self.pseudo_threshold = self.sda_fl_config.get('pseudo_threshold', 0.8)
        self.initial_rounds = self.sda_fl_config.get('initial_rounds', 3)
        self.regenerate_interval = self.sda_fl_config.get('regenerate_interval', 5)
        
        # 初始化GAN模型
        self.generator = None
        self.discriminator = None
        self.synthetic_data = None
        
        self.logger.info("初始化SDA-FL实验")
        
    def _setup_logging(self):
        """设置日志，覆盖父类方法"""
        experiment_type = "sda_fl"
        return super()._setup_logging()
    
    def _init_gan_models(self, feature_dim):
        """初始化GAN模型"""
        self.generator = Generator(self.noise_dim, feature_dim)
        self.discriminator = Discriminator(feature_dim)
        
        # 如果可用，将模型移到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)
        self.discriminator.to(device)
        
        return self.generator, self.discriminator
    
    def _train_gan(self, train_data, num_epochs=50, batch_size=32):
        """训练GAN模型"""
        self.logger.info("开始训练GAN模型")
        
        # 获取特征维度
        if isinstance(train_data[0][0], torch.Tensor):
            feature_dim = train_data[0][0].shape[0]
        else:
            feature_dim = len(train_data[0][0])
            
        self._init_gan_models(feature_dim)
        
        # 设置优化器
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # 准备数据
        dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        
        # 获取设备
        device = next(self.generator.parameters()).device
        
        # 训练循环
        for epoch in range(num_epochs):
            total_d_loss = 0
            total_g_loss = 0
            batches = 0
            
            for batch_idx, (real_data, _) in enumerate(dataloader):
                batch_size = real_data.size(0)
                
                # 将数据移到设备上
                real_data = real_data.to(device)
                
                # 真实标签为1，伪造标签为0
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                
                # 训练判别器
                d_optimizer.zero_grad()
                
                # 真实样本
                outputs = self.discriminator(real_data)
                d_loss_real = criterion(outputs, real_labels)
                
                # 生成伪造样本 - 添加DP噪声
                noise = torch.randn(batch_size, self.noise_dim).to(device)
                fake_data = self.generator(noise)
                outputs = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(outputs, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                
                # 差分隐私 - 梯度裁剪和添加噪声
                self._apply_dp_to_gradients(self.discriminator, self.dp_epsilon)
                
                d_optimizer.step()
                
                # 训练生成器
                g_optimizer.zero_grad()
                outputs = self.discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                
                # 同样对生成器应用DP
                self._apply_dp_to_gradients(self.generator, self.dp_epsilon)
                
                g_optimizer.step()
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                batches += 1
                
            # 每轮结束后记录损失
            avg_d_loss = total_d_loss / batches if batches > 0 else 0
            avg_g_loss = total_g_loss / batches if batches > 0 else 0
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(f"GAN训练 - Epoch [{epoch+1}/{num_epochs}], "
                              f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
                
        self.logger.info("GAN训练完成")
        
    def _apply_dp_to_gradients(self, model, epsilon, clipping_value=1.0):
        """应用差分隐私 - 梯度裁剪和噪声添加"""
        # 裁剪梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        
        # 添加噪声
        sensitivity = clipping_value
        noise_scale = sensitivity / epsilon
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_scale, param.grad.shape, device=param.grad.device)
                param.grad += noise
    
    def _generate_synthetic_data(self, num_classes):
        """生成合成数据"""
        self.logger.info(f"生成{self.num_synthetic_samples}个合成样本")
        
        # 确保生成器在评估模式
        self.generator.eval()
        
        # 获取设备
        device = next(self.generator.parameters()).device
        
        with torch.no_grad():
            noise = torch.randn(self.num_synthetic_samples, self.noise_dim, device=device)
            synthetic_samples = self.generator(noise)
        
        # 为生成的数据分配伪标签
        synthetic_dataset = []
        valid_samples = 0
        
        for sample in synthetic_samples:
            # 计算伪标签 - 使用现有的分类器
            pred_probs = self._get_pseudo_labels(sample)
            if max(pred_probs) >= self.pseudo_threshold:
                # 关键修复: 保持标签为Tensor类型
                label_idx = torch.argmax(pred_probs).item()
                label = torch.tensor(label_idx, dtype=torch.long, device='cpu')  # 确保是 CPU Tensor
                synthetic_dataset.append((sample.cpu(), label))
                valid_samples += 1
        
        self.logger.info(f"伪标签分配完成，有效样本数: {valid_samples}/{self.num_synthetic_samples}")
        
        # 如果获取到的有效样本太少，降低阈值再次尝试
        if valid_samples < self.num_synthetic_samples * 0.2 and self.pseudo_threshold > 0.5:
            reduced_threshold = max(0.5, self.pseudo_threshold - 0.1)
            self.logger.info(f"有效样本数太少，降低阈值从{self.pseudo_threshold}到{reduced_threshold}重新筛选")
            
            synthetic_dataset = []
            for sample in synthetic_samples:
                pred_probs = self._get_pseudo_labels(sample)
                if max(pred_probs) >= reduced_threshold:
                    label = torch.argmax(pred_probs).item()
                    synthetic_dataset.append((sample.cpu(), label))
            
            self.logger.info(f"重新筛选后，有效样本数: {len(synthetic_dataset)}/{self.num_synthetic_samples}")
        
        return synthetic_dataset
    
    def _get_pseudo_labels(self, sample):
        """为合成样本分配伪标签"""
        # 使用当前的全局模型进行预测
        device = next(self.model.parameters()).device
        sample_batch = sample.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sample_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        return probs.cpu()
    
    def _train_satellite(self, sat_id, round_number):
        """重写卫星训练方法，加入合成数据"""
        if sat_id not in self.clients:
            return False, {}
            
        client = self.clients[sat_id]
        
        # 如果有合成数据，加入到训练集
        if self.synthetic_data is not None:
            # 保存原始数据集
            if not hasattr(client, 'original_dataset'):
                client.original_dataset = client.dataset
                
            # 检查和打印数据类型 (调试信息)
            if len(self.synthetic_data) > 0 and len(client.original_dataset) > 0:
                orig_sample = client.original_dataset[0]
                synth_sample = self.synthetic_data[0]
                print(f"原始数据类型: 特征={type(orig_sample[0])}, 标签={type(orig_sample[1])}")
                print(f"合成数据类型: 特征={type(synth_sample[0])}, 标签={type(synth_sample[1])}")
            
            # 合并原始数据和合成数据
            combined_dataset = list(client.original_dataset) + self.synthetic_data
            client.update_dataset(combined_dataset)
            
        # 执行训练
        stats = client.train(round_number)
        
        # 恢复原始数据集
        if self.synthetic_data is not None and hasattr(client, 'original_dataset'):
            client.update_dataset(client.original_dataset)
            
        return True, stats
    
    def train(self):
        """执行SDA-FL训练过程"""
        # 初始化记录列表
        accuracies = []
        losses = []
        energy_stats = {
            'training_energy': [],
            'communication_energy': [],
            'total_energy': []
        }
        satellite_stats = {
            'training_satellites': [],
            'receiving_satellites': [],
            'total_active': []
        }
        
        # 为每个客户端保存原始数据集
        for client_id, client in self.clients.items():
            client.original_dataset = client.dataset
        
        # 初始阶段 - 常规训练几轮以建立基础模型
        initial_rounds = self.initial_rounds
        self.logger.info(f"执行{initial_rounds}轮初始训练以建立基础模型")
        
        for round_num in range(initial_rounds):
            self.current_round = round_num
            self.logger.info(f"\n=== SDA-FL 初始阶段：第 {round_num + 1}/{initial_rounds} 轮训练 ===")
            
            round_stats = self._execute_training_round(round_num)
            
            # 更新统计信息
            if round_stats.get('accuracy') is not None:
                accuracies.append(round_stats['accuracy'])
            if round_stats.get('loss') is not None:
                losses.append(round_stats['loss'])
                
            energy_stats['training_energy'].append(round_stats.get('training_energy', 0))
            energy_stats['communication_energy'].append(round_stats.get('communication_energy', 0))
            energy_stats['total_energy'].append(round_stats.get('total_energy', 0))
            
            satellite_stats['training_satellites'].append(round_stats.get('training_satellites', 0))
            satellite_stats['receiving_satellites'].append(round_stats.get('receiving_satellites', 0))
            satellite_stats['total_active'].append(round_stats.get('total_active', 0))
            
            self.logger.info(f"初始阶段第 {round_num + 1} 轮完成，准确率: {round_stats.get('accuracy', 0):.4f}")
        
        # GAN训练 - 使用所有客户端的部分数据
        self.logger.info("\n=== 开始GAN训练阶段 ===")
        
        # 收集部分训练数据用于GAN训练
        gan_training_data = []
        samples_per_client = self.sda_fl_config.get('gan_samples_per_client', 100)
        
        for client_id, client in self.clients.items():
            if hasattr(client, 'dataset') and client.dataset:
                # 抽样数据
                indices = np.random.choice(len(client.dataset), 
                                        min(samples_per_client, len(client.dataset)), 
                                        replace=False)
                gan_training_data.extend([client.dataset[i] for i in indices])
        
        # 训练GAN
        gan_epochs = self.sda_fl_config.get('gan_epochs', 50)
        self._train_gan(gan_training_data, num_epochs=gan_epochs)
        
        # 生成合成数据
        self.num_classes = self.config.get('data', {}).get('num_classes', 2)
        if not hasattr(self, 'num_classes') or self.num_classes is None:
            self.num_classes = len(set([y for _, y in gan_training_data]))
            
        self.synthetic_data = self._generate_synthetic_data(self.num_classes)
        
        # 主要训练循环 - 使用合成数据增强
        for round_num in range(initial_rounds, self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"\n=== SDA-FL 增强阶段：第 {round_num + 1} 轮训练 (使用合成数据) ===")
            
            round_stats = self._execute_training_round(round_num)
            
            # 更新统计信息
            if round_stats.get('accuracy') is not None:
                accuracies.append(round_stats['accuracy'])
            if round_stats.get('loss') is not None:
                losses.append(round_stats['loss'])
                
            energy_stats['training_energy'].append(round_stats.get('training_energy', 0))
            energy_stats['communication_energy'].append(round_stats.get('communication_energy', 0))
            energy_stats['total_energy'].append(round_stats.get('total_energy', 0))
            
            satellite_stats['training_satellites'].append(round_stats.get('training_satellites', 0))
            satellite_stats['receiving_satellites'].append(round_stats.get('receiving_satellites', 0))
            satellite_stats['total_active'].append(round_stats.get('total_active', 0))
            
            self.logger.info(f"增强阶段第 {round_num - initial_rounds + 1} 轮完成，准确率: {round_stats.get('accuracy', 0):.4f}")
            
            # 每隔几轮重新生成合成数据
            if (round_num + 1) % self.regenerate_interval == 0:
                self.logger.info(f"\n=== 第 {round_num + 1} 轮后重新生成合成数据 ===")
                self.synthetic_data = self._generate_synthetic_data(self.num_classes)
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats
        }
        
        # 生成可视化
        self.visualizer.plot_training_metrics(
            accuracies=stats['accuracies'],
            losses=stats['losses'],
            energy_stats=stats['energy_stats'],
            satellite_stats=stats['satellite_stats'],
            save_path=self.log_dir / 'training_metrics.png'
        )
        
        return stats
    
    def _execute_training_round(self, round_num):
        """执行单轮训练"""
        # 此方法可以从基类继承，也可以根据需要重写
        # 这里简单地统计一些指标并返回
        stats = {
            'training_energy': 0,
            'communication_energy': 0,
            'training_satellites': 0,
            'receiving_satellites': 0,
            'total_active': 0
        }
        
        # 选择本轮参与的卫星
        satellites = self._select_satellites()
        
        # 训练
        trained_satellites = []
        for sat_id in satellites:
            pre_train_energy = self.energy_model.get_battery_level(sat_id)
            success, train_stats = self._train_satellite(sat_id, round_num)
            post_train_energy = self.energy_model.get_battery_level(sat_id)
            
            if success:
                trained_satellites.append(sat_id)
                stats['training_energy'] += (pre_train_energy - post_train_energy)
        
        stats['training_satellites'] = len(trained_satellites)
        
        # 模型聚合
        if trained_satellites:
            self._aggregate_models(trained_satellites)
            
            # 分发更新后的模型
            receiving_satellites = []
            for sat_id in satellites:
                pre_comm_energy = self.energy_model.get_battery_level(sat_id)
                self.clients[sat_id].apply_model_update(self.model.state_dict())
                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                
                stats['communication_energy'] += (pre_comm_energy - post_comm_energy)
                receiving_satellites.append(sat_id)
                
            stats['receiving_satellites'] = len(receiving_satellites)
                
        # 评估
        accuracy = self.evaluate()
        stats['accuracy'] = accuracy
        
        # 计算损失
        total_loss = 0
        count = 0
        for sat_id in trained_satellites:
            if hasattr(self.clients[sat_id], 'train_stats') and self.clients[sat_id].train_stats:
                total_loss += self.clients[sat_id].train_stats[-1]['summary']['train_loss'][-1]
                count += 1
        
        if count > 0:
            stats['loss'] = total_loss / count
            
        stats['total_energy'] = stats['training_energy'] + stats['communication_energy']
        stats['total_active'] = len(set(trained_satellites) | set(receiving_satellites))
        
        return stats
    
    def _select_satellites(self):
        """选择参与训练的卫星"""
        # 这里可以用不同的策略选择卫星
        # 示例：随机选择一定比例的卫星
        total_satellites = len(self.clients)
        participation_rate = self.config.get('fedavg', {}).get('participation_rate', 0.8)
        
        num_to_select = max(1, int(total_satellites * participation_rate))
        selected_indices = np.random.choice(total_satellites, num_to_select, replace=False)
        
        # 获取卫星ID列表
        satellite_ids = list(self.clients.keys())
        selected_satellites = [satellite_ids[i] for i in selected_indices]
        
        return selected_satellites
    
    def _aggregate_models(self, satellite_ids):
        """聚合模型"""
        if not satellite_ids:
            return
            
        # 收集更新和权重
        updates = []
        weights = []
        
        for sat_id in satellite_ids:
            model_update, _ = self.clients[sat_id].get_model_update()
            dataset_size = len(self.clients[sat_id].dataset)
            updates.append(model_update)
            weights.append(dataset_size)
            
        # 标准化权重
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # 加权平均聚合
        aggregated_state_dict = {}
        for k in updates[0].keys():
            weighted_sum = None
            for i, update in enumerate(updates):
                weighted_param = update[k] * weights[i]
                if weighted_sum is None:
                    weighted_sum = weighted_param
                else:
                    weighted_sum += weighted_param
            aggregated_state_dict[k] = weighted_sum
        
        # 保留原始模型的批量归一化统计信息
        current_state_dict = self.model.state_dict()
        for k in current_state_dict:
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                if k not in aggregated_state_dict:
                    aggregated_state_dict[k] = current_state_dict[k]
            
        # 更新全局模型
        self.model.load_state_dict(aggregated_state_dict)