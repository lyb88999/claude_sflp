from experiments.baseline_experiment import BaselineExperiment
import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class Generator(nn.Module):
    """SDA-FL中使用的生成器"""
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.1),
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
    """SDA-FL实验类 - 考虑卫星可见性和参与节点控制"""
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
        
        # 传播配置
        propagation_config = self.config.get('propagation', {})
        self.propagation_hops = propagation_config.get('hops', 2)
        self.max_propagation_satellites = propagation_config.get('max_satellites', 24)
        self.intra_orbit_links = propagation_config.get('intra_orbit_links', True)
        self.inter_orbit_links = propagation_config.get('inter_orbit_links', True)
        self.link_reliability = propagation_config.get('link_reliability', 0.95)
        self.energy_per_hop = propagation_config.get('energy_per_hop', 0.05)
        
        # 初始化卫星网络拓扑
        self.satellite_neighbors = {}
        
        # 初始化GAN模型
        self.generator = None
        self.discriminator = None
        self.synthetic_data = None
        
        # 设置最大工作线程数
        self.max_workers = self.config.get('execution', {}).get('max_workers', 8)
        
        self.logger.info("初始化SDA-FL实验")
        self.logger.info(f"- 传播跳数: {self.propagation_hops}")
        self.logger.info(f"- 最大卫星数: {self.max_propagation_satellites}")
        self.logger.info(f"- 合成样本数: {self.num_synthetic_samples}")
        
    def _setup_logging(self):
        """设置日志"""
        experiment_type = "sda_fl"
        return super()._setup_logging()
    
    def _build_satellite_network(self):
        """构建卫星间通信网络"""
        self.logger.info("构建卫星间通信网络...")
        
        # 创建卫星距离矩阵
        satellites = list(self.clients.keys())
        for sat_id in satellites:
            if sat_id not in self.satellite_neighbors:
                self.satellite_neighbors[sat_id] = []
            
            # 解析卫星轨道和编号
            parts = sat_id.split('_')[1].split('-')
            if len(parts) != 2:
                continue
                
            orbit_id, sat_num = int(parts[0]), int(parts[1])
            satellites_per_orbit = self.config['fl']['satellites_per_orbit']
            
            # 添加同轨道邻居
            if self.intra_orbit_links:
                for distance in [1, 2]:  # 距离1和2的卫星
                    # 向后
                    next_num = sat_num + distance
                    if next_num > satellites_per_orbit:
                        next_num = next_num - satellites_per_orbit
                    next_id = f"satellite_{orbit_id}-{next_num}"
                    if next_id in satellites and next_id not in self.satellite_neighbors[sat_id]:
                        self.satellite_neighbors[sat_id].append(next_id)
                        
                    # 向前
                    prev_num = sat_num - distance
                    if prev_num <= 0:
                        prev_num = satellites_per_orbit + prev_num
                    prev_id = f"satellite_{orbit_id}-{prev_num}"
                    if prev_id in satellites and prev_id not in self.satellite_neighbors[sat_id]:
                        self.satellite_neighbors[sat_id].append(prev_id)
            
            # 添加跨轨道邻居
            if self.inter_orbit_links:
                for other_orbit in range(1, self.config['fl']['num_orbits'] + 1):
                    if other_orbit == orbit_id:
                        continue  # 跳过同轨道
                    
                    # 添加相同位置的卫星
                    other_id = f"satellite_{other_orbit}-{sat_num}"
                    if other_id in satellites:
                        self.satellite_neighbors[sat_id].append(other_id)
                    
                    # 添加相邻位置的卫星
                    for offset in [-1, 1]:
                        other_num = sat_num + offset
                        if other_num <= 0:
                            other_num = satellites_per_orbit
                        elif other_num > satellites_per_orbit:
                            other_num = 1
                        
                        neighbor_id = f"satellite_{other_orbit}-{other_num}"
                        if neighbor_id in satellites:
                            self.satellite_neighbors[sat_id].append(neighbor_id)
        
        # 打印网络统计信息
        total_edges = sum(len(neighbors) for neighbors in self.satellite_neighbors.values())
        avg_neighbors = total_edges / len(self.satellite_neighbors) if self.satellite_neighbors else 0
        self.logger.info(f"卫星网络构建完成: {len(self.satellite_neighbors)} 个节点, 平均 {avg_neighbors:.2f} 个邻居")
    
    def _get_visible_satellites(self, current_time):
        """获取当前时间点可见的卫星"""
        visible_satellites = []
        
        # 检查每个卫星是否可见任何地面站
        for sat_id in self.clients.keys():
            for station_id in ['station_0', 'station_1', 'station_2']:
                is_visible = self.network_model._check_visibility(station_id, sat_id, current_time)
                if is_visible:
                    visible_satellites.append(sat_id)
                    break  # 只要有一个地面站可见就可以
        
        self.logger.info(f"当前时间 {datetime.fromtimestamp(current_time)} 可见卫星数: {len(visible_satellites)}")
        return visible_satellites
    
    def _get_propagation_satellites(self, visible_satellites, max_count):
        """从可见卫星开始，获取可传播到的卫星"""
        if not self.satellite_neighbors:
            self._build_satellite_network()
            
        # 已访问的卫星集合
        visited = set(visible_satellites)
        # 当前边界
        frontier = set(visible_satellites)
        # 所有可传播到的卫星
        propagation_satellites = set(visible_satellites)
        
        # 广度优先搜索，传播指定跳数
        for hop in range(self.propagation_hops):
            next_frontier = set()
            for sat_id in frontier:
                neighbors = self.satellite_neighbors.get(sat_id, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
                        propagation_satellites.add(neighbor)
                        
                        # 检查是否达到最大卫星数量
                        if len(propagation_satellites) >= max_count:
                            return propagation_satellites
            
            frontier = next_frontier
            if not frontier:  # 如果没有新的卫星，终止传播
                break
                
        return propagation_satellites
    
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
                # 保持标签为Tensor类型
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
                    label_idx = torch.argmax(pred_probs).item()
                    label = torch.tensor(label_idx, dtype=torch.long, device='cpu')
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
        """训练单个卫星，加入合成数据"""
        if sat_id not in self.clients:
            return False, {}
            
        client = self.clients[sat_id]
        
        # 如果有合成数据，加入到训练集
        if self.synthetic_data is not None:
            # 保存原始数据集
            if not hasattr(client, 'original_dataset'):
                client.original_dataset = client.dataset
            
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
        """执行SDA-FL训练过程，考虑卫星可见性和参与节点数控制"""
        # 初始化记录列表
        accuracies = []
        losses = []
        precision_macros = []
        recall_macros = []
        f1_macros = []
        precision_weighteds = []
        recall_weighteds = []
        f1_weighteds = []
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
        
        # 初始化网络拓扑结构
        self._build_satellite_network()
        
        current_time = datetime.now().timestamp()
        
        # 初始阶段 - 常规训练几轮以建立基础模型
        initial_rounds = self.initial_rounds
        self.logger.info(f"执行{initial_rounds}轮初始训练以建立基础模型")
        
        for round_num in range(initial_rounds):
            self.current_round = round_num
            self.logger.info(f"\n=== SDA-FL 初始阶段：第 {round_num + 1}/{initial_rounds} 轮训练 ===")
            
            # 1. 确定当前可见的卫星
            visible_satellites = self._get_visible_satellites(current_time)
            
            if not visible_satellites:
                self.logger.warning(f"当前时间点没有可见卫星，等待60秒")
                current_time += 60  # 等待60秒
                self.topology_manager.update_topology(current_time)  # 更新拓扑
                continue
                
            self.logger.info(f"当前有 {len(visible_satellites)} 颗卫星可见")

            # 2. 使用有限传播扩展卫星集合
            max_sats = self.max_propagation_satellites
            propagation_satellites = self._get_propagation_satellites(visible_satellites, max_sats)
            
            self.logger.info(f"通过有限传播策略选择了 {len(propagation_satellites)} 颗卫星")
            
            # 记录传播参与卫星
            round_receiving_sats = set(propagation_satellites)
            
            # 3. 分发全局模型参数给传播卫星
            global_model = self.model.state_dict()
            round_comm_energy = 0
            
            for sat_id in propagation_satellites:
                # 记录通信能耗
                pre_comm_energy = self.energy_model.get_battery_level(sat_id)
                self.clients[sat_id].apply_model_update(global_model)
                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                
                round_comm_energy += (pre_comm_energy - post_comm_energy)
                
                self.logger.info(f"分发模型参数给卫星 {sat_id}")
            
            # 4. 所有传播卫星进行本地训练
            round_training_energy = 0
            round_training_satellites = set()
            trained_satellites = []
            
            # 使用线程池并行训练卫星
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_satellite = {
                    executor.submit(self._train_satellite, sat_id, round_num): sat_id
                    for sat_id in propagation_satellites
                }
                
                for future in as_completed(future_to_satellite):
                    sat_id = future_to_satellite[future]
                    try:
                        success, stats = future.result()
                        if success:
                            trained_satellites.append(sat_id)
                            round_training_energy += stats['summary']['energy_consumption']
                            round_training_satellites.add(sat_id)
                            self.logger.info(f"卫星 {sat_id} 完成训练，Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                        f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%")
                    except Exception as e:
                        self.logger.error(f"训练卫星 {sat_id} 时出错: {str(e)}")
            
            # 更新时间，模拟通信和训练的时间消耗
            communication_time = 60  # 假设通信需要1分钟
            training_time = 300      # 假设训练需要5分钟
            current_time += communication_time + training_time
            
            # 5. 等待下次卫星可见性窗口，收集模型更新
            # 为了模拟实际情况，需要前进时间直到足够的卫星再次可见
            while True:
                visible_satellites = self._get_visible_satellites(current_time)
                # 计算已训练且当前可见的卫星集合
                visible_trained = [sat for sat in trained_satellites if sat in visible_satellites]
                
                if len(visible_trained) >= self.config['aggregation']['min_updates']:
                    self.logger.info(f"当前有 {len(visible_trained)} 颗已训练的卫星可见，可以进行聚合")
                    break
                
                # 如果没有足够的卫星可见，前进时间
                current_time += 60  # 每次前进1分钟
                if current_time - (datetime.now().timestamp() + communication_time + training_time) > self.config['aggregation']['timeout']:
                    self.logger.warning(f"等待聚合超时，使用当前可见的卫星进行聚合")
                    break
            
            # 6. 收集可见训练卫星的模型更新并进行聚合
            if trained_satellites:
                updates = []
                weights = []
                
                # 记录模型上传通信能耗
                for sat_id in visible_trained:
                    pre_upload_energy = self.energy_model.get_battery_level(sat_id)
                    model_update, stats = self.clients[sat_id].get_model_update()
                    post_upload_energy = self.energy_model.get_battery_level(sat_id)
                    
                    round_comm_energy += (pre_upload_energy - post_upload_energy)
                    
                    dataset_size = len(self.clients[sat_id].dataset)
                    updates.append(model_update)
                    weights.append(dataset_size)
                
                # 标准化权重
                total_samples = sum(weights)
                weights = [w/total_samples for w in weights]
                
                # 聚合
                aggregated_update = {}
                for param_name in updates[0].keys():
                    weighted_sum = None
                    for i, update in enumerate(updates):
                        weighted_param = update[param_name] * weights[i]
                        if weighted_sum is None:
                            weighted_sum = weighted_param
                        else:
                            weighted_sum += weighted_param
                    aggregated_update[param_name] = weighted_sum

                # 7. 更新全局模型 - 保留BatchNorm层统计数据
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        aggregated_update[name] = param

                # 更新全局模型
                self.model.load_state_dict(aggregated_update)
                
                # 8. 评估全局模型
                # accuracy = self.evaluate()
                # accuracies.append(accuracy)
                metrics = self.evaluate() 

                accuracies.append(metrics['accuracy'])
                precision_macros.append(metrics['precision_macro'])
                recall_macros.append(metrics['recall_macro'])
                f1_macros.append(metrics['f1_macro'])
                precision_weighteds.append(metrics['precision_weighted'])
                recall_weighteds.append(metrics['recall_weighted'])
                f1_weighteds.append(metrics['f1_weighted'])
                
                # 计算平均损失
                round_loss = 0
                for sat_id in trained_satellites:
                    if self.clients[sat_id].train_stats:
                        round_loss += self.clients[sat_id].train_stats[-1]['summary']['train_loss'][-1]
                losses.append(round_loss / len(trained_satellites))
                
                self.logger.info(f"初始阶段第 {round_num + 1} 轮完成，准确率: {metrics['accuracy']:.2f}%")
                
                # 记录能源和卫星统计信息
                energy_stats['training_energy'].append(round_training_energy)
                energy_stats['communication_energy'].append(round_comm_energy)
                energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
                
                satellite_stats['training_satellites'].append(len(round_training_satellites))
                satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
                satellite_stats['total_active'].append(len(round_training_satellites | round_receiving_sats))
            else:
                self.logger.warning(f"没有足够的已训练卫星可见，跳过本轮聚合")
                
            # 更新时间
            current_time += self.config['fl']['round_interval']
        
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
            self.num_classes = len(set([label.item() if isinstance(label, torch.Tensor) else label for _, label in gan_training_data]))
            
        self.synthetic_data = self._generate_synthetic_data(self.num_classes)
        
        # 主要训练循环 - 使用合成数据增强
        for round_num in range(initial_rounds, self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"\n=== SDA-FL 增强阶段：第 {round_num + 1} 轮训练 (使用合成数据) ===")
            
            # 1. 确定当前可见的卫星
            visible_satellites = self._get_visible_satellites(current_time)
            
            if not visible_satellites:
                self.logger.warning(f"当前时间点没有可见卫星，等待60秒")
                current_time += 60  # 等待60秒
                self.topology_manager.update_topology(current_time)  # 更新拓扑
                continue
                
            self.logger.info(f"当前有 {len(visible_satellites)} 颗卫星可见")

            # 2. 使用有限传播扩展卫星集合
            max_sats = self.max_propagation_satellites
            propagation_satellites = self._get_propagation_satellites(visible_satellites, max_sats)
            
            self.logger.info(f"通过有限传播策略选择了 {len(propagation_satellites)} 颗卫星")
            
            # 记录传播参与卫星
            round_receiving_sats = set(propagation_satellites)
            
            # 3. 分发全局模型参数给传播卫星
            global_model = self.model.state_dict()
            round_comm_energy = 0
            
            for sat_id in propagation_satellites:
                # 记录通信能耗
                pre_comm_energy = self.energy_model.get_battery_level(sat_id)
                self.clients[sat_id].apply_model_update(global_model)
                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                
                round_comm_energy += (pre_comm_energy - post_comm_energy)
                
                self.logger.info(f"分发模型参数给卫星 {sat_id}")
            
            # 4. 所有传播卫星进行本地训练（使用合成数据增强）
            round_training_energy = 0
            round_training_satellites = set()
            trained_satellites = []
            
            # 使用线程池并行训练卫星
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_satellite = {
                    executor.submit(self._train_satellite, sat_id, round_num): sat_id
                    for sat_id in propagation_satellites
                }
                
                for future in as_completed(future_to_satellite):
                    sat_id = future_to_satellite[future]
                    try:
                        success, stats = future.result()
                        if success:
                            trained_satellites.append(sat_id)
                            round_training_energy += stats['summary']['energy_consumption']
                            round_training_satellites.add(sat_id)
                            self.logger.info(f"卫星 {sat_id} 完成训练(使用合成数据)，Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                        f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%")
                    except Exception as e:
                        self.logger.error(f"训练卫星 {sat_id} 时出错: {str(e)}")
            
            # 更新时间，模拟通信和训练的时间消耗
            communication_time = 60
            training_time = 300
            current_time += communication_time + training_time
            
            # 5. 等待下次卫星可见性窗口，收集模型更新
            while True:
                visible_satellites = self._get_visible_satellites(current_time)
                visible_trained = [sat for sat in trained_satellites if sat in visible_satellites]
                
                if len(visible_trained) >= self.config['aggregation']['min_updates']:
                    self.logger.info(f"当前有 {len(visible_trained)} 颗已训练的卫星可见，可以进行聚合")
                    break
                
                current_time += 60
                if current_time - (datetime.now().timestamp() + communication_time + training_time) > self.config['aggregation']['timeout']:
                    self.logger.warning(f"等待聚合超时，使用当前可见的卫星进行聚合")
                    break
            
            # 6. 收集可见训练卫星的模型更新并进行聚合
            if trained_satellites:
                updates = []
                weights = []
                
                for sat_id in visible_trained:
                    pre_upload_energy = self.energy_model.get_battery_level(sat_id)
                    model_update, stats = self.clients[sat_id].get_model_update()
                    post_upload_energy = self.energy_model.get_battery_level(sat_id)
                    
                    round_comm_energy += (pre_upload_energy - post_upload_energy)
                    
                    dataset_size = len(self.clients[sat_id].original_dataset)  # 使用原始数据集大小作为权重
                    updates.append(model_update)
                    weights.append(dataset_size)
                
                # 标准化权重
                total_samples = sum(weights)
                weights = [w/total_samples for w in weights]
                
                # 聚合
                aggregated_update = {}
                for param_name in updates[0].keys():
                    weighted_sum = None
                    for i, update in enumerate(updates):
                        weighted_param = update[param_name] * weights[i]
                        if weighted_sum is None:
                            weighted_sum = weighted_param
                        else:
                            weighted_sum += weighted_param
                    aggregated_update[param_name] = weighted_sum

                # 7. 更新全局模型 - 保留BatchNorm层统计数据
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        aggregated_update[name] = param

                # 更新全局模型
                self.model.load_state_dict(aggregated_update)
                
                # 8. 评估全局模型
                # accuracy = self.evaluate()
                # accuracies.append(accuracy)
                metrics = self.evaluate()
                accuracies.append(metrics['accuracy'])
                precision_macros.append(metrics['precision_macro'])
                recall_macros.append(metrics['recall_macro'])
                f1_macros.append(metrics['f1_macro'])
                precision_weighteds.append(metrics['precision_weighted'])
                recall_weighteds.append(metrics['recall_weighted'])
                f1_weighteds.append(metrics['f1_weighted'])
                
                # 计算平均损失
                round_loss = 0
                for sat_id in trained_satellites:
                    if self.clients[sat_id].train_stats:
                        round_loss += self.clients[sat_id].train_stats[-1]['summary']['train_loss'][-1]
                losses.append(round_loss / len(trained_satellites))
                
                self.logger.info(f"增强阶段第 {round_num - initial_rounds + 1} 轮完成，准确率: {metrics['accuracy']:.2f}%")
                
                # 记录能源和卫星统计信息
                energy_stats['training_energy'].append(round_training_energy)
                energy_stats['communication_energy'].append(round_comm_energy)
                energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
                
                satellite_stats['training_satellites'].append(len(round_training_satellites))
                satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
                satellite_stats['total_active'].append(len(round_training_satellites | round_receiving_sats))
            else:
                self.logger.warning(f"没有足够的已训练卫星可见，跳过本轮聚合")
            
            # 每隔几轮重新生成合成数据
            if (round_num + 1) % self.regenerate_interval == 0:
                self.logger.info(f"\n=== 第 {round_num + 1} 轮后重新生成合成数据 ===")
                self.synthetic_data = self._generate_synthetic_data(self.num_classes)
                
            # 更新时间
            current_time += self.config['fl']['round_interval']
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'precision_macros': precision_macros,
            'recall_macros': recall_macros,
            'f1_macros': f1_macros,
            'precision_weighteds': precision_weighteds,
            'recall_weighteds': recall_weighteds,
            'f1_weighteds': f1_weighteds,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats,
            'num_synthetic_samples': self.num_synthetic_samples
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
