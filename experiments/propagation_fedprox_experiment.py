# experiments/propagation_fedprox_experiment.py
from experiments.fedprox_experiment import FedProxExperiment
import logging
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class LimitedPropagationFedProx(FedProxExperiment):
    """
    有限传播FedProx变体 - 允许模型在卫星间有限范围传播
    """
    
    def __init__(self, config_path: str = "configs/propagation_fedprox_config.yaml"):
        """初始化有限传播FedProx实验"""
        super().__init__(config_path)
        
        # 获取传播相关配置
        propagation_config = self.config.get('propagation', {})
        self.propagation_hops = propagation_config.get('hops', 1)
        self.max_propagation_satellites = propagation_config.get('max_satellites', 24)
        self.intra_orbit_links = propagation_config.get('intra_orbit_links', True)
        self.inter_orbit_links = propagation_config.get('inter_orbit_links', True)
        self.link_reliability = propagation_config.get('link_reliability', 0.95)
        self.energy_per_hop = propagation_config.get('energy_per_hop', 0.05)
        
        # 初始化卫星邻居关系字典
        self.satellite_neighbors = {}
        
        self.logger.info(f"初始化有限传播FedProx实验")
        self.logger.info(f"- 传播跳数: {self.propagation_hops}")
        self.logger.info(f"- 最大卫星数: {self.max_propagation_satellites}")
        self.logger.info(f"- 轨道内链接: {self.intra_orbit_links}")
        self.logger.info(f"- 跨轨道链接: {self.inter_orbit_links}")
    
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
            
            # 寻找同一轨道的邻居卫星
            for neighbor_num in range(1, self.config['fl']['satellites_per_orbit'] + 1):
                if neighbor_num == sat_num:
                    continue  # 跳过自己
                    
                neighbor_id = f"satellite_{orbit_id}-{neighbor_num}"
                if neighbor_id in satellites:
                    self.satellite_neighbors[sat_id].append(neighbor_id)
            
            # 添加不同轨道的邻居（如果配置启用了跨轨道链接）
            if self.inter_orbit_links and self.propagation_hops > 1:
                for other_orbit in range(1, self.config['fl']['num_orbits'] + 1):
                    if other_orbit == orbit_id:
                        continue  # 跳过同一轨道
                        
                    # 添加其他轨道上对应位置的卫星作为邻居
                    other_sat_id = f"satellite_{other_orbit}-{sat_num}"
                    if other_sat_id in satellites:
                        self.satellite_neighbors[sat_id].append(other_sat_id)
        
        # 打印网络统计信息
        total_edges = sum(len(neighbors) for neighbors in self.satellite_neighbors.values())
        avg_neighbors = total_edges / len(self.satellite_neighbors) if self.satellite_neighbors else 0
        self.logger.info(f"卫星网络构建完成: {len(self.satellite_neighbors)} 个节点, 平均 {avg_neighbors:.2f} 个邻居")
    
    def _get_propagation_satellites(self, visible_satellites, max_count):
        """
        从可见卫星开始，获取可传播到的卫星
        
        Args:
            visible_satellites: 对地面站可见的卫星列表
            max_count: 最大传播卫星数量
            
        Returns:
            传播卫星集合
        """
        if not self.satellite_neighbors:
            self._build_satellite_network()
            
        # 已经访问的卫星集合
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
    
    def train(self):
        """执行有限传播FedProx训练过程"""
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
        
        # FedProx特有统计
        proximal_terms = []
        
        # 初始化网络拓扑结构
        self._build_satellite_network()
        self.logger.info(f"已构建卫星网络拓扑结构，卫星间邻居关系已建立")
        
        current_time = datetime.now().timestamp()
        self.current_round = 0
        best_accuracy = 0
        rounds_without_improvement = 0

        best_f1 = 0
        
        # 禁用早停或修改参数
        max_rounds_without_improvement = float('inf')  # 设置为无穷大
        min_rounds = self.config['fl']['num_rounds']   # 最小轮数设为总轮数
        accuracy_threshold = 100.0                     # 设置一个不可能达到的准确率阈值
        
        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"=== FedProx限制传播: 开始第 {round_num + 1} 轮训练 (μ={self.mu}) === 时间：{datetime.fromtimestamp(current_time)}")
            
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
            
            # 收集本轮的接近性项统计
            round_proximal_term = 0.0
            round_proximal_samples = 0
            
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
                            
                            # 收集FedProx特有统计 - 接近性项
                            client = self.clients[sat_id]
                            if hasattr(client, 'last_proximal_term'):
                                round_proximal_term += client.last_proximal_term
                                round_proximal_samples += 1
                            
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
            
            # 6. 收集可见训练卫星的模型更新并进行FedAvg聚合
            if visible_trained:
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
                
                # FedAvg聚合（加权平均）
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

                # 7. 更新全局模型 - 关键修改：保留批量归一化层的统计数据
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    # 如果是BatchNorm层的运行统计数据，保留原值
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        aggregated_update[name] = param

                # 更新全局模型
                self.model.load_state_dict(aggregated_update)
                
                # 8. 评估全局模型
                # accuracy = self.evaluate()
                # accuracies.append(accuracy)
                metrics = self.evaluate() 
                # 收集所有指标
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
                
                # 记录接近性项统计
                if round_proximal_samples > 0:
                    avg_proximal = round_proximal_term / round_proximal_samples
                    proximal_terms.append(avg_proximal)
                    self.logger.info(f"轮次 {round_num + 1} 平均接近性项: {avg_proximal:.6f}")
                else:
                    proximal_terms.append(0.0)
                
                # self.logger.info(f"第 {round_num + 1} 轮全局准确率: {accuracy:.4f}")
                self.logger.info(f"第 {round_num + 1} 轮指标: "
                           f"准确率={metrics['accuracy']:.2f}%, "
                           f"F1={metrics['f1_macro']:.2f}%, "
                           f"精确率={metrics['precision_macro']:.2f}%, "
                           f"召回率={metrics['recall_macro']:.2f}%")
                
                # 记录能源和卫星统计信息
                energy_stats['training_energy'].append(round_training_energy)
                energy_stats['communication_energy'].append(round_comm_energy)
                energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
                
                satellite_stats['training_satellites'].append(len(round_training_satellites))
                satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
                satellite_stats['total_active'].append(len(round_training_satellites | round_receiving_sats))
                

                current_accuracy = metrics['accuracy']
                current_f1 = metrics['f1_macro']
                # 更新最佳准确率和检查提升情况
                improvement_found = False
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    improvement_found = True
                    self.logger.info(f"找到更好的模型！新的最佳F1值: {current_f1:.2f}%")
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    if not improvement_found:
                        improvement_found = True
                        self.logger.info(f"找到更好的模型！新的最佳准确率: {current_accuracy:.2f}%")
                
                if improvement_found:
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    self.logger.info(f"性能未提升，已经 {rounds_without_improvement} 轮没有改进")
            else:
                self.logger.warning(f"没有足够的已训练卫星可见，跳过本轮聚合")
            # 调整仿真时间到下一轮开始
            current_time = datetime.now().timestamp() + round_num * self.config['fl']['round_interval']
        
        self.logger.info(f"\n=== 限制传播FedProx训练结束 ===")
        self.logger.info(f"总轮次: {self.current_round + 1}")
        self.logger.info(f"最佳准确率: {best_accuracy:.4f}")
        self.logger.info(f"最佳F1值: {best_f1:.4f}")  # 新增
        self.logger.info(f"接近性参数 μ: {self.mu}")
        
        # 保存接近性项统计
        self.proximal_stats['round_proximal_terms'] = proximal_terms
        
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
            'proximal_terms': proximal_terms,
            'mu': self.mu
        }

        # 生成可视化
        self.visualizer.plot_training_metrics(
            accuracies=stats['accuracies'],
            losses=stats['losses'],
            energy_stats=stats['energy_stats'],
            satellite_stats=stats['satellite_stats'],
            save_path=self.log_dir / 'training_metrics.png'  # 保存在实验日志目录
        )

        return stats
    
    def _get_visible_satellites(self, current_time: float):
        """
        获取当前时间点可见的卫星
        Args:
            current_time: 当前时间戳
        Returns:
            可见卫星ID列表
        """
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
    
    def _train_satellite(self, sat_id, round_number):
        """
        训练单个卫星
        Args:
            sat_id: 卫星ID
            round_number: 当前轮次
        Returns:
            Tuple[bool, Dict]: (是否成功, 训练统计信息)
        """
        try:
            if sat_id not in self.clients:
                return False, {}
                
            stats = self.clients[sat_id].train(round_number)
            
            # 检查训练是否成功以及stats是否有预期的结构
            if not stats or 'summary' not in stats or not stats['summary'].get('train_loss'):
                return False, {}
                
            return True, stats
        except Exception as e:
            self.logger.error(f"训练卫星 {sat_id} 出错: {str(e)}")
            return False, {}