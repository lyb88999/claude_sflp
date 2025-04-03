from experiments.fedavg_experiment import FedAvgExperiment
import logging
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class LimitedPropagationFedAvg(FedAvgExperiment):
    """
    有限传播FedAvg变体 - 允许模型在卫星间有限范围传播
    """
    
    def __init__(self, config_path: str = "configs/propagation_fedavg_config.yaml"):
        """初始化有限传播FedAvg实验"""
        super().__init__(config_path)
        
        # 获取传播相关配置
        propagation_config = self.config.get('propagation', {})
        self.propagation_hops = propagation_config.get('hops', 1)
        self.max_propagation_satellites = propagation_config.get('max_satellites', 24)
        self.intra_orbit_links = propagation_config.get('intra_orbit_links', True)
        self.inter_orbit_links = propagation_config.get('inter_orbit_links', True)
        self.link_reliability = propagation_config.get('link_reliability', 0.95)
        self.energy_per_hop = propagation_config.get('energy_per_hop', 0.05)
        
        # 记录卫星间通信的网络拓扑
        self.satellite_neighbors = {}  # {sat_id: [neighbor_ids]}
        # 初始化卫星邻居关系字典
        self.satellite_neighbors = {}
        
        self.logger.info(f"初始化有限传播FedAvg实验")
        self.logger.info(f"- 传播跳数: {self.propagation_hops}")
        self.logger.info(f"- 最大卫星数: {self.max_propagation_satellites}")
        self.logger.info(f"- 轨道内链接: {self.intra_orbit_links}")
        self.logger.info(f"- 跨轨道链接: {self.inter_orbit_links}")
        
        self.logger.info(f"初始化有限传播FedAvg实验，传播跳数: {self.propagation_hops}，最大卫星数: {self.max_propagation_satellites}")
    
    def _setup_logging(self):
        """设置日志"""
        experiment_type = "propagation_fedavg"
        super()._setup_logging()
    
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
            
            # 也可以添加不同轨道的邻居（如果需要）
            if self.propagation_hops > 1:
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
            self.logger.info(f"卫星网络构建完成，共 {len(self.satellite_neighbors)} 个节点")

            
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
    
    def _handle_orbit_training(self, station_id, orbit_id, current_time):
        """
        处理单个轨道的训练过程，允许模型在卫星间传播
        
        Args:
            station_id: 地面站ID
            orbit_id: 轨道ID
            current_time: 当前时间戳
            
        Returns:
            bool: 训练是否成功完成
            dict: 轨道统计信息
        """
        try:
            station = self.ground_stations[station_id]
            orbit_satellites = self._get_orbit_satellites(orbit_id)
            self.logger.info(f"\n=== 处理轨道 {orbit_id + 1} ===")
            
            # 记录本轮轨道的统计信息
            orbit_stats = {
                'training_energy': 0,  # 训练能耗
                'communication_energy': 0,  # 通信能耗
                'training_satellites': set(),  # 训练的卫星
                'receiving_satellites': set()  # 接收参数的卫星
            }
            
            # 1. 获取可见卫星
            visible_satellites = []
            max_wait_time = current_time + self.config['fl']['round_interval'] * 0.5
            
            while len(visible_satellites) == 0 and current_time < max_wait_time:
                for sat_id in orbit_satellites:
                    if self.network_model._check_visibility(station_id, sat_id, current_time):
                        visible_satellites.append(sat_id)
                
                if len(visible_satellites) == 0:
                    self.logger.info(f"轨道 {orbit_id + 1} 当前无可见卫星，等待60秒...")
                    current_time += 60
                    self.topology_manager.update_topology(current_time)
            
            # 如果找不到可见卫星，则失败
            if len(visible_satellites) == 0:
                self.logger.warning(f"轨道 {orbit_id + 1} 在指定时间内未找到可见卫星")
                return False, orbit_stats
            
            self.logger.info(f"轨道 {orbit_id + 1} 发现 {len(visible_satellites)} 个可见卫星")
            
            # 2. 获取全局模型
            model_state = self.model.state_dict()
            
            # 3. 确定参与传播的卫星数量限制
            max_satellites = getattr(self, 'max_propagation_satellites', 22)
            
            # 4. 首先将模型传播给可见卫星
            propagation_satellites = set(visible_satellites)
            self.logger.info(f"首先向 {len(visible_satellites)} 个可见卫星传递模型")
            
            for sat_id in visible_satellites:
                # 记录通信能耗
                pre_comm_energy = self.energy_model.get_battery_level(sat_id)
                self.clients[sat_id].apply_model_update(model_state)
                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                orbit_stats['communication_energy'] += (pre_comm_energy - post_comm_energy)
                orbit_stats['receiving_satellites'].add(sat_id)
            
            # 5. 如果可见卫星数量小于目标数量，进行卫星间传播
            if len(propagation_satellites) < max_satellites:
                self.logger.info(f"从可见卫星开始进行卫星间传播，目标卫星数: {max_satellites}")
                
                # 构建卫星邻居关系网络（如果尚未构建）
                if not hasattr(self, 'satellite_neighbors') or not self.satellite_neighbors:
                    self._build_satellite_network()
                
                # 广度优先搜索传播，直到达到目标卫星数或无法进一步传播
                frontier = set(visible_satellites)  # 当前边界节点
                visited = set(visible_satellites)   # 已访问节点
                
                # 计算能进行几跳传播
                hops = self.config.get('propagation', {}).get('hops', 1)
                
                for hop in range(hops):
                    if len(propagation_satellites) >= max_satellites:
                        break
                        
                    self.logger.info(f"执行第 {hop+1} 跳传播, 当前有 {len(propagation_satellites)} 颗卫星")
                    next_frontier = set()
                    
                    for sat_id in frontier:
                        # 获取当前卫星的邻居节点
                        neighbors = self.satellite_neighbors.get(sat_id, [])
                        
                        for neighbor in neighbors:
                            # 如果邻居尚未收到模型且未超过最大卫星数
                            if (neighbor not in visited and 
                                neighbor in orbit_satellites and
                                len(propagation_satellites) < max_satellites):
                                
                                # 传播模型给邻居
                                self.logger.info(f"卫星 {sat_id} 将模型传递给邻居 {neighbor}")
                                
                                # 记录通信能耗
                                pre_comm_energy = self.energy_model.get_battery_level(sat_id)  # 发送方消耗能量
                                self.clients[neighbor].apply_model_update(model_state)
                                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                                orbit_stats['communication_energy'] += (pre_comm_energy - post_comm_energy)
                                orbit_stats['receiving_satellites'].add(neighbor)
                                
                                # 添加到已传播集合和下一轮边界
                                propagation_satellites.add(neighbor)
                                visited.add(neighbor)
                                next_frontier.add(neighbor)
                                
                                # 检查是否达到最大卫星数
                                if len(propagation_satellites) >= max_satellites:
                                    self.logger.info(f"已达到最大卫星数 {max_satellites}")
                                    break
                    
                    # 更新边界
                    frontier = next_frontier
                    if not frontier:  # 如果没有新的边界节点，停止传播
                        self.logger.info("没有新的卫星可以传播，停止传播过程")
                        break
                
                self.logger.info(f"卫星间传播完成，共有 {len(propagation_satellites)} 个卫星接收到模型")
            self.logger.info(f"*** 卫星间传播状态 ***")
            self.logger.info(f"初始可见卫星: {len(visible_satellites)}颗") 
            self.logger.info(f"传播后总卫星: {len(propagation_satellites)}颗")
            self.logger.info(f"目标卫星数: {max_satellites}颗")
            
            # 6. 所有接收到模型的卫星都参与训练
            self.logger.info(f"\n=== 轨道 {orbit_id + 1} 卫星训练 ===")
            self.logger.info(f"共有 {len(propagation_satellites)} 个卫星参与训练")
            
            # 并行训练所有卫星
            trained_satellites = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_satellite = {
                    executor.submit(self._train_satellite, sat_id, self.current_round): sat_id
                    for sat_id in propagation_satellites
                }
                
                for future in as_completed(future_to_satellite):
                    sat_id = future_to_satellite[future]
                    try:
                        success, stats = future.result()
                        if success:
                            trained_satellites.append(sat_id)
                            orbit_stats['training_energy'] += stats['energy_consumption']
                            orbit_stats['training_satellites'].add(sat_id)
                            self.logger.info(f"卫星 {sat_id} 完成训练，Loss={stats['summary']['train_loss'][-1]:.4f}")
                    except Exception as e:
                        self.logger.error(f"训练卫星 {sat_id} 出错: {str(e)}")
            
            self.logger.info(f"完成训练的卫星数量: {len(trained_satellites)}")
            
            # 7. 轨道内聚合
            min_updates_required = self.config['aggregation'].get('min_updates', 2)
            self.logger.info(f"需要至少 {min_updates_required} 个卫星更新，当前有 {len(trained_satellites)} 个")

            if len(trained_satellites) >= min_updates_required:
                self.logger.info(f"\n=== 轨道 {orbit_id + 1} 聚合 ===")
                
                # 获取或创建聚合器
                aggregator = self.intra_orbit_aggregators.get(orbit_id)
                if not aggregator:
                    from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
                    aggregator = IntraOrbitAggregator(AggregationConfig(**self.config['aggregation']))
                    self.intra_orbit_aggregators[orbit_id] = aggregator

                # 收集训练卫星的更新并聚合
                updates_collected = 0
                for sat_id in trained_satellites:
                    try:
                        model_diff, stats = self.clients[sat_id].get_model_update()
                        if model_diff:
                            self.logger.info(f"收集卫星 {sat_id} 的模型更新")
                            aggregator.receive_update(sat_id, self.current_round, model_diff, current_time)
                            updates_collected += 1
                    except Exception as e:
                        self.logger.error(f"收集卫星 {sat_id} 更新时出错: {str(e)}")

                self.logger.info(f"成功收集了 {updates_collected} 个卫星的更新")

                # 获取聚合结果
                orbit_update = aggregator.get_aggregated_update(self.current_round)
                if orbit_update:
                    self.logger.info(f"轨道 {orbit_id + 1} 完成聚合")
                    
                    # 8. 更新所有参与传播的卫星的模型
                    update_success = 0
                    for sat_id in propagation_satellites:
                        try:
                            self.clients[sat_id].apply_model_update(orbit_update)
                            update_success += 1
                            orbit_stats['receiving_satellites'].add(sat_id)
                        except Exception as e:
                            self.logger.error(f"更新卫星 {sat_id} 模型时出错: {str(e)}")

                    self.logger.info(f"成功更新了 {update_success} 个卫星的模型")

                    # 9. 等待可见性窗口发送轨道聚合结果到地面站
                    visibility_start = current_time
                    best_visibility_time = None
                    max_search_time = 300  # 5分钟搜索窗口

                    # 先搜索一个最佳的可见性时间点
                    for check_time in range(int(visibility_start), int(visibility_start + max_search_time), 30):
                        for sat_id in visible_satellites:  # 使用原始可见卫星
                            if self.network_model._check_visibility(station_id, sat_id, check_time):
                                best_visibility_time = check_time
                                visibility_sat = sat_id
                                break
                        if best_visibility_time is not None:
                            break

                    if best_visibility_time is not None:
                        current_time = best_visibility_time
                        self.topology_manager.update_topology(current_time)

                        # 10. 发送轨道聚合结果到地面站
                        try:
                            # 使用可见卫星发送更新
                            model_diff, _ = self.clients[visibility_sat].get_model_update()
                            if model_diff:
                                success = station.receive_orbit_update(
                                    str(orbit_id),
                                    self.current_round,
                                    model_diff,
                                    len(trained_satellites)
                                )
                                if success:
                                    self.logger.info(f"轨道 {orbit_id + 1} 的模型成功发送给地面站 {station_id}")
                                    return True, orbit_stats
                                else:
                                    self.logger.error(f"地面站 {station_id} 拒绝接收轨道 {orbit_id + 1} 的更新")
                        except Exception as e:
                            self.logger.error(f"发送模型到地面站时出错: {str(e)}")
                    else:
                        self.logger.error(f"找不到合适的可见性窗口将轨道 {orbit_id + 1} 的更新发送回地面站")
                else:
                    self.logger.error(f"轨道 {orbit_id + 1} 聚合失败: 无法获取有效的聚合结果")
            else:
                self.logger.warning(f"轨道 {orbit_id + 1} 训练的卫星数量不足: {len(trained_satellites)} < {min_updates_required}")

            return False, orbit_stats
            
        except Exception as e:
            self.logger.error(f"处理轨道 {orbit_id + 1} 训练出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, orbit_stats

    def _select_satellites_forced(self, orbit_id, target_count=None):
        """
        强制选择指定数量的卫星，不考虑可见性限制
        
        Args:
            orbit_id: 轨道ID
            target_count: 目标卫星数量，如果为None则使用配置中的max_satellites
            
        Returns:
            List[str]: 选择的卫星ID列表
        """
        if target_count is None:
            target_count = self.max_propagation_satellites
            
        self.logger.info(f"强制选择 {target_count} 颗卫星用于轨道 {orbit_id+1}")
        
        # 获取该轨道的所有卫星
        orbit_satellites = self._get_orbit_satellites(orbit_id)
        
        # 如果目标数量大于轨道卫星数量，也获取其他轨道的卫星
        if target_count > len(orbit_satellites):
            additional_needed = target_count - len(orbit_satellites)
            self.logger.info(f"需要从其他轨道添加 {additional_needed} 颗卫星")
            
            other_satellites = []
            for other_orbit in range(self.config['fl']['num_orbits']):
                if other_orbit != orbit_id:
                    other_satellites.extend(self._get_orbit_satellites(other_orbit))
            
            # 打乱其他轨道的卫星顺序
            import random
            random.shuffle(other_satellites)
            
            # 合并卫星列表
            all_satellites = orbit_satellites + other_satellites[:additional_needed]
        else:
            # 如果目标数量小于等于轨道卫星数量，随机选择
            import random
            random.shuffle(orbit_satellites)
            all_satellites = orbit_satellites[:target_count]
        
        self.logger.info(f"最终选择 {len(all_satellites)} 颗卫星")
        return all_satellites

    def _build_satellite_network(self):
        """构建更丰富的卫星间通信网络"""
        self.logger.info("构建卫星间通信网络...")
        
        # 创建卫星距离矩阵
        satellites = list(self.clients.keys())
        self.satellite_neighbors = {}
        
        for sat_id in satellites:
            self.satellite_neighbors[sat_id] = []
            
            # 解析卫星轨道和编号
            parts = sat_id.split('_')[1].split('-')
            if len(parts) != 2:
                continue
                
            orbit_id, sat_num = int(parts[0]), int(parts[1])
            satellites_per_orbit = self.config['fl']['satellites_per_orbit']
            
            # 添加同轨道邻居 (包括距离=2的卫星)
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
        
        # 打印几个卫星的邻居示例
        for i, sat_id in enumerate(satellites[:3]):
            self.logger.info(f"卫星 {sat_id} 的邻居: {self.satellite_neighbors[sat_id]}")

    def train(self):
        """执行有限传播FedAvg训练过程"""
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
        
        # 初始化网络拓扑结构
        self._build_satellite_network()
        self.logger.info(f"已构建卫星网络拓扑结构，卫星间邻居关系已建立")
        
        current_time = datetime.now().timestamp()
        
        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"=== 开始第 {round_num + 1} 轮训练 === 时间：{datetime.fromtimestamp(current_time)}")
            
            # 1. 确定当前可见的卫星
            visible_satellites = self._get_visible_satellites(current_time)
            
            if not visible_satellites:
                self.logger.warning(f"当前时间点没有可见卫星，等待60秒")
                current_time += 60  # 等待60秒
                self.topology_manager.update_topology(current_time)  # 更新拓扑
                continue
                
            self.logger.info(f"当前有 {len(visible_satellites)} 颗卫星可见")

            # 2. 使用有限传播扩展卫星集合
            max_sats = self.config['propagation']['max_satellites']
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
            
            # 6. 收集可见训练卫星的模型更新并进行FedAvg聚合
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

                # 7. 更新全局模型 - 确保保留BatchNorm层统计数据
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    # 如果是BatchNorm层的运行统计数据，保留原值
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        aggregated_update[name] = param

                # 更新全局模型
                self.model.load_state_dict(aggregated_update)
                
                # 8. 评估全局模型
                accuracy = self.evaluate()
                accuracies.append(accuracy)
                
                # 计算平均损失
                round_loss = 0
                for sat_id in trained_satellites:
                    if self.clients[sat_id].train_stats:
                        round_loss += self.clients[sat_id].train_stats[-1]['summary']['train_loss'][-1]
                losses.append(round_loss / len(trained_satellites))
                
                self.logger.info(f"第 {round_num + 1} 轮全局准确率: {accuracy:.4f}")
                
                # 记录能源和卫星统计信息
                energy_stats['training_energy'].append(round_training_energy)
                energy_stats['communication_energy'].append(round_comm_energy)
                energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
                
                satellite_stats['training_satellites'].append(len(round_training_satellites))
                satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
                satellite_stats['total_active'].append(len(round_training_satellites | round_receiving_sats))
            else:
                self.logger.warning(f"没有足够的已训练卫星可见，跳过本轮聚合")
            
            # 调整仿真时间到下一轮开始
            current_time = datetime.now().timestamp() + round_num * self.config['fl']['round_interval']
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats
        }
        
        return stats
    
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