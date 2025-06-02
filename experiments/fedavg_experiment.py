from datetime import datetime
import random
from typing import List
from experiments.baseline_experiment import BaselineExperiment


class FedAvgExperiment(BaselineExperiment):
    def __init__(self, config_path: str = "configs/fedavg_config.yaml"):
        super().__init__(config_path)
        self.max_workers = 6
        
    def train(self):
        # 初始化记录列表
        accuracies = []
        losses = []
         # 新增的分类指标列表
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

            participation_rate = self.config['fedavg']['participation_rate']
            num_to_select = max(2, int(len(visible_satellites) * participation_rate))
            # num_to_select = min(num_to_select, len(visible_satellites))

            import random
            participating_satellites = random.sample(visible_satellites, num_to_select)

            self.logger.info(f"选择了 {len(participating_satellites)}/{len(visible_satellites)} 颗卫星 " +
                            f"(参与率: {participation_rate})")
            
            # 2. 分发全局模型参数给可见卫星
            global_model = self.model.state_dict()
            round_comm_energy = 0
            round_receiving_sats = set()
            
            for sat_id in visible_satellites:
                # 记录通信能耗
                pre_comm_energy = self.energy_model.get_battery_level(sat_id)
                self.clients[sat_id].apply_model_update(global_model)
                post_comm_energy = self.energy_model.get_battery_level(sat_id)
                
                round_comm_energy += (pre_comm_energy - post_comm_energy)
                round_receiving_sats.add(sat_id)
                
                self.logger.info(f"分发模型参数给卫星 {sat_id}")
            
            # 3. 所有可见卫星进行本地训练
            round_training_energy = 0
            round_training_satellites = set()
            trained_satellites = []
            
            for sat_id in visible_satellites:
                pre_train_energy = self.energy_model.get_battery_level(sat_id)
                stats = self.clients[sat_id].train(round_num)
                
                if stats['summary']['train_loss']:
                    post_train_energy = self.energy_model.get_battery_level(sat_id)
                    energy_used = pre_train_energy - post_train_energy
                    round_training_energy += energy_used
                    round_training_satellites.add(sat_id)
                    trained_satellites.append(sat_id)
                    
                    self.logger.info(f"卫星 {sat_id} 完成训练: "
                                  f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                  f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                                  f"能耗={energy_used:.4f}Wh")
            
            # 更新时间，模拟通信和训练的时间消耗
            communication_time = 60  # 假设通信需要1分钟
            training_time = 300      # 假设训练需要5分钟
            current_time += communication_time + training_time
            
            # 4. 等待下次卫星可见性窗口，收集模型更新
            # 为了模拟实际情况，需要前进时间直到足够的卫星再次可见
            while True:
                visible_satellites = self._get_visible_satellites(current_time)                # 计算已训练且当前可见的卫星集合
                visible_trained = [sat for sat in trained_satellites if sat in visible_satellites]
                
                if len(visible_trained) >= self.config['aggregation']['min_updates']:
                    self.logger.info(f"当前有 {len(visible_trained)} 颗已训练的卫星可见，可以进行聚合")
                    break
                
                # 如果没有足够的卫星可见，前进时间
                current_time += 60  # 每次前进1分钟
                if current_time - (datetime.now().timestamp() + communication_time + training_time) > self.config['aggregation']['timeout']:
                    self.logger.warning(f"等待聚合超时，使用当前可见的卫星进行聚合")
                    break
            
            # 5. 收集可见训练卫星的模型更新并进行FedAvg聚合
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

                # 6. 更新全局模型 - 修改后的代码
                current_state_dict = self.model.state_dict()
                for name, param in current_state_dict.items():
                    # 如果是BatchNorm层的运行统计数据，保留原值
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        aggregated_update[name] = param

                # 然后再更新模型
                self.model.load_state_dict(aggregated_update)
                
                # 7. 评估全局模型
                # accuracy = self.evaluate()
                # accuracies.append(accuracy)
                metrics = self.evaluate()  # 现在返回完整的指标字典
                
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
                for sat_id in visible_trained:
                    if self.clients[sat_id].train_stats:
                        round_loss += self.clients[sat_id].train_stats[-1]['summary']['train_loss'][-1]
                losses.append(round_loss / len(visible_trained))
                
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
            else:
                self.logger.warning(f"没有足够的已训练卫星可见，跳过本轮聚合")
            
            # 调整仿真时间到下一轮开始
            current_time = datetime.now().timestamp() + round_num * self.config['fl']['round_interval']
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
            # 新增的分类指标
            'precision_macros': precision_macros,
            'recall_macros': recall_macros,
            'f1_macros': f1_macros,
            'precision_weighteds': precision_weighteds,
            'recall_weighteds': recall_weighteds,
            'f1_weighteds': f1_weighteds,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats
        }
        
        return stats
    
    def _get_visible_satellites(self, current_time: float) -> List[str]:
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
                self.logger.debug(f"检查可见性: {station_id} -> {sat_id}: {is_visible}")
                if is_visible:
                    visible_satellites.append(sat_id)
                    break  # 只要有一个地面站可见就可以
        
        self.logger.info(f"当前时间 {datetime.fromtimestamp(current_time)} 可见卫星数: {len(visible_satellites)}")
        return visible_satellites