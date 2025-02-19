from baseline_experiment import BaselineExperiment
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

class GroupingExperiment(BaselineExperiment):
    def __init__(self, config_path: str = "configs/grouping_config.yaml"):
        super().__init__(config_path)
        self.group_size = 3  # 固定组大小为3
        self.switch_interval = self.config['group']['switch_interval']
        self.orbit_groups = {}  # {orbit_id: [group1, group2, ...]}
        self.active_representatives = {}  # {orbit_id: {group_id: representative_id}}
        
    def setup_groups(self):
        """设置初始分组"""
        # 固定分组：每个轨道的卫星按顺序分组
        # [1,2,3], [4,5,6], [7,8,9], [10,11]
        for orbit in range(1, 7):  # 轨道编号从1开始
            groups = []
            # 创建完整的三人组
            for i in range(0, 9, 3):
                group = [f"satellite_{orbit}-{j}" for j in range(i+1, i+4)]
                groups.append(group)
            # 添加最后的两人组
            groups.append([f"satellite_{orbit}-10", f"satellite_{orbit}-11"])
            
            self.orbit_groups[orbit] = groups
            
            # 为每个组选择初始代表节点（选择每组的第一个卫星作为代表）
            self.active_representatives[orbit] = {
                i: group[0] for i, group in enumerate(groups)
            }
            
            # 打印分组信息
            self.logger.info(f"轨道 {orbit} 分组情况:")
            for i, group in enumerate(groups):
                rep = self.active_representatives[orbit][i]
                members = [sat for sat in group if sat != rep]
                self.logger.info(f"  组 {i+1}:")
                self.logger.info(f"    代表节点: {rep}")
                self.logger.info(f"    组成员: {members}")

        self.logger.info("完成分组设置")

    def _switch_representatives(self, orbit_id: int):
        """切换代表节点"""
        self.logger.info(f"轨道 {orbit_id} 切换代表节点")
        for group_id, group in enumerate(self.orbit_groups[orbit_id]):
            # 从组内随机选择新的代表节点
            new_rep = random.choice(group)
            old_rep = self.active_representatives[orbit_id][group_id]
            self.active_representatives[orbit_id][group_id] = new_rep
            if new_rep != old_rep:
                self.logger.info(f"  组 {group_id + 1}: 代表节点从 {old_rep} 切换为 {new_rep}")

    def _handle_orbit_training(self, station_id: str, orbit_id: int, current_time: float):
        try:
            station = self.ground_stations[station_id]
            orbit_num = orbit_id + 1
            self.logger.info(f"处理轨道 {orbit_num}")
            
            # 1. 等待并选择可见卫星作为协调者
            coordinator = None
            orbit_satellites = self._get_orbit_satellites(orbit_id)
            max_wait_time = current_time + self.config['fl']['round_interval']
            
            while not coordinator and current_time < max_wait_time:
                for sat_id in orbit_satellites:
                    if self.network_model._check_visibility(station_id, sat_id, current_time):
                        coordinator = sat_id
                        break
                if not coordinator:
                    self.logger.info(f"轨道 {orbit_num} 当前无可见卫星，等待60秒...")
                    current_time += 60
                    self.topology_manager.update_topology(current_time)

            if not coordinator:
                self.logger.warning(f"轨道 {orbit_num} 在指定时间内未找到可见卫星")
                return False

            self.logger.info(f"轨道 {orbit_num} 选择 {coordinator} 作为协调者")

            # 2. 检查是否需要切换代表节点
            if self.current_round % self.switch_interval == 0:
                self._switch_representatives(orbit_num)

            # 3. 分发初始参数给协调者
            model_state = self.model.state_dict()
            self.clients[coordinator].apply_model_update(model_state)
            self.logger.info(f"成功将参数分发给协调者 {coordinator}")


            # 4. 只训练代表节点
            self.logger.info(f"\n=== 轨道 {orbit_num} 训练 ===")
            trained_satellites = set()
            
            # 获取当前轨道的代表节点列表
            representatives = list(self.active_representatives[orbit_num].values())

            # 记录本轮轨道的统计信息
            trained_satellites = set()
            orbit_stats = {
                'training_energy': 0,  # 训练能耗
                'communication_energy': 0,  # 通信能耗
                'training_satellites': set(),  # 训练的卫星
                'receiving_satellites': set()  # 接收参数的卫星
            }
            
            for rep_id in representatives:
                # 确保代表节点有最新的模型参数
                if rep_id != coordinator:
                    self.clients[rep_id].apply_model_update(model_state)

                # 记录训练前的电池电量
                pre_train_energy = self.energy_model.get_battery_level(rep_id)
                
                # 训练代表节点
                stats = self.clients[rep_id].train(self.current_round)
                if stats['summary']['train_loss']:
                    post_train_energy = self.energy_model.get_battery_level(rep_id)
                    orbit_stats['training_energy'] += (pre_train_energy - post_train_energy)
                    orbit_stats['training_satellites'].add(rep_id)
                    trained_satellites.add(rep_id)
                    self.logger.info(f"代表节点 {rep_id} 完成训练: "
                                f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                                f"能耗={stats['summary']['energy_consumption']:.4f}Wh")

                    # 将训练后的模型参数分发给组内其他成员并评估性能
                    group_id = [gid for gid, rid in self.active_representatives[orbit_num].items() if rid == rep_id][0]
                    group_members = self.orbit_groups[orbit_num][group_id]

                    # 记录参数分发的能耗和接收卫星
                    pre_comm_energy = self.energy_model.get_battery_level(rep_id)
                    for member in group_members:
                        if member != rep_id:
                            # 更新成员的模型参数
                            self.clients[member].apply_model_update(self.clients[rep_id].model.state_dict())
                            orbit_stats['receiving_satellites'].add(member)
                            self.logger.info(f"成员 {member} 更新模型参数")
                            
                            # 使用代表节点的模型在成员自己的数据上评估
                            member_dataset = self.clients[member].dataset
                            if member_dataset is not None:
                                # 创建数据加载器
                                test_loader = DataLoader(member_dataset, batch_size=32)
                                member_model = self.clients[member].model
                                member_model.eval()  # 设置为评估模式
                                
                                correct = 0
                                total = 0
                                test_loss = 0
                                
                                with torch.no_grad():
                                    for data, target in test_loader:
                                        output = member_model(data)
                                        test_loss += F.cross_entropy(output, target).item()
                                        pred = output.argmax(dim=1)
                                        total += target.size(0)
                                        correct += pred.eq(target).sum().item()
                                
                                accuracy = 100. * correct / total
                                avg_loss = test_loss / len(test_loader)
                                
                                self.logger.info(f"成员 {member} 使用代表节点 {rep_id} 的模型在自己数据上的性能:")
                                self.logger.info(f"    Loss: {avg_loss:.4f}")
                                self.logger.info(f"    Accuracy: {accuracy:.2f}%")
                    post_comm_energy = self.energy_model.get_battery_level(rep_id)
                    orbit_stats['communication_energy'] += (pre_comm_energy - post_comm_energy)

            # 5. 轨道内聚合
            min_updates_required = self.config['aggregation']['min_updates']
            self.logger.info(f"需要至少 {min_updates_required} 个代表节点更新，当前有 {len(trained_satellites)} 个")

            if len(trained_satellites) >= min_updates_required:
                self.logger.info(f"\n=== 轨道 {orbit_num} 聚合 ===")
                aggregator = self.intra_orbit_aggregators.get(orbit_id)
                if not aggregator:
                    aggregator = IntraOrbitAggregator(AggregationConfig(**self.config['aggregation']))
                    self.intra_orbit_aggregators[orbit_id] = aggregator

                # 收集代表节点的更新并聚合
                for sat_id in trained_satellites:
                    model_diff, _ = self.clients[sat_id].get_model_update()
                    if model_diff:
                        self.logger.info(f"收集代表节点 {sat_id} 的模型更新")
                        aggregator.receive_update(sat_id, self.current_round, model_diff, current_time)

                orbit_update = aggregator.get_aggregated_update(self.current_round)
                if orbit_update:
                    self.logger.info(f"轨道 {orbit_num} 完成聚合")
                    
                    # 更新所有卫星的模型
                    for sat_id in orbit_satellites:
                        try:
                            self.clients[sat_id].apply_model_update(orbit_update)
                            self.logger.info(f"更新卫星 {sat_id} 的模型参数")
                        except Exception as e:
                            self.logger.error(f"更新卫星 {sat_id} 模型时出错: {str(e)}")

                    # 等待可见性并发送模型回地面站
                    visibility_wait_start = current_time
                    max_visibility_wait = 600  # 增加到10分钟
                    
                    while not self.network_model._check_visibility(station_id, coordinator, current_time):
                        if current_time - visibility_wait_start > max_visibility_wait:
                            self.logger.warning(f"等待地面站 {station_id} 可见性超时")
                            return False
                        current_time += 60
                        self.topology_manager.update_topology(current_time)

                    # 发送模型到地面站
                    try:
                        # 使用聚合后的模型更新
                        success = station.receive_orbit_update(
                            str(orbit_id),
                            self.current_round,
                            orbit_update,  # 直接使用聚合后的更新
                            len(trained_satellites)
                        )
                        if success:
                            self.logger.info(f"轨道 {orbit_num} 的模型成功发送给地面站 {station_id}")
                            return True, orbit_stats
                        else:
                            self.logger.error(f"地面站 {station_id} 拒绝接收轨道 {orbit_num} 的更新")
                    except Exception as e:
                        self.logger.error(f"发送模型到地面站时出错: {str(e)}")
                else:
                    self.logger.error(f"轨道 {orbit_num} 聚合失败: 无法获取有效的聚合结果")
            else:
                self.logger.warning(f"轨道 {orbit_num} 训练的代表节点数量不足: {len(trained_satellites)} < {min_updates_required}")

            return False, orbit_stats

        except Exception as e:
            self.logger.error(f"处理轨道 {orbit_num} 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, None
        
    def run(self):
        """运行分组实验"""
        self.logger.info("开始分组实验")
        
        # 准备数据
        self.prepare_data()
        
        # 设置客户端
        self.setup_clients()
        
        # 设置分组（这是分组实验特有的步骤）
        self.setup_groups()
        
        # 执行训练并获取统计信息
        stats = self.train()
        
        self.logger.info("实验完成")
        
        # 返回统计信息供后续比较
        return stats

def main():
    experiment = GroupingExperiment()
    experiment.setup_groups()
    experiment.run()

if __name__ == "__main__":   
    main()