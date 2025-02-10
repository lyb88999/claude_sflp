from typing import List, Dict, Set, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import pdist, squareform

@dataclass
class Link:
    """链路信息"""
    source: str
    target: str
    quality: float
    delay: float
    bandwidth: float

@dataclass
class Group:
    """卫星分组信息"""
    group_id: str
    members: List[str]
    leader: str
    orbit_plane: float  # 轨道平面角度
    avg_connectivity: float

class TopologyManager:
    def __init__(self, network_model, comm_scheduler, energy_model):
        """
        初始化拓扑管理器
        Args:
            network_model: 卫星网络模型实例
            comm_scheduler: 通信调度器实例
            energy_model: 能源模型实例
        """
        self.network_model = network_model
        self.comm_scheduler = comm_scheduler
        self.energy_model = energy_model
        
        self.groups = {}  # 分组信息
        self.routing_table = {}  # 路由表
        self.topology_graph = nx.Graph()  # 网络拓扑图
        self.link_states = {}  # 链路状态
        
    def update_topology(self, current_time: float, window: float = 300):
        """
        更新网络拓扑
        Args:
            current_time: 当前时间戳
            window: 预测窗口大小(秒)
        """
        print("\n=== 开始更新拓扑 ===")
        satellites = list(self.network_model.satellites.keys())
        n_sats = len(satellites)
        print(f"卫星总数: {n_sats}")
        
        # 构建连接矩阵
        connectivity = np.zeros((n_sats, n_sats))
        quality_matrix = np.zeros((n_sats, n_sats))
        
        # 计算采样点
        sample_times = np.linspace(current_time, current_time + window, 5)
        print(f"采样时间点数量: {len(sample_times)}")
        
        # 更新链路状态
        visible_links = 0
        for i, sat1 in enumerate(satellites):
            pos1 = self.network_model.compute_position(sat1, current_time)
            print(f"\n检查卫星 {sat1} 的连接 (位置: {pos1})")
            
            for j, sat2 in enumerate(satellites[i+1:], i+1):
                pos2 = self.network_model.compute_position(sat2, current_time)
                distance = np.linalg.norm(pos2 - pos1)
                print(f"  与 {sat2} 的距离: {distance:.2f}km")
                
                # 检查多个时间点的可见性
                visible_count = 0
                quality_sum = 0.0
                
                for t in sample_times:
                    if self.network_model.check_visibility(sat1, sat2, t):
                        visible_count += 1
                        pos1_t = self.network_model.compute_position(sat1, t)
                        pos2_t = self.network_model.compute_position(sat2, t)
                        distance = np.linalg.norm(pos2_t - pos1_t)
                        quality = max(0, 1 - distance/6000.0)
                        quality_sum += quality
                
                if visible_count > 0:
                    visible_links += 1
                    connectivity[i][j] = connectivity[j][i] = 1
                    
                    # 计算基础质量（根据距离）
                    base_quality = max(0.1, 1 - distance/8000.0)  # 确保最小质量为0.1
                    
                    # 根据可见时间点数调整质量
                    visibility_factor = visible_count / len(sample_times)
                    quality = base_quality * visibility_factor
                    
                    quality_matrix[i][j] = quality_matrix[j][i] = quality
                    
                    print(f"    可见! 可见时间点数: {visible_count}, 平均质量: {quality:.3f}")
                    
                    delay = self._estimate_link_delay(sat1, sat2, current_time)
                    bandwidth = 100.0 * quality  # 最大带宽100Mbps
                    
                    self.link_states[(sat1, sat2)] = Link(
                        source=sat1,
                        target=sat2,
                        quality=quality,
                        delay=delay,
                        bandwidth=bandwidth
                    )
                else:
                    print(f"    不可见")
        
        print(f"\n总计发现可见链路数: {visible_links}")
        print("连接矩阵:")
        print(connectivity)
        
        # 更新拓扑图
        self.topology_graph.clear()
        for i, sat1 in enumerate(satellites):
            for j, sat2 in enumerate(satellites):
                if connectivity[i][j] > 0:
                    self.topology_graph.add_edge(
                        sat1, sat2,
                        weight=1/quality_matrix[i][j]
                    )
        
        print(f"拓扑图边数: {self.topology_graph.number_of_edges()}")
        
        # 更新分组
        self._update_groups(satellites, connectivity, quality_matrix)
        
        # 更新路由表
        self._update_routing_table()
        
    def _estimate_link_delay(self, sat1: str, sat2: str, time: float) -> float:
        """估计链路延迟(ms)"""
        pos1 = self.network_model.compute_position(sat1, time)
        pos2 = self.network_model.compute_position(sat2, time)
        
        # 计算传播延迟
        distance = np.linalg.norm(pos1 - pos2)
        propagation_delay = (distance / 299792.458)  # 光速传播延迟(ms)
        
        # 添加处理延迟和排队延迟
        processing_delay = 5.0  # 假设5ms的处理延迟
        queueing_delay = 10.0   # 假设10ms的排队延迟
        
        return propagation_delay + processing_delay + queueing_delay
        
    def _update_groups(self, satellites: List[str],
                      connectivity: np.ndarray,
                      quality_matrix: np.ndarray):
        """
        更新卫星分组
        使用谱聚类算法进行分组
        """
        # 构建亲和度矩阵
        affinity = quality_matrix * connectivity
        
        # 计算拉普拉斯矩阵
        degree = np.sum(affinity, axis=1)
        laplacian = np.diag(degree) - affinity
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # 确定最佳分组数
        n_groups = self._determine_group_number(eigenvalues)
        
        # 使用k-means进行分组
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        group_labels = kmeans.fit_predict(eigenvectors[:, 1:n_groups])
        
        # 更新分组信息
        self.groups.clear()
        for group_id in range(n_groups):
            members = [sat for sat, label in zip(satellites, group_labels) 
                      if label == group_id]
            
            # 选择组长（连接性最好的节点）
            leader = self._select_group_leader(members, connectivity, satellites)
            
            # 计算平均连接性
            group_indices = [satellites.index(sat) for sat in members]
            avg_connectivity = np.mean(connectivity[group_indices][:, group_indices])
            
            # 计算轨道平面
            orbit_plane = self._calculate_orbit_plane(members)
            
            self.groups[str(group_id)] = Group(
                group_id=str(group_id),
                members=members,
                leader=leader,
                orbit_plane=orbit_plane,
                avg_connectivity=avg_connectivity
            )
            
    def _determine_group_number(self, eigenvalues: np.ndarray) -> int:
        """
        使用特征值确定最佳分组数
        基于特征值差异最大的位置
        """
        gaps = np.diff(eigenvalues)
        # 选择最大间隔位置作为分组数
        n_groups = np.argmax(gaps) + 1
        # 限制分组数在合理范围内
        return max(min(n_groups, 10), 2)
        
    def _select_group_leader(self, members: List[str],
                           connectivity: np.ndarray,
                           satellites: List[str]) -> str:
        """选择组长"""
        if not members:
            return None
            
        member_indices = [satellites.index(sat) for sat in members]
        
        # 计算每个成员的连接度
        degrees = []
        for member in members:
            degree = 0
            member_idx = satellites.index(member)
            for other in members:
                other_idx = satellites.index(other)
                if connectivity[member_idx][other_idx] > 0:
                    degree += 1
            degrees.append(degree)
        
        # 转换为numpy数组
        degrees = np.array(degrees)
        
        # 获取能量水平
        energy_levels = []
        for member in members:
            try:
                energy = self.energy_model.get_battery_level(member)
                energy_levels.append(energy)
            except Exception:
                energy_levels.append(0.0)
        
        energy_levels = np.array(energy_levels)
        
        # 避免除以零
        max_degree = max(np.max(degrees), 1)
        max_energy = max(np.max(energy_levels), 1)
        
        # 综合考虑连接度和能量状态
        scores = 0.7 * (degrees / max_degree) + 0.3 * (energy_levels / max_energy)
        
        leader_idx = np.argmax(scores)
        return members[leader_idx]
        
    def _calculate_orbit_plane(self, satellites: List[str]) -> float:
        """计算卫星组的平均轨道平面角度"""
        angles = []
        for sat in satellites:
            normal = self.network_model.get_orbit_plane(sat)
            angle = np.arctan2(normal[1], normal[0])
            angles.append(angle)
        return np.mean(angles)
        
    def _update_routing_table(self):
        """更新路由表"""
        # 使用Dijkstra算法计算所有节点对之间的最短路径
        self.routing_table.clear()
        
        for source in self.topology_graph.nodes():
            # 计算到所有目标的最短路径
            paths = nx.single_source_dijkstra_path(
                self.topology_graph, source, weight='weight'
            )
            self.routing_table[source] = paths
            
    def get_next_hop(self, source: str, target: str) -> str:
        """获取下一跳节点"""
        if source not in self.routing_table or \
           target not in self.routing_table[source]:
            return None
            
        path = self.routing_table[source][target]
        return path[1] if len(path) > 1 else target
        
    def get_path_quality(self, path: List[str]) -> float:
        """计算路径质量"""
        if len(path) < 2:
            return 1.0
            
        qualities = []
        for i in range(len(path) - 1):
            link = tuple(sorted([path[i], path[i+1]]))
            if link in self.link_states:
                qualities.append(self.link_states[link].quality)
                
        return np.prod(qualities) if qualities else 0.0
        
    def get_group_members(self, group_id: str) -> List[str]:
        """获取分组成员"""
        return self.groups[group_id].members if group_id in self.groups else []
        
    def get_group_leader(self, group_id: str) -> str:
        """获取分组组长"""
        return self.groups[group_id].leader if group_id in self.groups else None
        
    def get_satellite_group(self, satellite: str) -> str:
        """获取卫星所属分组"""
        for group_id, group in self.groups.items():
            if satellite in group.members:
                return group_id
        return None
        
    def optimize_topology(self):
        """
        优化网络拓扑
        - 负载均衡
        - 能量感知路由
        - 故障恢复
        """
        # 计算节点负载
        loads = defaultdict(float)
        for paths in self.routing_table.values():
            for path in paths.values():
                for node in path[1:-1]:  # 不包括源节点和目标节点
                    loads[node] += 1
                    
        # 标准化负载
        max_load = max(loads.values()) if loads else 1
        normalized_loads = {k: v/max_load for k, v in loads.items()}
        
        # 更新图的边权重，考虑负载和能量
        for edge in self.topology_graph.edges():
            sat1, sat2 = edge
            link = tuple(sorted([sat1, sat2]))
            
            if link in self.link_states:
                quality = self.link_states[link].quality
                # 计算新的权重
                load_factor = (normalized_loads.get(sat1, 0) + 
                             normalized_loads.get(sat2, 0)) / 2
                energy_factor = (
                    self.energy_model.get_transmission_capacity(sat1) *
                    self.energy_model.get_transmission_capacity(sat2)
                ) ** 0.5
                
                # 更新边权重
                weight = (1 / quality) * (1 + 0.3 * load_factor) / energy_factor
                self.topology_graph[sat1][sat2]['weight'] = weight
                
        # 重新计算路由表
        self._update_routing_table()