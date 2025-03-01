import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random

class NetworkTrafficDataset(Dataset):
    """网络流量数据集"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NetworkTrafficGenerator:
    """网络流量数据生成器，考虑轨道内相邻卫星的数据相似性"""
    def __init__(self, num_satellites: int, num_orbits: int, satellites_per_orbit: int):
        """
        初始化数据生成器
        Args:
            num_satellites: 卫星节点总数
            num_orbits: 轨道数量
            satellites_per_orbit: 每个轨道的卫星数量
        """
        self.num_satellites = num_satellites
        self.num_orbits = num_orbits
        self.satellites_per_orbit = satellites_per_orbit
        self.feature_dim = 10  # 网络流量特征维度（如包大小、时间间隔、端口等）
        self.num_classes = 2   # 二分类：正常流量 vs 恶意流量
        self.random_state = np.random.RandomState(42)
        
        # 生成轨道和卫星索引映射
        self.orbit_satellite_map = {}
        for orbit in range(1, num_orbits+1):
            for pos in range(1, satellites_per_orbit+1):
                sat_id = f"satellite_{orbit}-{pos}"
                self.orbit_satellite_map[sat_id] = (orbit, pos)
        
    def generate_orbit_base_traffic_patterns(self, num_patterns_per_orbit=3):
        """
        为每个轨道生成基础流量模式
        Args:
            num_patterns_per_orbit: 每个轨道的基础模式数量
        Returns:
            Dict: 轨道ID -> 流量模式列表
        """
        orbit_patterns = {}
        
        for orbit in range(1, self.num_orbits+1):
            # 为每个轨道生成特定的流量模式
            patterns = []
            for i in range(num_patterns_per_orbit):
                # 正常流量模式
                normal_mean = self.random_state.rand(self.feature_dim) * 2 - 1
                normal_cov = self.random_state.rand(self.feature_dim, self.feature_dim)
                normal_cov = np.dot(normal_cov, normal_cov.T)  # 确保协方差矩阵是半正定的
                
                # 恶意流量模式 - 与正常流量有明显区别
                malicious_mean = normal_mean + (self.random_state.rand(self.feature_dim) * 4 - 2)
                malicious_cov = self.random_state.rand(self.feature_dim, self.feature_dim)
                malicious_cov = np.dot(malicious_cov, malicious_cov.T)
                
                patterns.append({
                    'normal': (normal_mean, normal_cov),
                    'malicious': (malicious_mean, malicious_cov)
                })
            
            orbit_patterns[orbit] = patterns
        
        return orbit_patterns
    
    def generate_position_weights(self, center_pos: int, total_positions: int, similarity_radius: int = 2):
        """
        生成位置权重，使得相邻位置的卫星具有相似的数据
        Args:
            center_pos: 中心位置
            total_positions: 总位置数量
            similarity_radius: 相似性半径
        Returns:
            List[float]: 各位置的权重
        """
        weights = []
        for pos in range(1, total_positions+1):
            # 计算在环形拓扑中的最短距离
            direct_dist = abs(pos - center_pos)
            wrap_dist = total_positions - direct_dist
            min_dist = min(direct_dist, wrap_dist)
            
            # 使用高斯衰减计算权重
            weight = np.exp(-min_dist**2 / (2 * similarity_radius**2))
            weights.append(weight)
            
        # 归一化权重
        weights = np.array(weights)
        return weights / np.sum(weights)
    
    def generate_data(self, 
                      total_samples: int,
                      malicious_ratio: float = 0.3,
                      orbit_similarity: float = 0.7,
                      position_similarity: float = 0.8) -> Dict[str, NetworkTrafficDataset]:
        """
        生成网络流量数据，考虑轨道内相邻卫星的数据相似性
        Args:
            total_samples: 总样本数
            malicious_ratio: 恶意流量比例
            orbit_similarity: 轨道内数据相似度（0-1）
            position_similarity: 位置相似度（0-1）
        Returns:
            Dict[str, NetworkTrafficDataset]: 卫星ID -> 数据集
        """
        # 生成每个轨道的基础流量模式
        orbit_patterns = self.generate_orbit_base_traffic_patterns()
        
        # 为每个卫星生成样本量（遵循长尾分布）
        samples_per_satellite = self._generate_sample_sizes(total_samples)
        
        # 生成每个卫星的数据集
        satellite_datasets = {}
        
        for sat_id, (orbit, position) in self.orbit_satellite_map.items():
            num_samples = samples_per_satellite[sat_id]
            
            # 计算位置权重
            position_weights = self.generate_position_weights(
                position, self.satellites_per_orbit
            )
            
            # 为该卫星生成数据
            features = []
            labels = []
            
            # 决定恶意样本数量
            num_malicious = int(num_samples * malicious_ratio)
            num_normal = num_samples - num_malicious
            
            # 生成正常流量
            normal_features = self._generate_orbit_specific_samples(
                orbit, orbit_patterns, 'normal', num_normal,
                orbit_similarity, position, position_similarity  # 传递position而不是position_weights
            )
            features.append(normal_features)
            labels.extend([0] * num_normal)
            
            # 生成恶意流量
            malicious_features = self._generate_orbit_specific_samples(
                orbit, orbit_patterns, 'malicious', num_malicious,
                orbit_similarity, position, position_similarity  # 传递position而不是position_weights
            )
            features.append(malicious_features)
            labels.extend([1] * num_malicious)
            
            # 合并并打乱数据
            features = np.vstack(features)
            labels = np.array(labels)
            
            # 打乱数据顺序
            indices = np.arange(len(labels))
            self.random_state.shuffle(indices)
            features = features[indices]
            labels = labels[indices]
            
            # 创建数据集
            satellite_datasets[sat_id] = NetworkTrafficDataset(
                torch.FloatTensor(features),
                torch.LongTensor(labels)
            )
            
        return satellite_datasets
    
    def _generate_orbit_specific_samples(self, 
                                 orbit: int,
                                 orbit_patterns: Dict,
                                 traffic_type: str,
                                 num_samples: int,
                                 orbit_similarity: float,
                                 center_position: int,
                                 position_similarity: float) -> np.ndarray:
        """
        生成特定轨道的流量样本
        Args:
            orbit: 轨道ID
            orbit_patterns: 轨道流量模式
            traffic_type: 流量类型（'normal'或'malicious'）
            num_samples: 样本数量
            orbit_similarity: 轨道相似度
            center_position: 中心位置
            position_similarity: 位置相似度
        Returns:
            np.ndarray: 生成的样本特征
        """
        # 获取当前轨道的模式
        orbit_pattern = orbit_patterns[orbit]
        
        # 生成位置权重(在函数内部动态生成)
        position_weights = self.generate_position_weights(
            center_position, 
            self.satellites_per_orbit, 
            similarity_radius=2 if position_similarity > 0.7 else 1
        )
        
        # 截取并重新归一化概率
        # 只保留前len(orbit_pattern)个权重，并重新归一化确保总和为1
        truncated_weights = position_weights[:len(orbit_pattern)]
        truncated_weights = truncated_weights / np.sum(truncated_weights)
        
        # 决定使用自身轨道的模式还是其他轨道的模式
        samples_from_own_orbit = int(num_samples * orbit_similarity)
        samples_from_other_orbits = num_samples - samples_from_own_orbit
        
        # 从自身轨道模式生成样本
        own_orbit_samples = []
        for i in range(samples_from_own_orbit):
            # 使用重新归一化后的权重
            pattern_idx = self.random_state.choice(len(orbit_pattern), p=truncated_weights)
            pattern = orbit_pattern[pattern_idx]
            
            mean, cov = pattern[traffic_type]
            sample = self.random_state.multivariate_normal(mean, cov)
            own_orbit_samples.append(sample)
        
        # 从其他轨道模式生成样本（如果需要）
        other_orbit_samples = []
        if samples_from_other_orbits > 0:
            # 选择其他轨道
            other_orbits = [o for o in range(1, self.num_orbits+1) if o != orbit]
            
            for i in range(samples_from_other_orbits):
                # 随机选择一个其他轨道
                other_orbit = self.random_state.choice(other_orbits)
                other_pattern = orbit_patterns[other_orbit]
                
                # 随机选择该轨道的一个模式
                pattern_idx = self.random_state.choice(len(other_pattern))
                pattern = other_pattern[pattern_idx]
                
                mean, cov = pattern[traffic_type]
                sample = self.random_state.multivariate_normal(mean, cov)
                other_orbit_samples.append(sample)
                
        # 合并样本
        all_samples = np.vstack([own_orbit_samples, other_orbit_samples]) if other_orbit_samples else np.array(own_orbit_samples)
        return all_samples
    
    def _generate_sample_sizes(self, total_samples: int) -> Dict[str, int]:
        """
        生成更合理的样本分配
        """
        # 创建结果字典
        sample_sizes = {}
        
        # 首先确定每个轨道的基础样本数 (略有差异但不会相差太多)
        orbit_base_samples = {}
        orbit_variance_factor = 0.3  # 控制轨道间差异
        
        for orbit in range(1, self.num_orbits+1):
            # 使用正弦函数产生波动，而不是单调递减
            # 这里用正弦函数只是为了产生波动，也可以用其他方式
            factor = 1.0 + orbit_variance_factor * np.sin(orbit * np.pi / 3)
            orbit_base_samples[orbit] = factor
        
        # 归一化轨道基础样本数
        total_factor = sum(orbit_base_samples.values())
        for orbit in orbit_base_samples:
            orbit_base_samples[orbit] = orbit_base_samples[orbit] / total_factor
        
        # 为每个轨道分配总样本数
        orbit_total_samples = {}
        remaining_samples = total_samples
        
        for orbit in range(1, self.num_orbits):  # 先分配前n-1个轨道
            orbit_samples = int(total_samples * orbit_base_samples[orbit])
            orbit_total_samples[orbit] = orbit_samples
            remaining_samples -= orbit_samples
        
        # 最后一个轨道分配剩余样本，确保总数正确
        orbit_total_samples[self.num_orbits] = remaining_samples
        
        # 在轨道内分配样本，使相邻卫星有相似数量
        for orbit in range(1, self.num_orbits+1):
            orbit_samples = orbit_total_samples[orbit]
            position_samples = {}
            
            # 生成轨道内的位置分布 (使用平滑的高斯形状)
            for pos in range(1, self.satellites_per_orbit+1):
                # 在轨道中创建2-3个"热点"区域，其他区域较少样本
                # 使用多个高斯分布的叠加来实现
                angle = 2 * np.pi * pos / self.satellites_per_orbit
                dist1 = np.exp(-0.5 * ((angle - np.pi/2) / 0.7)**2)  # 第一个高斯
                dist2 = np.exp(-0.5 * ((angle - 3*np.pi/2) / 0.7)**2)  # 第二个高斯
                position_samples[pos] = 0.5 * dist1 + 0.5 * dist2
            
            # 归一化位置样本分布
            total_weights = sum(position_samples.values())
            for pos in position_samples:
                position_samples[pos] = position_samples[pos] / total_weights
            
            # 分配轨道内样本
            remaining = orbit_samples
            for pos in range(1, self.satellites_per_orbit):  # 先分配前n-1个位置
                sat_samples = max(100, int(orbit_samples * position_samples[pos]))
                if sat_samples > remaining - 100:  # 确保留出至少100个样本给最后一个位置
                    sat_samples = remaining - 100
                
                sat_id = f"satellite_{orbit}-{pos}"
                sample_sizes[sat_id] = sat_samples
                remaining -= sat_samples
            
            # 最后一个位置分配剩余样本
            sat_id = f"satellite_{orbit}-{self.satellites_per_orbit}"
            sample_sizes[sat_id] = remaining
        
        # 添加随机扰动 (±10%)
        for sat_id in sample_sizes:
            base_samples = sample_sizes[sat_id]
            # 添加±10%的随机变化
            random_factor = 0.9 + 0.2 * self.random_state.random()  # 0.9到1.1之间
            # 确保至少有最小样本数
            sample_sizes[sat_id] = max(100, int(base_samples * random_factor))
        
        # 调整总样本数与目标一致
        current_total = sum(sample_sizes.values())
        if current_total != total_samples:
            # 按比例调整
            adjustment_factor = total_samples / current_total
            for sat_id in sample_sizes:
                sample_sizes[sat_id] = max(100, int(sample_sizes[sat_id] * adjustment_factor))
            
            # 处理舍入误差
            diff = total_samples - sum(sample_sizes.values())
            if diff != 0:
                # 随机选择卫星调整差值
                adjust_candidates = list(sample_sizes.keys())
                self.random_state.shuffle(adjust_candidates)
                for sat_id in adjust_candidates:
                    if diff > 0:
                        sample_sizes[sat_id] += 1
                        diff -= 1
                    elif diff < 0:
                        if sample_sizes[sat_id] > 100:  # 确保不低于最小值
                            sample_sizes[sat_id] -= 1
                            diff += 1
                    
                    if diff == 0:
                        break
        
        return sample_sizes
        
    def generate_test_data(self, num_samples: int = 1000) -> NetworkTrafficDataset:
        """生成测试数据集"""
        # 平均所有轨道的模式来生成测试数据
        orbit_patterns = self.generate_orbit_base_traffic_patterns()
        
        features = []
        labels = []
        
        # 正常流量样本
        num_normal = int(num_samples * 0.7)  # 70%正常流量
        normal_features = []
        
        for orbit in range(1, self.num_orbits+1):
            patterns = orbit_patterns[orbit]
            orbit_samples = num_normal // self.num_orbits
            
            for i in range(orbit_samples):
                pattern_idx = self.random_state.randint(0, len(patterns))
                mean, cov = patterns[pattern_idx]['normal']
                sample = self.random_state.multivariate_normal(mean, cov)
                normal_features.append(sample)
                
        # 恶意流量样本
        num_malicious = num_samples - num_normal
        malicious_features = []
        
        for orbit in range(1, self.num_orbits+1):
            patterns = orbit_patterns[orbit]
            orbit_samples = num_malicious // self.num_orbits
            
            for i in range(orbit_samples):
                pattern_idx = self.random_state.randint(0, len(patterns))
                mean, cov = patterns[pattern_idx]['malicious']
                sample = self.random_state.multivariate_normal(mean, cov)
                malicious_features.append(sample)
        
        # 合并所有样本
        features = np.vstack([normal_features, malicious_features])
        labels = np.array([0] * len(normal_features) + [1] * len(malicious_features))
        
        # 打乱数据
        indices = np.arange(len(labels))
        self.random_state.shuffle(indices)
        features = features[indices]
        labels = labels[indices]
        
        return NetworkTrafficDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels)
        )

class SimpleTrafficModel(torch.nn.Module):
    """简单的网络流量分类模型"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 2):
        super().__init__()
        self.__init__args__ = (input_dim,)
        self.__init__kwargs__ = {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes
        }
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)