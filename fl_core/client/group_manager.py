from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class GroupConfig:
    """分组配置"""
    min_group_size: int = 2
    max_group_size: int = 10
    similarity_threshold: float = 0.7
    feature_dims: int = 50
    dynamic_adjustment: bool = True
    
class DataSimilarityAnalyzer:
    """数据相似度分析器"""
    def __init__(self, feature_dims: int = 50):
        self.feature_dims = feature_dims
        self.svd = TruncatedSVD(n_components=feature_dims)
        
    def compute_features(self, dataset: Dataset) -> np.ndarray:
        """
        计算数据集的特征表示
        Args:
            dataset: 数据集
        Returns:
            特征矩阵
        """
        # 将数据集转换为矩阵形式
        if isinstance(dataset[0][0], torch.Tensor):
            data_matrix = torch.stack([x[0].flatten() for x in dataset]).numpy()
        else:
            data_matrix = np.stack([x[0].flatten() for x in dataset])
            
        # 使用SVD降维
        try:
            features = self.svd.fit_transform(data_matrix)
            return features
        except Exception as e:
            print(f"特征计算错误: {str(e)}")
            # 如果SVD失败，返回简化的特征
            return data_matrix.mean(axis=1).reshape(-1, 1)
            
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征集之间的相似度
        Args:
            features1: 第一个特征集
            features2: 第二个特征集
        Returns:
            相似度分数
        """
        if len(features1.shape) == 1:
            features1 = features1.reshape(1, -1)
        if len(features2.shape) == 1:
            features2 = features2.reshape(1, -1)
            
        similarity = cosine_similarity(features1.mean(axis=0).reshape(1, -1),
                                    features2.mean(axis=0).reshape(1, -1))
        return float(similarity[0, 0])

class GroupManager:
    def __init__(self, config: GroupConfig):
        """
        初始化群组管理器
        Args:
            config: 分组配置
        """
        self.config = config
        self.analyzer = DataSimilarityAnalyzer(config.feature_dims)
        self.groups = {}  # 分组信息
        self.features_cache = {}  # 特征缓存
        self.similarity_matrix = None  # 相似度矩阵
        
    def analyze_client_data(self, client_id: str, dataset: Dataset):
        """
        分析客户端数据
        Args:
            client_id: 客户端ID
            dataset: 数据集
        """
        features = self.analyzer.compute_features(dataset)
        self.features_cache[client_id] = features
        
    def update_similarity_matrix(self):
        """更新相似度矩阵"""
        client_ids = list(self.features_cache.keys())
        n_clients = len(client_ids)
        
        similarity_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                sim = self.analyzer.compute_similarity(
                    self.features_cache[client_ids[i]],
                    self.features_cache[client_ids[j]]
                )
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim
                
        self.similarity_matrix = similarity_matrix
        
    def form_groups(self) -> Dict[str, List[str]]:
        """
        形成分组
        Returns:
            分组结果，格式: {group_id: [client_ids]}
        """
        if self.similarity_matrix is None:
            self.update_similarity_matrix()
            
        client_ids = list(self.features_cache.keys())
        n_clients = len(client_ids)
        
        # 使用层次聚类或K-means进行分组
        if n_clients <= self.config.max_group_size:
            # 如果客户端数量较少，直接使用相似度阈值分组
            groups = self._threshold_based_grouping()
        else:
            # 否则使用K-means聚类
            n_clusters = max(2, n_clients // self.config.max_group_size)
            groups = self._kmeans_grouping(n_clusters)
            
        self.groups = groups
        return groups
        
    def _threshold_based_grouping(self) -> Dict[str, List[str]]:
        """基于相似度阈值的分组"""
        client_ids = list(self.features_cache.keys())
        n_clients = len(client_ids)
        
        # 初始化分组
        groups = {}
        unassigned = set(range(n_clients))
        group_id = 0
        
        while unassigned:
            current = unassigned.pop()
            current_group = [current]
            
            # 查找相似的客户端
            for other in list(unassigned):
                if self.similarity_matrix[current, other] >= self.config.similarity_threshold:
                    current_group.append(other)
                    unassigned.remove(other)
                    
            # 保存分组
            groups[f"group_{group_id}"] = [client_ids[i] for i in current_group]
            group_id += 1
            
        return groups
        
    def _kmeans_grouping(self, n_clusters: int) -> Dict[str, List[str]]:
        """基于K-means的分组"""
        client_ids = list(self.features_cache.keys())
        features = np.array([self.features_cache[cid].mean(axis=0) 
                           for cid in client_ids])
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # 整理分组结果
        groups = {}
        for i in range(n_clusters):
            group_members = [client_ids[j] for j, label in enumerate(labels) if label == i]
            if group_members:  # 只保存非空分组
                groups[f"group_{i}"] = group_members
                
        return groups
        
    def adjust_groups(self, network_topology: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        根据网络拓扑调整分组
        Args:
            network_topology: 网络拓扑信息，格式: {client_id: [neighbor_ids]}
        Returns:
            调整后的分组
        """
        if not self.config.dynamic_adjustment:
            return self.groups
            
        adjusted_groups = {}
        
        for group_id, members in self.groups.items():
            new_members = []
            for member in members:
                # 检查组内其他成员是否可达
                reachable_members = [m for m in members 
                                   if m in network_topology.get(member, [])]
                                   
                if len(reachable_members) >= self.config.min_group_size - 1:
                    new_members.append(member)
                    
            if len(new_members) >= self.config.min_group_size:
                adjusted_groups[group_id] = new_members
                
        return adjusted_groups
        
    def get_group_stats(self) -> Dict[str, Dict]:
        """
        获取分组统计信息
        Returns:
            分组统计，格式: {group_id: {size, avg_similarity, ...}}
        """
        stats = {}
        client_ids = list(self.features_cache.keys())
        
        for group_id, members in self.groups.items():
            member_indices = [client_ids.index(m) for m in members]
            
            # 计算组内平均相似度
            similarities = []
            for i in range(len(member_indices)):
                for j in range(i + 1, len(member_indices)):
                    idx1, idx2 = member_indices[i], member_indices[j]
                    similarities.append(self.similarity_matrix[idx1, idx2])
                    
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            stats[group_id] = {
                'size': len(members),
                'avg_similarity': avg_similarity,
                'members': members
            }
            
        return stats
        
    def get_client_group(self, client_id: str) -> Optional[str]:
        """
        获取客户端所属的组
        Args:
            client_id: 客户端ID
        Returns:
            组ID，如果不在任何组中则返回None
        """
        for group_id, members in self.groups.items():
            if client_id in members:
                return group_id
        return None