import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import os
from typing import Dict, List, Tuple

class TrafficFlowDataset(Dataset):
    """卫星网络流量数据集"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RealTrafficGenerator:
    """真实流量数据生成器"""
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
        self.feature_dim = None
        self.num_classes = None
        self.scaler = None
        self.label_encoder = None
        self.random_state = np.random.RandomState(42)
        
    def load_and_preprocess_data(self, csv_file: str, test_size: float = 0.2):
        """
        加载并预处理单个CSV文件数据
        
        Args:
            csv_file: CSV文件路径
            test_size: 测试集比例
                
        Returns:
            Tuple: (特征维度, 类别数)
        """
        print(f"加载CSV文件: {csv_file}")
        
        try:
            # 读取CSV文件 - 对于大文件，使用更高效的方法
            # 首先读取小样本来确定数据类型
            df_sample = pd.read_csv(csv_file, nrows=1000)
            
            # 确定数值型列，只对这些列应用类型转换
            numeric_cols = df_sample.select_dtypes(include=['float64', 'int64']).columns
            
            # 创建列类型字典，将数值型列设为更高效的类型
            dtypes = {col: 'float32' if col in numeric_cols else 'object' for col in df_sample.columns}
            
            # 使用类型字典和分块读取来处理大文件
            print("开始分块读取CSV文件...")
            chunks = pd.read_csv(csv_file, dtype=dtypes, chunksize=100000)
            combined_df = pd.concat(chunks, ignore_index=True)
            
            print(f"成功加载数据，形状: {combined_df.shape}")
        except Exception as e:
            print(f"加载 {csv_file} 出错: {str(e)}")
            raise
            
        # 检查和处理缺失值
        missing_values = combined_df.isnull().sum()
        print(f"缺失值统计:\n{missing_values[missing_values > 0]}")
        
        # 用列的中位数填充数值型特征的缺失值
        for col in combined_df.select_dtypes(include=['float32', 'float64', 'int64']).columns:
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        # 提取特征和标签
        if 'Label' not in combined_df.columns:
            raise ValueError("数据中缺少'Label'列")
            
        X = combined_df.drop(['Label'], axis=1)
        y = combined_df['Label']
        
        # 移除非数值列(如果有)
        non_numeric_cols = X.select_dtypes(exclude=['float32', 'float64', 'int64']).columns
        if len(non_numeric_cols) > 0:
            print(f"移除非数值列: {non_numeric_cols}")
            X = X.drop(non_numeric_cols, axis=1)
        
        # 统计标签分布
        label_counts = y.value_counts()
        print(f"标签分布:\n{label_counts}")
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"类别编码: {dict(zip(self.label_encoder.classes_, range(self.num_classes)))}")
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_dim = X.shape[1]
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
        
        # 转换为PyTorch张量
        self.X_train_tensor = torch.FloatTensor(X_train)
        self.y_train_tensor = torch.LongTensor(y_train)
        self.X_test_tensor = torch.FloatTensor(X_test)
        self.y_test_tensor = torch.LongTensor(y_test)
        
        print(f"训练集: {len(self.X_train_tensor)}个样本, 测试集: {len(self.X_test_tensor)}个样本")
        print(f"特征维度: {self.feature_dim}, 类别数: {self.num_classes}")
        
        return self.feature_dim, self.num_classes
    
    def generate_data(self, iid: bool = True, alpha: float = 1.0) -> Dict[str, TrafficFlowDataset]:
        """
        生成并分配数据给卫星
        
        Args:
            iid: 是否为独立同分布数据
            alpha: Dirichlet分布参数(仅在non-iid时使用)
            
        Returns:
            Dict: 卫星ID -> 数据集
        """
        if not hasattr(self, 'X_train_tensor'):
            raise ValueError("请先调用load_and_preprocess_data加载数据")
            
        print(f"为 {self.num_satellites} 个卫星分配数据, IID={iid}")
        
        # 获取所有索引
        all_indices = list(range(len(self.X_train_tensor)))
        self.random_state.shuffle(all_indices)
        
        satellite_datasets = {}
        
        if iid:
            # IID分配: 随机均匀分配
            indices_per_satellite = len(all_indices) // self.num_satellites
            remaining = len(all_indices) % self.num_satellites
            
            start_idx = 0
            for orbit in range(1, self.num_orbits + 1):
                for sat in range(1, self.satellites_per_orbit + 1):
                    sat_id = f"satellite_{orbit}-{sat}"
                    sat_idx = (orbit - 1) * self.satellites_per_orbit + (sat - 1)
                    
                    # 确定该卫星分配的样本数
                    extra = 1 if sat_idx < remaining else 0
                    num_samples = indices_per_satellite + extra
                    
                    # 选择样本
                    if start_idx + num_samples <= len(all_indices):
                        satellite_indices = all_indices[start_idx:start_idx + num_samples]
                        start_idx += num_samples
                        
                        # 创建卫星数据集
                        sat_features = self.X_train_tensor[satellite_indices]
                        sat_labels = self.y_train_tensor[satellite_indices]
                        satellite_datasets[sat_id] = TrafficFlowDataset(sat_features, sat_labels)
                        
                        print(f"为 {sat_id} 分配 {len(satellite_indices)} 个样本")
        else:
            # Non-IID分配: 使用Dirichlet分布
            # 按标签分组
            label_indices = {}
            for i, label in enumerate(self.y_train_tensor):
                label_item = label.item()
                if label_item not in label_indices:
                    label_indices[label_item] = []
                label_indices[label_item].append(i)
            
            # 使用Dirichlet分布来分配每个卫星的标签比例
            label_distribution = np.random.dirichlet(
                [alpha] * self.num_satellites, 
                size=self.num_classes
            )
            
            # 分配数据
            for orbit in range(1, self.num_orbits + 1):
                for sat in range(1, self.satellites_per_orbit + 1):
                    sat_id = f"satellite_{orbit}-{sat}"
                    sat_idx = (orbit - 1) * self.satellites_per_orbit + (sat - 1)
                    
                    if sat_idx < self.num_satellites:
                        satellite_indices = []
                        
                        # 为每个标签分配样本
                        for label, indices in label_indices.items():
                            # 计算该卫星应获取的该标签样本数
                            sat_prop = label_distribution[label][sat_idx]
                            num_samples = int(sat_prop * len(indices))
                            
                            # 随机选择样本
                            if num_samples > 0 and indices:
                                selected = self.random_state.choice(
                                    indices, 
                                    min(num_samples, len(indices)), 
                                    replace=False
                                )
                                satellite_indices.extend(selected)
                                # 从可用索引中移除已选择的样本
                                indices = list(set(indices) - set(selected))
                                label_indices[label] = indices
                        
                        # 创建卫星数据集
                        if satellite_indices:
                            sat_features = self.X_train_tensor[satellite_indices]
                            sat_labels = self.y_train_tensor[satellite_indices]
                            satellite_datasets[sat_id] = TrafficFlowDataset(sat_features, sat_labels)
                            
                            label_dist = torch.bincount(sat_labels, minlength=self.num_classes)
                            print(f"为 {sat_id} 分配 {len(satellite_indices)} 个样本, 标签分布: {label_dist}")
        
        return satellite_datasets
    
    def generate_region_similar_data(self, iid: bool = False, alpha: float = 0.6, overlap_ratio: float = 0.5) -> Dict[str, TrafficFlowDataset]:
        """
        生成具有区域相似性的数据分布 (修改版)
        
        Args:
            iid: 是否为独立同分布数据（在本方法中不起作用，保留参数是为了保持接口一致）
            alpha: Dirichlet参数（控制非IID程度）
            overlap_ratio: 区域内数据重叠比例（0-1之间）
            
        Returns:
            Dict: 卫星ID -> 数据集
        """
        if not hasattr(self, 'X_train_tensor'):
            raise ValueError("请先调用load_and_preprocess_data加载数据")
            
        print(f"为 {self.num_satellites} 个卫星生成具有区域相似性的数据分布，重叠比例: {overlap_ratio}")
        
        # 按轨道分组卫星
        orbit_satellites = {}
        for orbit in range(1, self.num_orbits + 1):
            orbit_satellites[orbit] = []
            for sat in range(1, self.satellites_per_orbit + 1):
                orbit_satellites[orbit].append(f"satellite_{orbit}-{sat}")
        
        # 获取所有特征和标签
        all_features = self.X_train_tensor
        all_labels = self.y_train_tensor
        
        # 计算每个轨道分配的样本数
        total_samples = len(all_features)
        samples_per_orbit = total_samples // self.num_orbits
        
        # 为每个轨道创建特定的数据偏移
        orbit_shifts = {}
        for orbit in range(1, self.num_orbits + 1):
            # 生成一个统一的区域偏移向量
            if orbit == 1:  # 为区域1创建较小的偏移，使其更接近原始数据
                orbit_shifts[orbit] = np.random.uniform(-0.1, 0.1, size=self.feature_dim)
            else:
                orbit_shifts[orbit] = np.random.uniform(-0.5, 0.5, size=self.feature_dim)
        
        # 打乱所有数据索引
        all_indices = list(range(total_samples))
        self.random_state.shuffle(all_indices)
        
        # 为每个轨道创建基础数据集
        orbit_data = {}
        start_idx = 0
        for orbit in range(1, self.num_orbits + 1):
            # 随机分配数据，而不是按顺序
            orbit_size = samples_per_orbit
            if orbit == self.num_orbits:  # 最后一个轨道分配剩余的所有样本
                orbit_size = total_samples - start_idx
                
            # 使用打乱后的索引
            orbit_indices = all_indices[start_idx:start_idx + orbit_size]
            orbit_features = all_features[orbit_indices]
            orbit_labels = all_labels[orbit_indices]
            
            # 对区域1进行特殊处理 - 平衡标签分布
            if orbit == 1:
                label_counts = torch.bincount(orbit_labels, minlength=self.num_classes)
                if torch.max(label_counts) > 0:  # 确保有数据
                    # 计算每个标签的比例
                    label_ratios = label_counts.float() / torch.sum(label_counts).float()
                    # 找出小于平均比例的标签
                    avg_ratio = 1.0 / self.num_classes
                    underrepresented = (label_ratios < avg_ratio * 0.7).nonzero(as_tuple=True)[0]
                    
                    # 对不平衡的标签进行过采样
                    if len(underrepresented) > 0:
                        extended_features = []
                        extended_labels = []
                        
                        for label in underrepresented:
                            # 找出此标签的所有索引
                            label_indices = (orbit_labels == label).nonzero(as_tuple=True)[0]
                            
                            # 如果存在此标签的样本，进行过采样
                            if len(label_indices) > 0:
                                target_count = int(avg_ratio * len(orbit_labels) * 0.8)
                                oversample_size = target_count - len(label_indices)
                                
                                if oversample_size > 0:
                                    # 随机选择过采样的样本
                                    choice_indices = self.random_state.choice(
                                        len(label_indices), size=oversample_size, replace=True)
                                    oversample_indices = label_indices[choice_indices]
                                    
                                    # 添加到扩展数据中
                                    extended_features.append(orbit_features[oversample_indices])
                                    extended_labels.append(orbit_labels[oversample_indices])
                        
                        # 合并原始数据和过采样数据
                        if extended_features:
                            orbit_features = torch.cat([orbit_features] + extended_features)
                            orbit_labels = torch.cat([orbit_labels] + extended_labels)
                
                print(f"区域1经过平衡处理后的标签分布: {torch.bincount(orbit_labels, minlength=self.num_classes)}")
            
            # 应用区域特定的偏移
            region_shift = torch.tensor(orbit_shifts[orbit], dtype=torch.float32)
            orbit_features_shifted = orbit_features + region_shift
            
            orbit_data[orbit] = {
                'features': orbit_features_shifted,
                'labels': orbit_labels,
                'size': len(orbit_features_shifted)
            }
            
            start_idx += orbit_size
        
        # 分配具有重叠的数据集
        satellite_datasets = {}
        for orbit, satellites in orbit_satellites.items():
            orbit_features = orbit_data[orbit]['features']
            orbit_labels = orbit_data[orbit]['labels']
            orbit_size = orbit_data[orbit]['size']
            
            # 计算每个卫星的基础样本数
            base_samples_per_sat = orbit_size // len(satellites)
            
            # 计算共享样本数
            shared_size = int(base_samples_per_sat * overlap_ratio)
            
            # 随机选择共享数据而不是总是使用前面的部分
            all_orbit_indices = list(range(len(orbit_features)))
            self.random_state.shuffle(all_orbit_indices)
            
            shared_indices = all_orbit_indices[:shared_size]
            shared_features = orbit_features[shared_indices]
            shared_labels = orbit_labels[shared_indices]
            
            # 剩余可分配的非共享数据
            remaining_indices = all_orbit_indices[shared_size:]
            self.random_state.shuffle(remaining_indices)
            
            # 计算每个卫星应获得的独特数据量
            if len(satellites) > 0:
                unique_indices_per_sat = len(remaining_indices) // len(satellites)
                
                # 为每个卫星分配数据
                for i, sat_id in enumerate(satellites):
                    # 分配共享数据
                    sat_features = [shared_features]
                    sat_labels = [shared_labels]
                    
                    # 分配独特数据
                    start = i * unique_indices_per_sat
                    if i < len(satellites) - 1:
                        end = start + unique_indices_per_sat
                        indices_slice = remaining_indices[start:end]
                    else:
                        # 最后一个卫星获取所有剩余数据
                        indices_slice = remaining_indices[start:]
                    
                    if indices_slice:
                        unique_features = orbit_features[indices_slice]
                        unique_labels = orbit_labels[indices_slice]
                        
                        sat_features.append(unique_features)
                        sat_labels.append(unique_labels)
                    
                    # 合并数据
                    if sat_features:
                        combined_features = torch.cat(sat_features)
                        combined_labels = torch.cat(sat_labels)
                        
                        # 随机打乱数据
                        shuffle_indices = torch.randperm(len(combined_features))
                        shuffle_features = combined_features[shuffle_indices]
                        shuffle_labels = combined_labels[shuffle_indices]
                        
                        satellite_datasets[sat_id] = TrafficFlowDataset(
                            shuffle_features,
                            shuffle_labels
                        )
                        
                        label_dist = torch.bincount(shuffle_labels, minlength=self.num_classes)
                        print(f"卫星 {sat_id} 数据集大小: {len(satellite_datasets[sat_id])}, "
                            f"共享: {len(shared_features)}, 独特: {len(indices_slice)}, "
                            f"标签分布: {label_dist}")
        
        return satellite_datasets
    
    def extract_region_data(self, dataset, orbit_id):
        """
        提取特定区域的数据（用于跨区域性能评估）
        
        Args:
            dataset: 数据集对象
            orbit_id: 轨道ID
            
        Returns:
            特定区域的数据子集
        """
        # 复制一个新的数据集
        region_dataset = TrafficFlowDataset(
            features=dataset.features.clone(),
            labels=dataset.labels.clone()
        )
        
        # 应用相同的区域偏移（与训练数据相同）
        if orbit_id == 1:
            # 区域1使用较小的偏移
            region_shift = np.random.uniform(-0.1, 0.1, size=self.feature_dim)
        else:
            region_shift = np.random.uniform(-0.5, 0.5, size=self.feature_dim)
            
        # 应用偏移到特征
        region_dataset.features += torch.tensor(region_shift, dtype=torch.float32)
        
        return region_dataset
    
    def generate_test_data(self) -> TrafficFlowDataset:
        """生成测试数据集"""
        if not hasattr(self, 'X_test_tensor'):
            raise ValueError("请先调用load_and_preprocess_data加载数据")
            
        return TrafficFlowDataset(self.X_test_tensor, self.y_test_tensor)
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return self.num_classes
        
    def get_class_names(self) -> List[str]:
        """获取类别名称"""
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return []