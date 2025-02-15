from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple, Dict
import random
from torchvision import datasets, transforms
class SatelliteDataset(Dataset):
    """卫星数据集"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NonIIDGenerator:
    """非独立同分布数据生成器"""
    def __init__(self, num_satellites: int, feature_dim: int = 10, num_classes: int = 2):
        """
        初始化数据生成器
        Args:
            num_satellites: 卫星节点数量
            feature_dim: 特征维度
            num_classes: 类别数量
        """
        self.num_satellites = num_satellites
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.random_state = np.random.RandomState(42)
        
    def generate_data(self, total_samples: int,
                 dirichlet_alpha: float = 0.5,
                 mean_samples_per_satellite: int = 1000,
                 std_samples: int = 200,
                 num_satellites: int = None) -> Dict[str, SatelliteDataset]:
        """
        生成非独立同分布的数据
        Args:
            total_samples: 总样本数
            dirichlet_alpha: Dirichlet分布的alpha参数
            mean_samples_per_satellite: 每个卫星平均样本数
            std_samples: 样本数的标准差
            num_satellites: 指定卫星数量（如果不指定则使用self.num_satellites）
        """
        if num_satellites is not None:
            self.num_satellites = num_satellites
        # 生成基础数据
        features, labels = self._generate_base_data(total_samples)
        
        # 使用Dirichlet分布生成类别分布
        class_priors = self.random_state.dirichlet(
            [dirichlet_alpha] * self.num_classes, 
            size=self.num_satellites
        )
        
        # 为每个卫星分配数据量
        samples_per_satellite = self._generate_sample_sizes(
            mean_samples_per_satellite,
            std_samples
        )
        
        # 分配数据
        satellite_datasets = {}
        current_idx = 0
        
        for i in range(self.num_satellites):
            # 确定此卫星的样本数
            num_samples = samples_per_satellite[i]
            
            # 根据类别分布选择样本
            satellite_indices = []
            for class_idx in range(self.num_classes):
                # 计算此类别需要的样本数
                class_samples = int(num_samples * class_priors[i][class_idx])
                # 从对应类别中选择样本
                class_indices = np.where(labels == class_idx)[0]
                if len(class_indices) > 0:
                    selected_indices = self.random_state.choice(
                        class_indices,
                        size=min(class_samples, len(class_indices)),
                        replace=False
                    )
                    satellite_indices.extend(selected_indices)
            
            # 打乱选择的样本顺序
            self.random_state.shuffle(satellite_indices)
            
            # 创建卫星数据集
            sat_features = features[satellite_indices]
            sat_labels = labels[satellite_indices]
            
            satellite_datasets[f"satellite_{i+1}"] = SatelliteDataset(
                torch.FloatTensor(sat_features),
                torch.LongTensor(sat_labels)
            )
            
        return satellite_datasets
        
    def _generate_base_data(self, total_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成基础数据"""
        # 为每个类别生成高斯分布的数据
        features = []
        labels = []
        samples_per_class = total_samples // self.num_classes
        
        for class_idx in range(self.num_classes):
            # 为每个类别生成不同均值的数据
            mean = self.random_state.randn(self.feature_dim) * 2
            cov = self.random_state.rand(self.feature_dim, self.feature_dim)
            cov = np.dot(cov, cov.T)  # 确保协方差矩阵是半正定的
            
            class_features = self.random_state.multivariate_normal(
                mean,
                cov,
                size=samples_per_class
            )
            
            features.append(class_features)
            labels.extend([class_idx] * samples_per_class)
            
        return np.vstack(features), np.array(labels)
        
    def _generate_sample_sizes(self, mean_samples: int, std_samples: int) -> List[int]:
        """生成每个卫星的样本数量"""
        sizes = self.random_state.normal(mean_samples, std_samples, self.num_satellites)
        # 确保至少有batch_size个样本
        sizes = np.maximum(sizes, 32)  # 使用典型的batch_size作为最小值
        return sizes.astype(int)
        
    def generate_test_data(self, num_samples: int = 1000) -> SatelliteDataset:
        """生成测试数据集"""
        features, labels = self._generate_base_data(num_samples)
        return SatelliteDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels)
        )
    
    def generate_empty_dataset(self) -> SatelliteDataset:
        """生成空数据集"""
        return SatelliteDataset(
            features=torch.FloatTensor(0, self.feature_dim),
            labels=torch.LongTensor(0)
        )
    
class MNISTDataGenerator:
    def __init__(self, num_satellites: int):
        self.num_satellites = num_satellites
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载MNIST数据集
        self.train_dataset = datasets.MNIST('../data', train=True, download=True,
                                        transform=self.transform)
        self.test_dataset = datasets.MNIST('../data', train=False,
                                        transform=self.transform)

    def generate_non_iid_data(self, 
                            dirichlet_alpha: float = 0.5,
                            mean_samples_per_satellite: int = 1000) -> Dict[str, Dataset]:
        """生成非独立同分布的MNIST数据"""
        # 按标签分组数据
        labels = self.train_dataset.targets.numpy()
        label_indices = [np.where(labels == i)[0] for i in range(10)]

        # 使用Dirichlet分布生成每个卫星的标签分布
        label_distribution = np.random.dirichlet([dirichlet_alpha] * 10, 
                                            size=self.num_satellites)

        # 创建与原代码相同格式的数据集字典
        satellite_datasets = {}
        for sat_idx in range(self.num_satellites):
            target_size = int(mean_samples_per_satellite * 
                            (0.8 + 0.4 * np.random.random()))  # 随机化数据量
            
            sat_indices = []
            for label, indices in enumerate(label_indices):
                num_samples = int(target_size * label_distribution[sat_idx][label])
                if len(indices) > 0:
                    selected = np.random.choice(indices, 
                                            size=min(num_samples, len(indices)),
                                            replace=False)
                    sat_indices.extend(selected)

            # 创建卫星数据集
            satellite_datasets[f"satellite_{sat_idx+1}"] = CustomMNISTDataset(
                [self.train_dataset[i] for i in sat_indices]
            )

        return satellite_datasets

    def get_test_dataset(self) -> Dataset:
        """获取测试数据集"""
        return self.test_dataset

class CustomMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]
        return image, target