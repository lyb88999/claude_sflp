import unittest
import os
import torch
import numpy as np
import pandas as pd
import tempfile
from real_traffic_generator import RealTrafficGenerator, TrafficFlowDataset

class TestRealTrafficGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """创建测试用的CSV文件"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.csv_files = []
        
        # 扩展样本数据，增加每个类别的样本数，确保有足够样本分配给所有卫星
        n_samples = 20  # 每个类别的样本数量
        
        # 文件1: 正常流量 - 增加样本数量
        normal_data = {
            'Duration': np.linspace(0.1, 1.0, n_samples),
            'Size': np.linspace(100, 1000, n_samples),
            'Protocol': np.random.choice([6, 17], n_samples),
            'Sinr': np.linspace(10, 20, n_samples),
            'Throughput': np.linspace(5, 10, n_samples),
            'Flow_bytes_s': np.linspace(100, 1000, n_samples),
            'Flow_packets_s': np.linspace(10, 100, n_samples),
            'Inv_mean': np.linspace(0.01, 0.1, n_samples),
            'Inv_min': np.linspace(0.001, 0.01, n_samples),
            'Inv_max': np.linspace(0.1, 1.0, n_samples),
            'DNS_query_id': np.arange(1, n_samples+1),
            'L7_protocol': np.full(n_samples, 7),
            'DNS_type': np.full(n_samples, 1),
            'TTL_min': np.full(n_samples, 64),
            'TTL_max': np.full(n_samples, 64),
            'DNS_TTL_answer': np.full(n_samples, 3600),
            'Next_Current_diff': np.full(n_samples, 0.1),
            'Next_Pre_diff': np.full(n_samples, 0.2),
            'SNext_Current_diff': np.full(n_samples, 0.3),
            'SNext_Pre_diff': np.full(n_samples, 0.4),
            'Label': ['normal'] * n_samples
        }
        
        # 文件2: 恶意流量 - 增加样本数量
        malicious_data = {
            'Duration': np.linspace(1.1, 2.0, n_samples),
            'Size': np.linspace(1100, 2000, n_samples),
            'Protocol': np.random.choice([6, 17], n_samples),
            'Sinr': np.linspace(21, 30, n_samples),
            'Throughput': np.linspace(11, 20, n_samples),
            'Flow_bytes_s': np.linspace(1100, 2000, n_samples),
            'Flow_packets_s': np.linspace(110, 200, n_samples),
            'Inv_mean': np.linspace(0.11, 0.2, n_samples),
            'Inv_min': np.linspace(0.011, 0.02, n_samples),
            'Inv_max': np.linspace(1.1, 2.0, n_samples),
            'DNS_query_id': np.arange(n_samples+1, 2*n_samples+1),
            'L7_protocol': np.full(n_samples, 7),
            'DNS_type': np.full(n_samples, 1),
            'TTL_min': np.full(n_samples, 64),
            'TTL_max': np.full(n_samples, 64),
            'DNS_TTL_answer': np.full(n_samples, 3600),
            'Next_Current_diff': np.full(n_samples, 0.6),
            'Next_Pre_diff': np.full(n_samples, 0.7),
            'SNext_Current_diff': np.full(n_samples, 0.8),
            'SNext_Pre_diff': np.full(n_samples, 0.9),
            'Label': ['malicious'] * n_samples
        }
        
        # 保存测试数据到CSV文件
        normal_df = pd.DataFrame(normal_data)
        malicious_df = pd.DataFrame(malicious_data)
        
        normal_file = os.path.join(cls.temp_dir.name, "normal_traffic.csv")
        malicious_file = os.path.join(cls.temp_dir.name, "malicious_traffic.csv")
        
        normal_df.to_csv(normal_file, index=True)
        malicious_df.to_csv(malicious_file, index=True)
        
        cls.csv_files = [normal_file, malicious_file]
        
        # 初始化生成器
        cls.generator = RealTrafficGenerator(
            num_satellites=6,  # 增加卫星数量
            num_orbits=2,
            satellites_per_orbit=3
        )
        
        # 加载数据
        cls.feature_dim, cls.num_classes = cls.generator.load_and_preprocess_data(
            cls.csv_files,
            test_size=0.2
        )
    
    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        cls.temp_dir.cleanup()
    
    def test_load_and_preprocess_data(self):
        """测试数据加载和预处理功能"""
        # 检查特征维度和类别数量
        self.assertEqual(self.feature_dim, 20)  # 21列减去Label
        self.assertEqual(self.num_classes, 2)   # normal和malicious
        
        # 检查训练集和测试集
        self.assertTrue(hasattr(self.generator, 'X_train_tensor'))
        self.assertTrue(hasattr(self.generator, 'y_train_tensor'))
        self.assertTrue(hasattr(self.generator, 'X_test_tensor'))
        self.assertTrue(hasattr(self.generator, 'y_test_tensor'))
        
        # 检查标签编码器
        self.assertTrue(hasattr(self.generator, 'label_encoder'))
        self.assertIn('normal', self.generator.get_class_names())
        self.assertIn('malicious', self.generator.get_class_names())
        
        # 检查缩放器
        self.assertTrue(hasattr(self.generator, 'scaler'))
    
    def test_generate_iid_data(self):
        """测试IID数据生成"""
        satellite_datasets = self.generator.generate_data(iid=True)
        
        # 检查卫星数量
        expected_satellites = 6  # 2个轨道 * 3个卫星
        self.assertEqual(len(satellite_datasets), expected_satellites)
        
        # 检查卫星ID格式
        for orbit in range(1, 3):
            for sat in range(1, 4):
                sat_id = f"satellite_{orbit}-{sat}"
                self.assertIn(sat_id, satellite_datasets)
        
        # 检查数据集格式
        for sat_id, dataset in satellite_datasets.items():
            self.assertIsInstance(dataset, TrafficFlowDataset)
            self.assertTrue(hasattr(dataset, 'features'))
            self.assertTrue(hasattr(dataset, 'labels'))
            
            # 检查每个卫星的样本数是否大约相等
            total_samples = len(self.generator.X_train_tensor)
            min_expected = total_samples // expected_satellites - 1
            max_expected = total_samples // expected_satellites + 1
            self.assertTrue(
                min_expected <= len(dataset) <= max_expected,
                f"卫星 {sat_id} 的样本数 {len(dataset)} 不在预期范围 {min_expected}-{max_expected} 内"
            )
            
            # 检查特征维度
            self.assertEqual(dataset.features.shape[1], self.feature_dim)
    
    def test_generate_non_iid_data(self):
        """测试非IID数据生成"""
        # 使用较低的alpha值确保明显的Non-IID特性
        satellite_datasets = self.generator.generate_data(iid=False, alpha=0.1)
        
        # 检查是否有卫星分配到数据
        self.assertGreater(len(satellite_datasets), 0, "至少应该有一个卫星分配到数据")
        
        # 收集标签分布信息
        label_distributions = {}
        label_ratios = {}
        
        for sat_id, dataset in satellite_datasets.items():
            labels = dataset.labels.numpy()
            unique, counts = np.unique(labels, return_counts=True)
            label_distributions[sat_id] = dict(zip(unique, counts))
            
            # 计算类别0占比
            total = sum(counts)
            if len(unique) > 1 and 0 in label_distributions[sat_id] and 1 in label_distributions[sat_id]:
                ratio = label_distributions[sat_id][0] / total
                label_ratios[sat_id] = ratio
        
        # 检查是否至少有两个卫星的标签分布不同
        if len(label_ratios) >= 2:
            ratios = list(label_ratios.values())
            max_ratio = max(ratios)
            min_ratio = min(ratios)
            
            # 在非IID情况下，卫星之间的类别分布应该有明显差异
            self.assertGreater(max_ratio - min_ratio, 0.1, 
                              f"不同卫星的标签分布差异不够明显: {label_ratios}")
        else:
            # 如果没有足够的卫星有完整的标签分布，至少检查一些基本属性
            self.assertGreaterEqual(len(satellite_datasets), 1, "应该至少有一个卫星分配到数据")
            
            if len(satellite_datasets) > 1:
                # 比较不同卫星的样本数，应该不均匀
                sample_counts = [len(dataset) for dataset in satellite_datasets.values()]
                max_count = max(sample_counts)
                min_count = min(sample_counts)
                
                # 样本分配应该不均匀
                sample_ratio = min_count / max_count if max_count > 0 else 1
                self.assertLess(sample_ratio, 0.8, f"样本分配应该不均匀，但比例为 {sample_ratio}")
    
    def test_generate_test_data(self):
        """测试测试数据集生成"""
        test_dataset = self.generator.generate_test_data()
        
        # 检查测试集
        self.assertIsInstance(test_dataset, TrafficFlowDataset)
        self.assertEqual(test_dataset.features.shape[1], self.feature_dim)
        
        # 测试集大小应该是原始数据的20%
        expected_test_size = int(40 * 0.2)  # 40个样本 * 0.2
        self.assertEqual(len(test_dataset), expected_test_size)
    
    def test_accessor_methods(self):
        """测试访问器方法"""
        self.assertEqual(self.generator.get_feature_dim(), self.feature_dim)
        self.assertEqual(self.generator.get_num_classes(), self.num_classes)
        
        class_names = self.generator.get_class_names()
        self.assertEqual(len(class_names), self.num_classes)
        self.assertIn('normal', class_names)
        self.assertIn('malicious', class_names)

if __name__ == "__main__":
    unittest.main()