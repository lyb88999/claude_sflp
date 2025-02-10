import unittest
import torch
import numpy as np
from datetime import datetime
from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
from fl_core.aggregation.ground_station import GroundStationAggregator, GroundStationConfig
from fl_core.aggregation.global_aggregator import GlobalAggregator, GlobalConfig

class TestIntraOrbitAggregation(unittest.TestCase):
    def setUp(self):
        config = AggregationConfig(
            min_updates=2,
            max_staleness=300.0,
            timeout=600.0,
            weighted_average=True
        )
        self.aggregator = IntraOrbitAggregator(config)
        
        # 添加测试客户端
        self.aggregator.add_client("client1", 1.0)
        self.aggregator.add_client("client2", 1.0)
        self.aggregator.add_client("client3", 1.0)
        
    def test_client_management(self):
        """测试客户端管理"""
        self.aggregator.add_client("client4", 2.0)
        self.assertEqual(self.aggregator.client_weights["client4"], 2.0)
        
        self.aggregator.remove_client("client4")
        self.assertNotIn("client4", self.aggregator.client_weights)
        
    def test_update_reception(self):
        """测试更新接收"""
        round_number = 1
        model_update = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.ones(10)
        }
        
        success = self.aggregator.receive_update(
            "client1", round_number, model_update,
            datetime.now().timestamp()
        )
        self.assertTrue(success)
        
    def test_aggregation(self):
        """测试聚合过程"""
        round_number = 1
        current_time = datetime.now().timestamp()
        
        # 提交多个更新
        for i, client_id in enumerate(["client1", "client2"]):
            model_update = {
                "layer1.weight": torch.ones(10, 10) * (i + 1),
                "layer1.bias": torch.ones(10) * (i + 1)
            }
            self.aggregator.receive_update(
                client_id, round_number, model_update, current_time
            )
            
        # 获取聚合结果
        result = self.aggregator.get_aggregated_update(round_number)
        self.assertIsNotNone(result)
        self.assertIn("layer1.weight", result)
        self.assertIn("layer1.bias", result)

class TestGroundStationAggregation(unittest.TestCase):
    def setUp(self):
        config = GroundStationConfig(
            bandwidth_limit=100.0,
            storage_limit=1000.0,
            priority_levels=3
        )
        self.aggregator = GroundStationAggregator(config)
        
        # 添加测试轨道
        self.aggregator.add_orbit("orbit1", 1.0)
        self.aggregator.add_orbit("orbit2", 1.0)
        
    def test_orbit_management(self):
        """测试轨道管理"""
        self.aggregator.add_orbit("orbit3", 2.0)
        self.assertEqual(self.aggregator.orbit_weights["orbit3"], 2.0)
        
        self.aggregator.remove_orbit("orbit3")
        self.assertNotIn("orbit3", self.aggregator.orbit_weights)
        
    def test_update_processing(self):
        """测试更新处理"""
        round_number = 1
        model_update = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.ones(10)
        }
        
        success = self.aggregator.receive_orbit_update(
            "orbit1", round_number, model_update, 
            num_clients=5, priority=2
        )
        self.assertTrue(success)
        
    def test_bandwidth_limit(self):
        """测试带宽限制"""
        # 创建大型更新
        large_update = {
            "layer1.weight": torch.ones(1000, 1000),  # 大约4MB
            "layer1.bias": torch.ones(1000)
        }
        
        # 连续发送多个更新
        successes = []
        for i in range(5):
            success = self.aggregator.receive_orbit_update(
                "orbit1", i, large_update,
                num_clients=5, priority=1
            )
            successes.append(success)
            
        # 验证部分更新被拒绝
        self.assertTrue(any(not s for s in successes))

class TestGlobalAggregation(unittest.TestCase):
    def setUp(self):
        config = GlobalConfig(
            min_ground_stations=2,
            consistency_threshold=0.8,
            max_version_diff=2
        )
        self.aggregator = GlobalAggregator(config)
        
        # 添加测试地面站
        self.aggregator.add_ground_station("station1", 1.0)
        self.aggregator.add_ground_station("station2", 1.0)
        
    def test_station_management(self):
        """测试地面站管理"""
        self.aggregator.add_ground_station("station3", 2.0)
        self.assertEqual(self.aggregator.ground_stations["station3"], 2.0)
        
        self.aggregator.remove_ground_station("station3")
        self.assertNotIn("station3", self.aggregator.ground_stations)
        
    def test_version_control(self):
        """测试版本控制"""
        round_number = 1
        model_update = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.ones(10)
        }
        metrics = {"accuracy": 0.95}
        
        # 关闭验证要求以简化测试
        self.aggregator.config.validation_required = False
        
        # 提交地面站更新
        for station_id in ["station1", "station2"]:
            success = self.aggregator.receive_station_update(
                station_id, round_number, model_update,
                metrics, base_version=0
            )
            self.assertTrue(success)
            
            # 提交验证结果并检查返回值
            validation_success = self.aggregator.submit_validation_result(
                station_id, round_number, metrics
            )
            self.assertTrue(validation_success)
            
        # 手动触发聚合（如果没有自动触发）
        if self.aggregator.current_version == 0:
            self.aggregator._aggregate_round(round_number)
            
        # 验证新版本已创建
        self.assertEqual(self.aggregator.current_version, 1)
        self.assertEqual(len(self.aggregator.model_versions), 1)
        
        # 验证版本内容
        latest_version = self.aggregator.model_versions[-1]
        self.assertEqual(latest_version.version, 1)
        self.assertTrue(torch.all(latest_version.parameters["layer1.weight"] == torch.ones(10, 10)))
        
    def test_consistency_check(self):
        """测试一致性检查"""
        round_number = 1
        model_update = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.ones(10)
        }

        # 确保验证是必需的
        self.aggregator.config.validation_required = True
        # 设置较严格的一致性阈值
        self.aggregator.config.consistency_threshold = 0.9

        # 提交不一致的指标
        metrics1 = {"accuracy": 0.95}
        metrics2 = {"accuracy": 0.75}  # 差异较大

        # 第一个站点更新
        self.assertTrue(
            self.aggregator.receive_station_update(
                "station1", round_number, model_update,
                metrics1, base_version=0
            )
        )
        self.assertTrue(
            self.aggregator.submit_validation_result(
                "station1", round_number, metrics1
            )
        )

        # 第二个站点更新（不一致的指标）
        self.assertTrue(
            self.aggregator.receive_station_update(
                "station2", round_number, model_update,
                metrics2, base_version=0
            )
        )
        self.assertTrue(
            self.aggregator.submit_validation_result(
                "station2", round_number, metrics2
            )
        )

        # 手动检查一致性并尝试聚合
        self.assertFalse(self.aggregator._check_consistency(round_number))
        self.aggregator._aggregate_round(round_number)

        # 由于一致性检查失败，版本号应该保持不变
        self.assertEqual(self.aggregator.current_version, 0)
        self.assertEqual(len(self.aggregator.model_versions), 0)
        
    def test_timeout_handling(self):
        """测试超时处理"""
        round_number = 1
        model_update = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.ones(10)
        }
        metrics = {"accuracy": 0.95}
        
        # 设置较短的超时时间
        self.aggregator.config.aggregation_timeout = 0.1
        
        # 提交一个地面站的更新
        self.aggregator.receive_station_update(
            "station1", round_number, model_update,
            metrics, base_version=0
        )
        
        # 等待超时
        import time
        time.sleep(0.2)
        
        # 提交第二个地面站的更新
        self.aggregator.receive_station_update(
            "station2", round_number, model_update,
            metrics, base_version=0
        )
        
        # 验证是否正确处理了超时情况
        self.assertEqual(self.aggregator.current_version, 0)

if __name__ == '__main__':
    unittest.main()