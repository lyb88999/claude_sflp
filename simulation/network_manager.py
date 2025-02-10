class NetworkManager:
    def __init__(self, network_model, topology_manager):
        """
        初始化网络管理器
        Args:
            network_model: 卫星网络模型
            topology_manager: 拓扑管理器
        """
        self.network_model = network_model
        self.topology_manager = topology_manager
        self.priority_tasks = {}  # 优先级任务队列
        self._connection_status = {}  # 连接状态
        
    def is_connected(self, satellite_id: str = None) -> bool:
        """检查网络连接状态"""
        if satellite_id not in self._connection_status:
            # 根据拓扑状态初始化连接状态
            self._connection_status[satellite_id] = True
            
        return self._connection_status[satellite_id]
        
    def has_priority_task(self) -> bool:
        """检查是否有高优先级任务"""
        return False  # 简化处理，默认没有高优先级任务
        
    def add_priority_task(self, task_id: str, priority: int):
        """添加优先级任务"""
        self.priority_tasks[task_id] = priority
        
    def remove_priority_task(self, task_id: str):
        """移除优先级任务"""
        self.priority_tasks.pop(task_id, None)
        
    def get_connection_quality(self, source: str, target: str) -> float:
        """获取连接质量"""
        # 可以从拓扑管理器获取链路质量
        return 1.0  # 默认返回最好的质量