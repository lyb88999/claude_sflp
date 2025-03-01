import torch

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
        x = self.fc3(x)
        # 如果使用nll_loss，添加log_softmax
        # return F.log_softmax(x, dim=1)
        # 否则直接返回logits用于cross_entropy
        return x