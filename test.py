import torch

# 创建一个2x3的张量
signal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
signal2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 计算整个张量的均值
mean_all = torch.mean(signal)
print(mean_all)  # 输出：tensor(3.5)

# 沿着第0维计算均值，并保持维度
mean_dim0 = torch.mean(signal, dim=0, keepdim=True)
print(mean_dim0)  # 输出：tensor([[2.5, 3.5, 4.5]])

# 沿着第1维计算均值，不保持维度
mean_dim1 = torch.mean(signal, dim=1, keepdim=True)
print(mean_dim1)  # 输出：tensor([2., 5.])

signal3 = signal2 + signal
print(signal3)
