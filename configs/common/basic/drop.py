import torch
import torch.nn as nn

class DropPath(nn.Module):
    """随机路径丢弃（Stochastic Depth），用于残差块中的路径丢弃。
    这个类是 drop_path 函数的封装。我们遵循了以下实现：
    参数:
        drop_prob (float): 路径被置零的概率。默认值为 0.05。
            较低的 drop_prob 更适合回归任务，因为它有助于保持输出的连续性。
    """
    def __init__(self, drop_prob=0.05):  # 设置较低的默认 drop_prob
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob=0., training=False):
    """随机路径丢弃（Stochastic Depth），用于残差块中的路径丢弃。

    参数:
        x (Tensor): 输入张量。
        drop_prob (float): 路径被丢弃的概率。
        training (bool): 模型是否处于训练模式。

    返回:
        Tensor: 丢弃路径后的输出张量。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class Dropout(nn.Dropout):
    """torch.nn.Dropout 的封装。我们将 torch.nn.Dropout 中的 `p` 重命名为 `drop_prob`，
    以便与 DropPath 保持一致。

    参数:
        drop_prob (float): 元素被置零的概率。默认值为 0.1。
            较低的 drop_prob 更适合回归任务，因为它有助于保持输出的连续性。
        inplace (bool): 是否在原地执行操作。默认值为 False。
    """

    def __init__(self, drop_prob=0.1, inplace=False):  # 设置较低的默认 drop_prob
        super().__init__(p=drop_prob, inplace=inplace)