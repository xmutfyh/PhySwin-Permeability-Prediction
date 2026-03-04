import torch
from torch import nn

class LayerScale(nn.Module):
    """LayerScale layer, modified for regression tasks.
    该层用于调整特征的尺度，适合于回归任务。

    Args:
        dim (int): 输入特征的维度。
        inplace (bool): 是否就地进行操作。默认: ``False``
        data_format (str): 输入数据的格式，可以是 'channels_last'
            或 'channels_first'，分别表示 (B, C, H, W) 和
            (B, N, C) 格式的数据。
    """

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' 只能是 'channels_last' 或 'channels_first'。"
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * 1e-5)  # 初始值为一个很小的数

    def forward(self, x):
        # 根据数据格式调整输出
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight