import torch
import torch.nn as nn
from ..common.base_module import BaseModule
from configs.losses.mse_loss import MSELoss
from .reg_head import RegHead
from ..losses import HuberLoss
import torch.nn.functional as F

class LinearRegHead3(RegHead):
    """线性回归头部。

    用于回归任务，如多孔介质渗透率预测。

    Args:
        in_channels (int): 输入特征的通道数。
        out_channels (int): 输出特征的通道数（通常为 1）。
        loss (dict): 损失函数的配置。
        init_cfg (dict | None): 初始化配置字典。默认使用 None。
    """
    def __init__(self, in_channels, hidden_dim,out_channels=1,init_cfg=dict(type='Normal', layer='Linear', std=0.01),*args,
                 **kwargs):
        super(LinearRegHead3, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        # # Two fully connected layers
        # self.fc1 = nn.Linear(self.in_channels, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, 1)  # Output 1 value for regression
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # Add an additional hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_channels)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。

        Returns:
            Tensor: 回归预测结果，形状为 `(num_samples, out_channels)`。
        """
        # print(f"Input to LinearRegHead: {x.size()}")  # Add this line for debugging
        # print(f"Input to LinearRegHead: {x.size()}")
        # x = x.flatten(1)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        # x = F.relu(self.fc1(x))  #通过 self.fc1 和 ReLU 激活函数得到隐藏层的输出。最后，将隐藏层的输出输入到 self.fc2，产生回归预测结果 pred，并计算损失。
        # # print(f"After fc1: {x.size()}")
        # x = self.fc2(x)
        # print(f"After fc2: {x.size()}")
        return x

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        """训练过程的前向传播。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。
            gt_label (Tensor): 真实的回归目标，形状为 `(num_samples, out_channels)`。

        Returns:
            dict: 包含回归损失的字典。
        """
        x = self.pre_logits(x)
        # print(f"Input to forward_train: {x.size()}")  # 调试输入
        # x = x.view(x.size(0), -1)
        pred = self.forward(x)
        # losses = self.loss(pred, gt_label)
        losses = self.loss(pred, gt_label, **kwargs)
        # return pred, losses
        return losses

    def simple_test(self, x, post_process=False):
        """简单的测试过程，没有数据增强。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。
            post_process (bool): 是否对推理结果进行后处理。

        Returns:
            Tensor | list: 推理结果。无后处理时为张量，后处理时为列表。
        """
        x = self.pre_logits(x)
        # x = x.view(x.size(0), -1)
        pred = self.forward(x)

        if post_process:
            return pred.tolist()  # 转换为列表
        else:
            return pred

    def post_process(self, pred):
        return pred  # No post-processing for regression needed
