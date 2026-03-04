import torch.nn as nn
from ..common.base_module import BaseModule
from configs.losses.mse_loss import MSELoss
from .reg_head import RegHead
from ..losses import HuberLoss


class CNNRegHead(RegHead):
    """线性回归头部。

    用于回归任务，如多孔介质渗透率预测。

    Args:
        in_channels (int): 输入特征的通道数。
        out_channels (int): 输出特征的通道数（通常为 1）。
        loss (dict): 损失函数的配置。
        init_cfg (dict | None): 初始化配置字典。默认使用 None。
    """
    def __init__(self, in_channels, out_channels=1, loss=dict(type='HuberLoss', loss_weight=1.0), init_cfg=None):
        super(CNNRegHead, self).__init__(loss=loss, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 线性层，用于回归预测
        self.fc1 = nn.Linear(self.in_channels, 64)  # 第一层全连接层
        self.fc2 = nn.Linear(64, self.out_channels)  # 第二层全连接层
        self.loss = HuberLoss(**loss)

    def forward(self, x):
        """前向传播。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。

        Returns:
            Tensor: 回归预测结果，形状为 `(num_samples, out_channels)`。
        """
        # print(f"Input to LinearRegHead: {x.size()}")  # Add this line for debugging
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))  # 使用ReLU作为激活函数
        return self.fc2(x)

    def forward_train(self, x, gt_label, **kwargs):
        """训练过程的前向传播。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。
            gt_label (Tensor): 真实的回归目标，形状为 `(num_samples, out_channels)`。

        Returns:
            dict: 包含回归损失的字典。
        """
        pred = self.forward(x)
        losses = self.loss(pred, gt_label)
        return pred, losses

    def simple_test(self, x, post_process=False):
        """简单的测试过程，没有数据增强。

        Args:
            x (Tensor): 输入特征，形状为 `(num_samples, in_channels)`。
            post_process (bool): 是否对推理结果进行后处理。

        Returns:
            Tensor | list: 推理结果。无后处理时为张量，后处理时为列表。
        """
        pred = self.forward(x)

        if post_process:
            return pred.tolist()  # 转换为列表
        else:
            return pred
