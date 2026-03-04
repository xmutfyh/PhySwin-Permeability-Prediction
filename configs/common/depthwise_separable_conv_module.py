import torch.nn as nn
from .conv_module import ConvModule

class DepthwiseSeparableConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=None,  # 默认不使用激活函数
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # 如果没有指定深度卷积和逐点卷积的归一化/激活配置，使用默认配置。
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # 深度卷积层
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs)

        # 逐点卷积层
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

        # 最后一层的激活函数, 默认为 None
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.final_activation = nn.ReLU()
            elif act_cfg['type'] == 'Sigmoid':
                self.final_activation = nn.Sigmoid()
            elif act_cfg['type'] == 'Tanh':
                self.final_activation = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation function: {act_cfg['type']}")
        else:
            self.final_activation = None

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x
