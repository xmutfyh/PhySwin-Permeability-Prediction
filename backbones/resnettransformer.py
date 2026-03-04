import torch
import torch.nn as nn

from configs.backbones import ResNet
from configs.common import BaseModule
import torch.nn.functional as F

class TransformerEncoderLayers(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayers, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        # Normalization layer before attention
        self.norm1 = self.build_norm_layer(norm_cfg, embed_dims, postfix=1)

        # Multihead Attention with batch_first=True
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            bias=qkv_bias,
            batch_first=True  # Important change
        )

        # Normalization layer before feedforward network
        self.norm2 = self.build_norm_layer(norm_cfg, embed_dims, postfix=2)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(feedforward_channels, embed_dims)
        )

        # Dropout for stochastic depth
        self.drop_path = nn.Dropout(drop_path_rate)

    def forward(self, x):
        norm_x = self.norm1(x)

        # Apply Multihead Attention (no need to permute)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_output

        # Apply Feedforward Network
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.drop_path(ffn_output)
        return x



class ResNetTransformer(BaseModule):
    """Combines ResNet and Transformer into a single model.

    Args:
        resnet_depth (int): Depth of the ResNet backbone (e.g., 50, 101).
        embed_dims (int): The feature dimension for Transformer.
        num_heads (int): Number of attention heads in Transformer.
        feedforward_channels (int): Hidden dimension for FFNs in Transformer.
        num_transformer_layers (int): Number of Transformer encoder layers.
        drop_rate (float): Dropout rate for Transformer feedforward networks.
        attn_drop_rate (float): Dropout rate for Transformer attention weights.
        drop_path_rate (float): Dropout rate for Transformer stochastic depth.
        qkv_bias (bool): Whether to include bias in Transformer QKV layers.
        norm_cfg (dict): Config for normalization layer in Transformer.
    """

    def __init__(self,
                 resnet_depth,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 num_transformer_layers,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN')):
        super(ResNetTransformer, self).__init__()

        # Initialize ResNet backbone
        self.resnet = ResNet(depth=resnet_depth)
        print(dir(self.resnet))

        self.feat_dim = self.resnet.feat_dim
        # 定义多层全连接层
        hidden_dim1 = 1024
        hidden_dim2 = 512
        self.fc1 = nn.Linear(self.feat_dim*7*7, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, embed_dims)

        # Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayers(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg
            ) for _ in range(num_transformer_layers)
        ])

        # Final linear layer for classification or regression
        # self.fc = nn.Linear(embed_dims, 1)  # Adjust output features as needed

    def forward(self, x):
        # Step 1: Extract features using ResNet
        features = self.resnet(x)
        features = features[0]  # 提取特征图
        features = features.view(features.size(0), -1)
        # 通过全连接层
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.fc3(features)

        # Step 5: Pass through Transformer layers
        for transformer_layer in self.transformer_layers:
            features = transformer_layer(features)

        # output = self.fc(features)

        return features

# # Example usage
# if __name__ == "__main__":
#     # Define the model
#     model = ResNetTransformer(
#         resnet_depth=50,
#         embed_dims=256,
#         num_heads=8,
#         feedforward_channels=512,
#         num_transformer_layers=6
#     )
#
#     # Print model summary
#     print(model)
#
#     # Example input
#     x = torch.randn(1, 3, 224, 224)  # Batch of images with 3 channels and 224x224 size
#     output = model(x)
#     print(output.shape)