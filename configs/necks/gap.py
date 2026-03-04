import torch
import torch.nn as nn

class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], f'GlobalAveragePooling dim only support {1, 2, 3}, get {dim} instead.'
        self.dim = dim
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        # else:
        #     raise ValueError(f"Unsupported dim {dim} for GlobalAveragePooling")
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            # print(f"Input to GAP: {inputs.shape}")
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
            # print(f"Output from GAP: {outs.shape}")
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
    # def forward(self, inputs):
    #     if isinstance(inputs, tuple):
    #         outs = [self.gap(x) for x in inputs]
    #         outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
    #         outs = [self.dropout(out) for out in outs]  # Apply Dropout to each tensor in the tuple
    #         return tuple(outs)
    #     elif isinstance(inputs, torch.Tensor):
    #         outs = self.gap(inputs)
    #         outs = outs.view(inputs.size(0), -1)
    #         return self.dropout(outs)  # Apply Dropout to the tensor
    #     else:
    #         raise TypeError('neck inputs should be tuple or torch.tensor')

