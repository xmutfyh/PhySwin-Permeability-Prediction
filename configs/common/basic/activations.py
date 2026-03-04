import torch.nn as nn
import torch
import torch.nn.functional as F

# 适用于回归任务的激活函数定义

def ReLU(inplace=True):
    return nn.ReLU(inplace=inplace)

def ReLU6(inplace=True):
    return nn.ReLU6(inplace=inplace)

def LeakyReLU(inplace=True):
    return nn.LeakyReLU(inplace=inplace)

def Tanh():
    return nn.Tanh()

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class HSigmoid(nn.Module):
    """Hard Sigmoid - Adjusted for regression tasks by expanding output range."""
    def __init__(self, bias=3.0, divisor=6.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)

class HSwish(nn.Module):
    """Hard Swish - Kept for non-linearity but adjusted for regression with inplace option."""
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0, 6, inplace=self.inplace) / 6

class Swish(nn.Module):
    """Swish - Suitable for regression tasks due to its smooth and unbounded output."""
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)