import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np


# Hybrid Attention模块
class HybridAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        """
        混合注意力模块，融合通道注意力和空间注意力
        :param dim: 输入特征图的通道数
        :param reduction: 通道压缩系数
        """
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )
        # 通道混洗
        self.shuffle = nn.Conv2d(dim, dim, kernel_size=1, groups=dim // 2)
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        shuffled = self.shuffle(x)
        ca = self.channel_attention(shuffled)
        channel_out = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        spatial_out = channel_out * sa

        return spatial_out


# 改进的 gConvBlock 模块
class gConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, gate_act=nn.Sigmoid, net_depth=8):
        """
        gConvBlock模块，集成Hybrid Attention
        :param dim: 输入特征图的通道数
        :param kernel_size: 卷积核大小
        :param gate_act: 门控激活函数
        :param net_depth: 网络深度，用于初始化
        """
        super().__init__()
        self.dim = dim
        self.net_depth = net_depth
        self.kernel_size = kernel_size

        # 替换原始卷积部分为Hybrid Attention
        self.Wv = nn.Sequential(
            HybridAttention(dim),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, padding_mode='reflect')
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1 / 4)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            nn.init.trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        """
        前向传播
        :param X: 输入特征图，形状为 [batch_size, channels, height, width]
        :return: 加权后的特征图
        """
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out



class HAGConv(RepC3):
    def __init__(self, c1, c2, n=3, e=1):
        """
        HAGConv模块
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param n: gConvBlock 堆叠数量
        :param e: 扩展系数
        """
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[gConvBlock(c_) for _ in range(n)])


# C2f_gConv 模块保持不变
class C2f_gConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        C2f_gConv模块
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param n: gConvBlock 堆叠数量
        :param shortcut: 是否使用残差连接
        :param g: 分组数
        :param e: 扩展系数
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(gConvBlock(self.c) for _ in range(n))


# 稀疏性约束函数
def sparse_loss(model, lambda_sparse=1e-5):
    """
    对模型的所有可训练权重施加 L1 稀疏性正则化约束
    :param model: 待训练的模型
    :param lambda_sparse: 稀疏性正则化系数
    :return: 稀疏性正则化损失
    """
    l1_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l1_loss += torch.sum(torch.abs(param))
    return lambda_sparse * l1_loss
