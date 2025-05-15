import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange
from collections import OrderedDict
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *

from .transformer import LocalWindowAttention

from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible

from timm.layers import trunc_normal_
from timm.layers import DropPath

__all__ = ["DEA_CARAFE","HAGConv","LW_Fusion"]




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

class DetailEnhanceAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(DetailEnhanceAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 增加一个边缘检测卷积，用于捕捉高频细节
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        # 通道注意力
        avg_out = self.conv1(self.global_avg_pool(x))
        avg_out = self.sigmoid(self.conv2(avg_out))

        # 边缘细节增强
        edge_out = self.edge_conv(x)
        enhanced = x * avg_out + edge_out  # 细节增强与通道加权相结合
        return enhanced


class DEA_CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(DEA_CARAFE, self).__init__()
        self.scale = scale

        # 多尺度特征压缩，增加对小目标的感受
        self.comp_3x3 = nn.Conv2d(c, c_mid, kernel_size=3, padding=1)
        self.comp_5x5 = nn.Conv2d(c, c_mid, kernel_size=5, padding=2)

        # 引入细节增强注意力机制
        self.detail_attention = DetailEnhanceAttention(c_mid)

        # 编码特征生成重构权重
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=k_enc, padding=k_enc // 2, bias=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        # 多尺度特征融合
        W3 = self.comp_3x3(X)  # 3x3卷积特征
        W5 = self.comp_5x5(X)  # 5x5卷积特征
        W = W3 + W5  # 多尺度特征融合

        # 细节增强，突出小目标
        W = self.detail_attention(W)

        # 编码特征生成重构权重
        W = self.enc(W)  # b * (scale * k_up)^2 * h * w
        W = self.pix_shf(W)  # b * k_up^2 * h_ * w_
        W = F.softmax(W, dim=1)  # b * k_up^2 * h_ * w_

        # 上采样和特征融合
        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * k_up^2 * c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * k_up^2 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
# 改进后的空间注意力模块
class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.weight = nn.Parameter(torch.ones(1), requires_grad=True)  # 添加可学习权重参数

    def forward(self, x, y):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        y_avg = torch.mean(y, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)

        # 将两个模态特征融合
        combined_avg = x_avg * self.weight + y_avg * (1 - self.weight)
        combined_max = x_max * self.weight + y_max * (1 - self.weight)

        x2 = torch.concat([combined_avg, combined_max], dim=1)
        sattn = self.sa(x2)
        return sattn


# 改进后的通道注意力模块
class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
        self.weight = nn.Parameter(torch.ones(1), requires_grad=True)  # 添加可学习权重参数

    def forward(self, x, y):
        x_gap = self.gap(x)
        y_gap = self.gap(y)

        # 多模态融合
        combined_gap = x_gap * self.weight + y_gap * (1 - self.weight)

        cattn = self.ca(combined_gap)
        return cattn


# 改进后的像素注意力模块
class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=True)  # 添加可学习权重参数

    def forward(self, x, y, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)

        # 模态间融合
        combined = torch.cat([x * self.weight + y * (1 - self.weight), pattn1], dim=2)
        x2 = rearrange(combined, 'b c t h w -> b (c t) h w')

        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


# 多模态融合模块
class MultiModalFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(MultiModalFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        combined = x + y
        out = self.fc1(combined.mean([2, 3]))  # 全局平均池化
        out = self.sigmoid(self.fc2(out)).unsqueeze(-1).unsqueeze(-1)
        return x * out + y * (1 - out)  # 加权融合



class LW_Fusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(LW_Fusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.mm_fusion = MultiModalFusion(dim, reduction)  # 引入多模态融合层
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = self.mm_fusion(x, y)  # 使用多模态融合替代初始简单相加
        cattn = self.ca(x, y)  # 将两个模态输入到改进的通道注意力
        sattn = self.sa(x, y)  # 将两个模态输入到改进的空间注意力
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(x, y, pattn1))  # 将融合后的特征用于像素注意力

        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


## Convolution and Attention Fusion Module  (CAFM)
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


class CAFMFusion(nn.Module):
    def __init__(self, dim, heads):
        super(CAFMFusion, self).__init__()
        self.cfam = CAFM(dim, num_heads=heads)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        pattn1 = self.cfam(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

