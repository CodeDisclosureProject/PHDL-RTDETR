import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


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
