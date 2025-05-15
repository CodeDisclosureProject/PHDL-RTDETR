import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


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