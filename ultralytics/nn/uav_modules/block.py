import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from einops import rearrange
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from collections import OrderedDict
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, WTConvNormLayer,BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
import math
from einops import rearrange
import pywt
__all__ = ['DySample','SPDConv','FrequencyFocusedDownSampling','SemanticAlignmenCalibration','MFFF','Add','AdaptiveWaveletFeatureExtractor','XSSBlock']

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4):
        super(DySample, self).__init__()
        # 新增：强制将 groups 转为整数（核心修复）
        if isinstance(groups, str):
            try:
                groups = int(groups)  # 字符串转整数（如 "8" → 8）
            except ValueError:
                raise ValueError(f"groups 必须是整数或可转为整数的字符串，当前值：{groups}")
        assert in_channels % groups == 0, "in_channels 必须能被 groups 整除"
        self.scale = scale
        self.groups = groups
        self.in_channels = in_channels
        self.cg = in_channels // groups

        # 预测 offset：每个 group 预测 2*(scale^2) 个通道（用于 pixel_shuffle -> (dx,dy)）
        self.offset = nn.Conv2d(in_channels, 2 * groups * scale ** 2, 1)
        self.normal_init(self.offset, std=0.001)

        # 自注意力机制
        self.attention = SelfAttention(in_channels)

        # 融合权重系数
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Sobel 核注册为 buffer（会跟随 device）
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def compute_confidence(self, x):
        """
        多通道 Sobel：depthwise conv -> (B,C,H,W)，再聚合成 (B,1,H,W)
        """
        B, C, H, W = x.shape
        wx = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)  # (C,1,3,3)
        wy = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)

        grad_x = F.conv2d(x, wx, padding=1, groups=C)
        grad_y = F.conv2d(x, wy, padding=1, groups=C)
        mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)  # (B,C,H,W)

        conf = mag.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # 简单归一化到 0~1（每个样本独立）
        mn = conf.amin(dim=(2, 3), keepdim=True)
        mx = conf.amax(dim=(2, 3), keepdim=True)
        conf = (conf - mn) / (mx - mn + 1e-6)
        return conf

    def compute_attention(self, x):
        """
        注意力输出仍是 (B,C,H,W)，后面会压成 (B,1,H,W) 用作权重图
        """
        return self.attention(x)

    def sample(self, x, offset, confidence_map, attention_map):
        """
        正确的 deform 上采样采样流程：
        - offset: (B, 2*groups*s^2, H, W)
        - pixel_shuffle -> (B*groups, 2, H*s, W*s)
        - 构造 base_grid: (B*groups, H*s, W*s, 2)
        - grid_sample 输入: (B*groups, Cg, H, W)
        - 输出 reshape 回 (B, C, H*s, W*s)
        """
        B, C, H, W = x.shape
        s = self.scale
        G = self.groups
        Cg = self.cg

        # attention_map (B,C,H,W) -> (B,1,H,W)
        att_w = attention_map.mean(dim=1, keepdim=True)

        # 融合权重图 (B,1,H,W)，再上采样到 HR
        w = torch.sigmoid(self.alpha * confidence_map + (1 - self.alpha) * att_w)  # (B,1,H,W)
        w_hr = F.interpolate(w, scale_factor=s, mode="bilinear", align_corners=False)  # (B,1,H*s,W*s)

        # offset reshape -> (B,G,2*s^2,H,W) -> (B*G,2*s^2,H,W) -> pixel_shuffle -> (B*G,2,H*s,W*s)
        off = offset.view(B, G, 2 * s * s, H, W).reshape(B * G, 2 * s * s, H, W)
        off_hr = F.pixel_shuffle(off, s)  # (B*G,2,H*s,W*s)

        # base grid：HR 像素中心映射回 LR
        ys = torch.arange(H * s, device=x.device, dtype=x.dtype)
        xs = torch.arange(W * s, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H*s, W*s)

        # HR -> LR（像素坐标）
        yy = (yy + 0.5) / s - 0.5
        xx = (xx + 0.5) / s - 0.5

        # LR 像素坐标 -> normalized [-1,1] (align_corners=False)
        x_norm = (xx + 0.5) / W * 2 - 1
        y_norm = (yy + 0.5) / H * 2 - 1
        base_grid = torch.stack([x_norm, y_norm], dim=-1)  # (H*s, W*s, 2)
        base_grid = base_grid.unsqueeze(0).repeat(B * G, 1, 1, 1)  # (B*G,H*s,W*s,2)

        # 把 w_hr 扩展到 (B*G,1,H*s,W*s)，调制 offset 强度
        w_hr_rep = w_hr.repeat_interleave(G, dim=0)  # (B*G,1,H*s,W*s)

        dx = off_hr[:, 0:1] * w_hr_rep
        dy = off_hr[:, 1:2] * w_hr_rep

        # 像素偏移 -> normalized 偏移
        dx_norm = dx.permute(0, 2, 3, 1) * (2.0 / W)
        dy_norm = dy.permute(0, 2, 3, 1) * (2.0 / H)

        grid = base_grid.clone()
        grid[..., 0:1] = grid[..., 0:1] + dx_norm
        grid[..., 1:2] = grid[..., 1:2] + dy_norm

        # 分组输入：x -> (B*G, Cg, H, W)
        xg = x.view(B, G, Cg, H, W).reshape(B * G, Cg, H, W)

        yg = F.grid_sample(xg, grid, mode="bilinear", align_corners=False, padding_mode="border")
        y = yg.view(B, G, Cg, H * s, W * s).reshape(B, C, H * s, W * s)
        return y

    def forward(self, x):
        confidence_map = self.compute_confidence(x)
        attention_map = self.compute_attention(x)
        offset = self.offset(x)
        return self.sample(x, offset, confidence_map, attention_map)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        mid = max(1, in_channels // 8)  # 防止为 0
        self.query_conv = nn.Conv2d(in_channels, mid, 1)
        self.key_conv = nn.Conv2d(in_channels, mid, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        HW = H * W
        query = self.query_conv(x).view(B, -1, HW).permute(0, 2, 1)  # (B,HW,mid)
        key = self.key_conv(x).view(B, -1, HW)                      # (B,mid,HW)
        value = self.value_conv(x).view(B, -1, HW)                  # (B,C,HW)

        attention = torch.bmm(query, key)                           # (B,HW,HW)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))          # (B,C,HW)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# class DySample(nn.Module):
#     def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
#         super().__init__()
#         self.scale = scale
#         self.style = style
#         self.groups = groups
#         assert style in ['lp', 'pl']
#         if style == 'pl':
#             assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
#         assert in_channels >= groups and in_channels % groups == 0
#
#         if style == 'pl':
#             in_channels = in_channels // scale ** 2
#             out_channels = 2 * groups
#         else:
#             out_channels = 2 * groups * scale ** 2
#
#         self.offset = nn.Conv2d(in_channels, out_channels, 1)
#         self.normal_init(self.offset, std=0.001)
#         if dyscope:
#             self.scope = nn.Conv2d(in_channels, out_channels, 1)
#             self.constant_init(self.scope, val=0.)
#
#         self.register_buffer('init_pos', self._init_pos())
#
#     def normal_init(self, module, mean=0, std=1, bias=0):
#         if hasattr(module, 'weight') and module.weight is not None:
#             nn.init.normal_(module.weight, mean, std)
#         if hasattr(module, 'bias') and module.bias is not None:
#             nn.init.constant_(module.bias, bias)
#
#     def constant_init(self, module, val, bias=0):
#         if hasattr(module, 'weight') and module.weight is not None:
#             nn.init.constant_(module.weight, val)
#         if hasattr(module, 'bias') and module.bias is not None:
#             nn.init.constant_(module.bias, bias)
#
#     def _init_pos(self):
#         h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
#         return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
#
#     def sample(self, x, offset):
#         B, _, H, W = offset.shape
#         offset = offset.view(B, 2, -1, H, W)
#         coords_h = torch.arange(H) + 0.5
#         coords_w = torch.arange(W) + 0.5
#         coords = torch.stack(torch.meshgrid([coords_w, coords_h])
#                              ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
#         normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
#         coords = 2 * (coords + offset) / normalizer - 1
#         coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
#             B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
#         return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
#                              align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)
#
#     def forward_lp(self, x):
#         if hasattr(self, 'scope'):
#             offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
#         else:
#             offset = self.offset(x) * 0.25 + self.init_pos
#         return self.sample(x, offset)
#
#     def forward_pl(self, x):
#         x_ = F.pixel_shuffle(x, self.scale)
#         if hasattr(self, 'scope'):
#             offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
#         else:
#             offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
#         return self.sample(x, offset)
#
#     def forward(self, x):
#         if self.style == 'pl':
#             return self.forward_pl(x)
#         return self.forward_lp(x)

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x



class FFM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


# class ImprovedFFTKernel(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#
#         ker = 31
#         pad = ker // 2
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
#             nn.GELU()
#         )
#         self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
#         self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
#         self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
#
#         self.act = nn.SiLU()
#
#         # 改进后的 SCA 部分
#         self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
#         self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)
#
#         # self.pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.ffm = FFM(dim)
#
#         #通道注意力
#         self.channel_attention = nn.Sequential(
#             nn.Conv2d(dim, dim // 4, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(dim // 4, dim, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         #1*1 进行通道融合
#         out = self.in_conv(x)
#         #公式1开始
#         # 池化后1*1卷积
#         x_att = self.fac_conv(self.fac_pool(out)) # 平均池化加卷积
#         x_fft = torch.fft.fft2(out, norm='backward') #FF
#         x_fft = x_att * x_fft # 元素级相乘
#         x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')  # FF
#         x_fca = torch.abs(x_fca)
#         #公式1结束
#
#         #公式2
#         x_sca1 = self.conv1x1(x_fca)
#         x_sca2 = self.conv3x3(x_fca)
#         x_sca3 = self.conv5x5(x_fca)
#         x_sca = x_sca1 + x_sca2 + x_sca3
#         #公式2结束
#
#         # 使用通道注意力机制
#         channel_weights = self.channel_attention(x_att)
#         x_sca = x_sca * channel_weights
#
#         #FF的公式
#         x_sca = self.ffm(x_sca)
#
#         # 最终融合 公式4
#         out = x + self.dw_33(out) + self.dw_11(out) + x_sca
#         out = self.act(out)
#         return self.out_conv(out)

# class MFFF(nn.Module):
#     def __init__(self, dim, e=0.25):
#         super().__init__()
#         self.e = e
#         self.cv1 = Conv(dim, dim, 1)
#         self.cv2 = Conv(dim, dim, 1)
#         self.m = ImprovedFFTKernel(int(dim * self.e))
#
#     def forward(self, x):
#         c1 = round(x.size(1) * self.e)
#         c2 = x.size(1) - c1
#         ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
#         return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

class ADown(nn.Module): # Downsample x2分支
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling(nn.Module):  # Downsample x2分支 with parallel FGM
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)  # FGM 模块处理 x2 分支

        # 1x1 卷积用于在拼接后减少通道数
        self.conv_reduce = Conv(self.c*2, self.c, 1, 1)


        # 新增的卷积层用于调整 fgm_out 的空间尺寸
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
#经过池化后分成两个分支，一个分支经过 cv1 处理，另一个分支经过 fgm + maxpool cv2 处理，然后将两个分支拼接在一起，最后使用 1x1 卷积将通道数减少到预期的值。公式写一个表达一下，x1,x2用文字描述一下是什么，cv1,cv2也是呀
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True) # torch.Size([2, 256, 79, 79])
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支

        fgm_out = self.ffm(x2)  # FGM 处理的输出  torch.Size([2, 128, 79, 79])
        fgm_out = self.conv_resize(fgm_out)  # 调整 fgm_out 的空间尺寸  torch.Size([2, 128, 40, 40])
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out) # torch.Size([2, 128, 40, 40])

        # 将 FGM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fgm_out, pooled_out), 1)
        
        # 使用 1x1 卷积将通道数减少到预期的值
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)

class SemanticAlignmenCalibration(nn.Module):  #
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)  # 用于处理高分辨率的空间特征
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)  # 用于处理低分辨率的语义特征

        # FGM模块：用于在频域中增强特征
        self.frequency_enhancer = FFM(hidden_channels)
        # 门控卷积：结合空间和频域特征
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        # 用于生成偏移量的卷积序列
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),  # 处理拼接后的特征
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  # 生成偏移量
        )
        # 这样初始化权重可以有效地提高模型训练的稳定性，防止梯度爆炸或消失的问题，特别是当网络较深时。
        self.init_weights()
        self.offset_conv[1].weight.data.zero_()  # 初始化最后一层卷积的权重为零

    def init_weights(self):
        # 初始化卷积层的权重
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()  # [2,256,80,80]

        # 处理低分辨率的语义特征 (1/8 下采样)[2,256,20,20]
        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear',
                                          align_corners=True)

        # 频域增强特征
        enhanced_frequency = self.frequency_enhancer(semantic_features)

        # 门控机制融合频域和空间域的特征
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        # 处理高分辨率的空间特征 (1/8 下采样)
        coarse_features = self.spatial_conv(coarse_features)

        # 拼接处理后的空间特征和融合后的特征
        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        # 调整特征维度以适应分组
        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 获取偏移量
        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1,
                                                                                     out_h, out_w)

        # 生成归一化网格用于偏移校正
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(
            fused_features.device)

        # 使用生成的偏移量对网格进行调整
        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors

        # 进行特征采样
        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)

        # 调整维度回到原始形状
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)

        # 融合增强后的特征
        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2,
                                                                                              :, :]

        return final_features


#--------------------------------------------------------------------------------------------------------------------------------------------
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            kernel_size: int,  # 卷积核大小
            stride: int = 1,  # 步长，默认为1
            padding: int = 0,  # 填充，默认为0
            groups: int = 1,  # 组数，默认为1
            norm_cfg: Optional[dict] = None,  # 归一化配置，默认为None
            act_cfg: Optional[dict] = None):  # 激活函数配置，默认为None
        super().__init__()
        layers = []
        # 卷积层：bias是否启用由norm_cfg是否为None决定（有归一化则无需bias）
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=(norm_cfg is None)
        ))
        # 归一化层（按需添加）
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 激活层（按需添加）
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # 组合所有层为序列化模块
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播：直接调用序列化模块
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        # 构建归一化层，目前支持BN（BatchNorm2d）
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(
                num_features=num_features,
                momentum=norm_cfg.get('momentum', 0.1),
                eps=norm_cfg.get('eps', 1e-5)
            )
        # 未实现的归一化类型抛出异常
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        # 构建激活层，目前支持ReLU和SiLU
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # 未实现的激活函数类型抛出异常
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")


class CAA(nn.Module):
    """Context Anchor Attention（上下文锚点注意力模块）"""

    def __init__(
            self,
            channels: int,  # 通道数
            h_kernel_size: int = 11,  # 水平卷积核大小，默认为7
            v_kernel_size: int = 11,  # 垂直卷积核大小，默认为7
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
            act_cfg: Optional[dict] = dict(type='SiLU')):  # 激活函数配置
        super().__init__()


        # 1x1卷积模块（通道保持不变，带归一化和激活）
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 水平卷积模块（1xh_kernel_size，分组卷积，无归一化和激活）
        self.h_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, h_kernel_size),
            stride=1,
            padding=(0, h_kernel_size // 2),
            groups=channels,
            norm_cfg=None,
            act_cfg=None
        )

        # 垂直卷积模块（v_kernel_sizex1，分组卷积，无归一化和激活）
        self.v_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(v_kernel_size, 1),
            stride=1,
            padding=(v_kernel_size // 2, 0),
            groups=channels,
            norm_cfg=None,
            act_cfg=None
        )

        # 1x1卷积模块（通道保持不变，带归一化和激活）
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # Sigmoid激活函数（生成注意力权重因子）
        self.act = nn.Sigmoid()

    def forward(self, x):
        # 步骤1：平均池化 + 第一个1x1卷积
        avg = self.conv1(x)

        # 步骤2：水平卷积 + 垂直卷积（提取上下文空间信息）
        h_conv = self.h_conv(avg)
        v_conv = self.v_conv(h_conv)

        # 步骤3：第二个1x1卷积 + Sigmoid激活，生成注意力因子
        attn_factor = self.act(self.conv2(v_conv))

        return attn_factor

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()

class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim,fc_ratio=4, dilation=[3, 5, 7]) -> None:
        super().__init__()
        self.CAA=CAA(dim)
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        # 改进后的 SCA 部分
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        #扩展感受野
        self.conv0_1 = Conv(dim, dim // fc_ratio)
        self.conv0_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-3], g=dim // fc_ratio)
        self.conv0_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-1], g=dim // fc_ratio)
        self.conv2_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)

        # 可学习加权参数
        # self.weight_attn = nn.Parameter(torch.tensor([1.0]))
        # self.weight_result = nn.Parameter(torch.tensor([1.0]))
        # self.gate = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1) ,# 卷积层
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # 1*1 进行通道融合

        out = self.avg_pool(self.in_conv(x))


        #第一个分叉路径
        out_ttf = torch.fft.fft2(out, norm='backward')
        out_ttf = self.conv1x1(torch.abs(out_ttf))

        #第二个分支路径
        out_strip = self.CAA(out)

        #点乘,合并分支
        result = out_strip * out_ttf

        #还原傅里叶变换
        result = torch.fft.ifft2(result, dim=(-2, -1), norm='backward')

        result = torch.abs(result)


        # 结束
        result = self.act(self.dw_conv(result))
        #合并之后
        # 1X1+3X3+5X5
        # x_sca1 = self.conv1x1(x)
        # x_sca2 = self.conv3x3(x)
        # x_sca3 = self.conv5x5(x)
        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)  # 分支1 1X1

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)  # 分支2 1X1

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)  # 分支3 1X1

        attn = attn0_3 + attn1_3 + attn2_3  # 矩阵加法运算

        #attn = self.conv3(attn)

        # # 1. 加权融合（可能产生大值）
        # gate_1 = self.gate(attn)
        # gate_2 = self.gate(result)

        # result_last = gate_1 * attn + gate_2 * result
        result_last = self.dw_conv(attn) + result + x



        # #改良
        # x_fft = torch.fft.fft2(out, norm='backward')
        # x_fft =self.fac_conv(torch.abs(x_fft))
        # x_sca1 = self.conv1x1(out)
        # x_sca2 = self.conv3x3(out)
        # x_sca3 = self.conv5x5(out)
        # x_sca = x_sca1 + x_sca2 + x_sca3
        # x_att = x_sca * x_fft
        # x_fca = torch.fft.ifft2(x_att, dim=(-2, -1), norm='backward')
        # x_fca = torch.abs(x_fca)

        # # 公式1开始
        # # 池化后1*1卷积
        # x_att = self.fac_conv(self.fac_pool(out))  # 平均池化加卷积 分支2
        # x_fft = torch.fft.fft2(out, norm='backward')  # FF 分支1
        # x_fft = x_att * x_fft  # 元素级相乘
        # x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')  # FF
        # x_fca = torch.abs(x_fca)
        # # 公式1结束

        # x_c = self.fac_conv(self.fac_pool(x_fca))
        # channel_weights = self.channel_attention(x_sca)  # 分支3
        # x_sca = x_c * channel_weights


        # 公式2
        # x_sca1 = self.conv1x1(x_fca)
        # x_sca2 = self.conv3x3(x_fca)
        # x_sca3 = self.conv5x5(x_fca)
        # x_sca = x_sca1 + x_sca2 + x_sca3
        # 公式2结束

        # 使用通道注意力机制
        # channel_weights = self.channel_attention(x_att) # 分支3
        # x_sca = x_sca * channel_weights

        # FF的公式
        # x_sca = self.ffm(x_sca)
        #
        # # 最终融合 公式4
        # out = self.dw_33(out) + self.dw_11(out) + x_sca
        # out = self.act(out)
        return self.out_conv(result_last)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class MFFF(nn.Module):
    def __init__(self, dim,e=0.75):
        super().__init__()
        self.e=e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim*e))
    #     # 在模型初始化后应用剪枝
    #     self.apply_pruning()
    #
    # def apply_pruning(self):
    #     # 对每个卷积层应用剪枝
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.Conv2d):  # 判断是否是卷积层
    #             # 演示中剪枝 20% 的权重
    #             prune.random_unstructured(module, name='weight', amount=0.2)  # 剪枝 20%
    #             print(f"剪枝已应用到 {name}")

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch),identity), dim=1))



#-------------
class Add(nn.Module):
    def __init__(self, inc):
        super(Add, self).__init__()
        self.conv=nn.Conv2d(inc[0],inc[1],1)
        self.pool=nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        coarse_features, semantic_features = x
        coarse_features = self.pool(coarse_features)
        coarse_features = self.conv(coarse_features)
        semantic_features = coarse_features + semantic_features
        return semantic_features

#小波变换---------------------------------------------------------------------------------------------------------------------------
class AdaptiveWaveletFeatureExtractor(nn.Module):
    def __init__(self, c1, c2):
        super(AdaptiveWaveletFeatureExtractor, self).__init__()
        self.conv = Conv(c1, c2, 1, 1,act=True)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        #self.dwconv=nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=True,
                                #groups=c2)
        # 核心：1×1卷积实现通道对齐（512→256），完全保尺寸
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),  # 仅改通道，H/W不变
            nn.BatchNorm2d(c2)
        )
        self.lobck = RGBlock(c2)

    def wavelet_transform(self, x):
        """
        GPU版二维 Haar 小波分解
        输入:
            x: [B, C, H, W]
        输出:
            low_freq:  [B, 2C, H/2, W/2] -> 通道维拼接 (LL, LH)
            high_freq: [B, 2C, H/2, W/2] -> 通道维拼接 (HL, HH)

        这样可以保持你原来的 forward 完全不变。
        """
        B, C, H, W = x.shape

        # 若尺寸为奇数，先补齐到偶数，避免 stride=2 时尺寸问题
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='replicate')

        # 懒初始化固定 Haar 核，兼容旧 .pt 权重（不依赖 __init__ 重跑）
        if not hasattr(self, "_haar_kernel_cache") or self._haar_kernel_cache is None:
            self._haar_kernel_cache = torch.tensor([
                [[0.5, 0.5],
                 [0.5, 0.5]],  # LL

                [[-0.5, -0.5],
                 [0.5, 0.5]],  # LH

                [[-0.5, 0.5],
                 [-0.5, 0.5]],  # HL

                [[0.5, -0.5],
                 [-0.5, 0.5]],  # HH
            ], dtype=torch.float32).unsqueeze(1)  # [4,1,2,2]

        base_kernel = self._haar_kernel_cache.to(device=x.device, dtype=x.dtype)  # [4,1,2,2]
        weight = base_kernel.repeat(C, 1, 1, 1)  # [4C,1,2,2]

        # groups=C: 每个输入通道单独做 4 个 Haar 滤波器
        coeffs = F.conv2d(x, weight, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
        h2, w2 = coeffs.shape[-2:]

        coeffs = coeffs.view(B, C, 4, h2, w2)  # [B,C,4,h,w]

        LL = coeffs[:, :, 0, :, :]  # [B,C,h,w]
        LH = coeffs[:, :, 1, :, :]
        HL = coeffs[:, :, 2, :, :]
        HH = coeffs[:, :, 3, :, :]

        # 为了兼容你原来的 forward，直接在通道维拼接成 4D
        low_freq = torch.cat([LL, LH], dim=1)  # [B,2C,h,w]
        high_freq = torch.cat([HL, HH], dim=1)  # [B,2C,h,w]

        return low_freq, high_freq
    # # 小波变换函数：对输入的图像进行小波变换
    # def wavelet_transform(self,x):
    #     """
    #     对输入特征图进行二维离散小波变换，并返回低频（LL）和高频（LH, HL, HH）部分
    #     """
    #     B, C, H, W = x.shape  # 输入形状: (B, C, H, W)
    #     low_freq = []
    #     high_freq = []
    
    #     for b in range(B):
    #         for c in range(C):
    #             # 对每个通道进行二维小波变换，使用'haar'小波
    #             coeffs = pywt.dwt2(x[b, c].detach().cpu().numpy(), 'haar')
    #             LL, (LH, HL, HH) = coeffs  # 低频（LL）和高频（LH, HL, HH）
    
    #             # 将结果转换为PyTorch张量
    #             LL_tensor = torch.tensor(LL, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    #             LH_tensor = torch.tensor(LH, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    #             HL_tensor = torch.tensor(HL, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    #             HH_tensor = torch.tensor(HH, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    
    #             # 将低频（LL）和高频（LH, HL, HH）部分合并
    #             low_freq.append(torch.cat([LL_tensor,LH_tensor],dim=1))
    #             high_freq.append(torch.cat([HL_tensor, HH_tensor], dim=1))
    
    #     # 合并低频和高频部分
    #     low_freq = torch.cat(low_freq, dim=1).to(x.device)  # 合并所有低频部分
    
    
    #     high_freq = torch.cat(high_freq, dim=1).to(x.device) # 合并所有高频部分
    
    
    #     return low_freq, high_freq

    def forward(self, x):
        low_freq, high_freq = self.wavelet_transform(x)
        low_freq=self.pool(low_freq)


        l=high_freq
        h=low_freq

        fusion1 = low_freq + (high_freq * torch.sigmoid(low_freq)) - (
                    torch.abs(high_freq - low_freq) * torch.sigmoid(high_freq))
        fusion2 = l + (h * torch.sigmoid(l)) - (
                    torch.abs(h - l) * torch.sigmoid(h))
        fusion = torch.cat([fusion1, fusion2], dim=1)
        _,C,H,W = fusion.shape
        fusion = fusion.view(x.size(0),-1,H*2,W*2)

        #return  self.lobck(self.conv(fusion)) + self.shortcut(x)
        return  self.conv(fusion) + self.shortcut(x)

#mamba扫描机制-----------------------------------------------------------------------------------------------------------------------------------------------------
from .common_utils_mbyolo import *

class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)




        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)+ x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        #self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim//2,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        self.dwconv = nn.Conv2d(hidden_dim//2, hidden_dim//2, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_dim//2)
        self.act = nn.SiLU()

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim*2)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim*2, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)
            self.LHLA=LHLA(hidden_dim*2,hidden_dim*2)
            self.conv=nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, stride=1, padding=0)
            self.conv_1 = nn.Conv2d(in_channels, hidden_dim * 2, kernel_size=1, stride=1, padding=0)



    def forward(self, input):
        input = self.in_proj(input)#[2,256,20,20]512

        x1, x2, x3, x4 = torch.chunk(input, 4, dim=1)

        x1_copy = x1
        x1 = torch.cat([x1_copy, x1_copy], dim=1) #c/2
        x2_copy = x2
        x2 = torch.cat([x2_copy, x2_copy], dim=1)
        x3_copy = x3
        x3 = torch.cat([x3_copy, x3_copy], dim=1)
        x4_copy = x4
        x4 = torch.cat([x4_copy, x4_copy], dim=1)

        x_mamba1 = self.act(self.dwconv(self.ss2d(x1))) * x1
        x_mamba2 = self.act(self.dwconv(self.ss2d(x2))) * x2
        x_mamba3 = self.act(self.dwconv(self.ss2d(x3))) * x3
        x_mamba4 = self.act(self.dwconv(self.ss2d(x4))) * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=1) #2c
        # ====================
        #X1 = self.lsblock(input)
        #input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = self.drop_path(self.norm2(x_mamba)) + self.conv_1(input)
        return self.conv(input)

class LHLA(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(LHLA, self).__init__()
        self.hidden_dim = input_dim//4
        self.output_dim = output_dim
        # 线性变换和卷积层
        self.conv1 = nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1, padding=0,stride=1)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # 用于生成Q的线性变换
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # 用于生成K的线性变换
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # 用于生成V的线性变换
        self.kv_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # 用于生成QKV的线性变换

        self.attn1 = nn.MultiheadAttention(self.hidden_dim, num_heads=8)
        self.attn2 = nn.MultiheadAttention(self.hidden_dim, num_heads=8)

        self.linear_out = nn.Linear(self.hidden_dim*2, output_dim)

    def forward(self, x):
        B,C,H,W = x.shape
        # Step 2: 通过卷积得到特征图
        conv_features = self.conv1(x)  # (B, hidden_dim, H, W)

        # Step 3: 投影V, Q, K
        V = self.v_proj(conv_features.flatten(2).transpose(1, 2))  # (B, hidden_dim, L)
        Q = self.q_proj(conv_features.flatten(2).transpose(1, 2))  # (B, hidden_dim, L)
        K = self.k_proj(conv_features.flatten(2).transpose(1, 2))  # (B, hidden_dim, L)
        QKV = self.kv_proj(conv_features.flatten(2).transpose(1, 2))  # (B, hidden_dim, L)

        # Step 4: Scaled Dot-Product Attention
        attn_output1, _ = self.attn1(Q, K, V)  # (B, L, hidden_dim)
        attn_output2, _ = self.attn2(QKV, K, V)  # (B, L, hidden_dim)

        # Step 5: 不使用平均池化，保持空间维度
        # 直接将输出与空间维度保持一致
        y1 = attn_output1.transpose(0, 1).reshape(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W)
        y2 = attn_output2.transpose(0, 1).reshape(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W)

        # Step 6: 展平并通过线性层输出
        # 将 (B, hidden_dim, H, W) 展平为 (B * H * W, hidden_dim)
        #out = y1 + y2  # (B, hidden_dim, H, W)
        out = torch.cat([y1, y2], dim=1)  # (B, hidden_dim, H, W)
        out = out.flatten(2).transpose(1, 2)  # 展平并转置为 (B * H * W, hidden_dim)[1,400,256]

        out = self.linear_out(out)  # (B * H * W, output_dim) [1,400,128]

        # Step 7: 恢复输出形状为 (B, output_dim, H, W)
        out = out.transpose(1, 2).reshape(B, self.output_dim, H, W)  # 恢复到 (B, output_dim, H, W)

        return out
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------