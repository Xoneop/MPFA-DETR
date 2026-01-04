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

__all__ = ['DySample','SPDConv','FrequencyFocusedDownSampling','SemanticAlignmenCalibration']

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


class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        # 改进后的 SCA 部分
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        #通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #1*1 进行通道融合
        out = self.in_conv(x)
        #公式1开始
        # 池化后1*1卷积 
        x_att = self.fac_conv(self.fac_pool(out)) # 平均池化加卷积
        x_fft = torch.fft.fft2(out, norm='backward') #FF
        x_fft = x_att * x_fft # 元素级相乘
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')  # FF
        x_fca = torch.abs(x_fca)
        #公式1结束
        
        #公式2
        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3
        #公式2结束

        # 使用通道注意力机制
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        #FF的公式
        x_sca = self.ffm(x_sca)

        # 最终融合 公式4
        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

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
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)

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
    
class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class SemanticAlignmenCalibration(nn.Module):  # 
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        self.Akconv = AKConv(inc[1],inc[0],11)


    def forward(self, x):

        return self.Akconv(x)