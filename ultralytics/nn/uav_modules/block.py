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
__all__ = ['DySample','SPDConv','FrequencyFocusedDownSampling','SemanticAlignmenCalibration','FMPFM','Add','HLFASM','CCSM','MFFF','AdaptiveWaveletFeatureExtractor','XSSBlock']

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4):
        super(DySample, self).__init__()
        if isinstance(groups, str):
            try:
                groups = int(groups)
            except ValueError:
                raise ValueError(f"groups 必须是整数或可转为整数的字符串，当前值：{groups}")
        assert in_channels % groups == 0, "in_channels 必须能被 groups 整除"
        self.scale = scale
        self.groups = groups
        self.in_channels = in_channels
        self.cg = in_channels // groups

        self.offset = nn.Conv2d(in_channels, 2 * groups * scale ** 2, 1)
        self.normal_init(self.offset, std=0.001)

        self.attention = SelfAttention(in_channels)

        self.alpha = nn.Parameter(torch.tensor(0.5))

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
        B, C, H, W = x.shape
        wx = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
        wy = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)

        grad_x = F.conv2d(x, wx, padding=1, groups=C)
        grad_y = F.conv2d(x, wy, padding=1, groups=C)
        mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)

        conf = mag.mean(dim=1, keepdim=True)

        mn = conf.amin(dim=(2, 3), keepdim=True)
        mx = conf.amax(dim=(2, 3), keepdim=True)
        conf = (conf - mn) / (mx - mn + 1e-6)
        return conf

    def compute_attention(self, x):
        return self.attention(x)

    def sample(self, x, offset, confidence_map, attention_map):
        B, C, H, W = x.shape
        s = self.scale
        G = self.groups
        Cg = self.cg

        att_w = attention_map.mean(dim=1, keepdim=True)

        w = torch.sigmoid(self.alpha * confidence_map + (1 - self.alpha) * att_w)
        w_hr = F.interpolate(w, scale_factor=s, mode="bilinear", align_corners=False)

        off = offset.view(B, G, 2 * s * s, H, W).reshape(B * G, 2 * s * s, H, W)
        off_hr = F.pixel_shuffle(off, s)

        ys = torch.arange(H * s, device=x.device, dtype=x.dtype)
        xs = torch.arange(W * s, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        yy = (yy + 0.5) / s - 0.5
        xx = (xx + 0.5) / s - 0.5

        x_norm = (xx + 0.5) / W * 2 - 1
        y_norm = (yy + 0.5) / H * 2 - 1
        base_grid = torch.stack([x_norm, y_norm], dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B * G, 1, 1, 1)

        w_hr_rep = w_hr.repeat_interleave(G, dim=0)

        dx = off_hr[:, 0:1] * w_hr_rep
        dy = off_hr[:, 1:2] * w_hr_rep

        dx_norm = dx.permute(0, 2, 3, 1) * (2.0 / W)
        dy_norm = dy.permute(0, 2, 3, 1) * (2.0 / H)

        grid = base_grid.clone()
        grid[..., 0:1] = grid[..., 0:1] + dx_norm
        grid[..., 1:2] = grid[..., 1:2] + dy_norm

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
        mid = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, mid, 1)
        self.key_conv = nn.Conv2d(in_channels, mid, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        HW = H * W
        query = self.query_conv(x).view(B, -1, HW).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, HW)
        value = self.value_conv(x).view(B, -1, HW)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class SPDConv(nn.Module):
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
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class ADown(nn.Module):
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

class FrequencyFocusedDownSampling(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)

        self.conv_reduce = Conv(self.c*2, self.c, 1, 1)


        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)


        fgm_out = self.ffm(x2)
        fgm_out = self.conv_resize(fgm_out)
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        x2 = torch.cat((fgm_out, pooled_out), 1)

        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)

class SemanticAlignmenCalibration(nn.Module):
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()
        self.offset_conv[1].weight.data.zero_()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()

        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear',
                                          align_corners=True)

        enhanced_frequency = self.frequency_enhancer(semantic_features)

        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        coarse_features = self.spatial_conv(coarse_features)

        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1,
                                                                                     out_h, out_w)

        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(
            fused_features.device)

        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors

        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)

        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)

        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2,
                                                                                              :, :]

        return final_features


class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=(norm_cfg is None)
        ))
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(
                num_features=num_features,
                momentum=norm_cfg.get('momentum', 0.1),
                eps=norm_cfg.get('eps', 1e-5)
            )
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")


class CAA(nn.Module):

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()


        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

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

        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = self.conv1(x)

        h_conv = self.h_conv(avg)
        v_conv = self.v_conv(h_conv)

        attn_factor = self.act(self.conv2(v_conv))

        return attn_factor

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
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

        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)


        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.conv0_1 = Conv(dim, dim // fc_ratio)
        self.conv0_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-3], g=dim // fc_ratio)
        self.conv0_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-1], g=dim // fc_ratio)
        self.conv2_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)


    def forward(self, x):

        out = self.avg_pool(self.in_conv(x))


        out_ttf = torch.fft.fft2(out, norm='backward')
        out_ttf = self.conv1x1(torch.abs(out_ttf))

        out_strip = self.CAA(out)

        result = out_strip * out_ttf

        result = torch.fft.ifft2(result, dim=(-2, -1), norm='backward')

        result = torch.abs(result)


        result = self.act(self.dw_conv(result))
        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3


        result_last = self.dw_conv(attn) + result + x


        return self.out_conv(result_last)

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class FMPFM(nn.Module):
    def __init__(self, dim,e=0.75):
        super().__init__()
        self.e=e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim*e))

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch),identity), dim=1))


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

class HLFASM(nn.Module):
    def __init__(self, c1, c2):
        super(HLFASM, self).__init__()
        self.conv = Conv(c1, c2, 1, 1,act=True)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.lobck = RGBlock(c2)

    def wavelet_transform(self, x):
        B, C, H, W = x.shape

        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='replicate')

        if not hasattr(self, "_haar_kernel_cache") or self._haar_kernel_cache is None:
            self._haar_kernel_cache = torch.tensor([
                [[0.5, 0.5],
                 [0.5, 0.5]],

                [[-0.5, -0.5],
                 [0.5, 0.5]],

                [[-0.5, 0.5],
                 [-0.5, 0.5]],

                [[0.5, -0.5],
                 [-0.5, 0.5]],
            ], dtype=torch.float32).unsqueeze(1)

        base_kernel = self._haar_kernel_cache.to(device=x.device, dtype=x.dtype)
        weight = base_kernel.repeat(C, 1, 1, 1)

        coeffs = F.conv2d(x, weight, stride=2, padding=0, groups=C)
        h2, w2 = coeffs.shape[-2:]

        coeffs = coeffs.view(B, C, 4, h2, w2)

        LL = coeffs[:, :, 0, :, :]
        LH = coeffs[:, :, 1, :, :]
        HL = coeffs[:, :, 2, :, :]
        HH = coeffs[:, :, 3, :, :]

        low_freq = torch.cat([LL, LH], dim=1)
        high_freq = torch.cat([HL, HH], dim=1)

        return low_freq, high_freq


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

        return  self.conv(fusion) + self.shortcut(x)

from .common_utils_mbyolo import *

class SS2D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            forward_type="v2",
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.K = 4

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

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

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
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
            SelectiveScan=SelectiveScan, ssoflex=self.training,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)
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


class CCSM(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
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
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
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
        input = self.in_proj(input)

        x1, x2, x3, x4 = torch.chunk(input, 4, dim=1)

        x1_copy = x1
        x1 = torch.cat([x1_copy, x1_copy], dim=1)
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
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=1)
        if self.mlp_branch:
            input = self.drop_path(self.norm2(x_mamba)) + self.conv_1(input)
        return self.conv(input)

class LHLA(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(LHLA, self).__init__()
        self.hidden_dim = input_dim//4
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1, padding=0,stride=1)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.kv_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.attn1 = nn.MultiheadAttention(self.hidden_dim, num_heads=8)
        self.attn2 = nn.MultiheadAttention(self.hidden_dim, num_heads=8)

        self.linear_out = nn.Linear(self.hidden_dim*2, output_dim)

    def forward(self, x):
        B,C,H,W = x.shape
        conv_features = self.conv1(x)

        V = self.v_proj(conv_features.flatten(2).transpose(1, 2))
        Q = self.q_proj(conv_features.flatten(2).transpose(1, 2))
        K = self.k_proj(conv_features.flatten(2).transpose(1, 2))
        QKV = self.kv_proj(conv_features.flatten(2).transpose(1, 2))

        attn_output1, _ = self.attn1(Q, K, V)
        attn_output2, _ = self.attn2(QKV, K, V)

        y1 = attn_output1.transpose(0, 1).reshape(B, self.hidden_dim, H, W)
        y2 = attn_output2.transpose(0, 1).reshape(B, self.hidden_dim, H, W)

        out = torch.cat([y1, y2], dim=1)
        out = out.flatten(2).transpose(1, 2)

        out = self.linear_out(out)

        out = out.transpose(1, 2).reshape(B, self.output_dim, H, W)

        return out

MFFF = FMPFM
AdaptiveWaveletFeatureExtractor = HLFASM
XSSBlock = CCSM
