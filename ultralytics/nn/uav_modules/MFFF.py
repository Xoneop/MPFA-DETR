__all__ = ['MFFF']
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import time
from typing import Optional
from torch.nn.utils import prune

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
            h_kernel_size: int = 7,  # 水平卷积核大小，默认为7
            v_kernel_size: int = 7,  # 垂直卷积核大小，默认为7
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
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1) ,# 卷积层
            nn.Sigmoid()
        )

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
        result = self.act(self.conv1x1(result))
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

        attn = self.conv3(attn)

        # 1. 加权融合（可能产生大值）
        gate_1 = self.gate(attn)
        gate_2 = self.gate(result)

        result_last = gate_1 * attn + gate_2 * result





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
    def __init__(self, dim):
        super().__init__()
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim))
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

        return self.m(self.cv1(x))




if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 256, 20, 20
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)

    # 建议 groups 取 8/16/32 之类，且要整除 C
    model = MFFF(256).to(device)

    # 记录开始时间
    start_time = time.time()

    y = model(x)
    # 记录结束时间
    end_time = time.time()

    print("x:", x.shape)
    print("y:", y.shape)  # (2,256,40,40)
    print(f"运行时间: {end_time - start_time:.6f} 秒")
