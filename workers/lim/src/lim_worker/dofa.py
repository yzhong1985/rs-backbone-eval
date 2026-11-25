"""DOFA ViT-B backbone for the NIR injector (vendored, MIT License).

Adapted from zhu-xlab/DOFA (master, Copyright (c) 2024 Zhitong Xiong), files
``wave_dynamic_layer.py`` and ``dofa_v1.py``. Changes: no timm dependency (the
ViT block is written here with the same parameter names), no global seed, and the
backbone returns the last block's spatial tokens as a feature map. The fixed
sin-cos position table is interpolated bicubically when the token grid differs
from the pretrained 14x14.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])  # Using the last output to generate bias
        return weights, bias


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(wv_planes, self._num_kernel, embed_dim)
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # Fix bug
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DofaViTBase(nn.Module):
    """ViT-B/16 with DOFA's dynamic wavelength patch embedding; returns F3-style maps."""

    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, grid: int = 14):
        super().__init__()
        self.grid = grid
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid * grid + 1, embed_dim), requires_grad=False
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)  # pretrained output norm
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])

    def _pos_for(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        cls_pos = self.pos_embed[:, :1]
        spatial = self.pos_embed[:, 1:]
        if (height, width) != (self.grid, self.grid):
            channels = spatial.shape[-1]
            grid_pos = spatial.reshape(1, self.grid, self.grid, channels).permute(0, 3, 1, 2)
            grid_pos = F.interpolate(
                grid_pos.float(), size=(height, width), mode="bicubic", align_corners=False
            )
            spatial = grid_pos.permute(0, 2, 3, 1).reshape(1, height * width, channels)
        return torch.cat([cls_pos, spatial.to(dtype)], dim=1)

    def forward_map(self, image: torch.Tensor, wave_um: list[float]) -> torch.Tensor:
        """``image``: [B, 1, H, W] normalised NIR. Returns [B, C, H/16, W/16]."""
        waves = torch.tensor(wave_um, device=image.device, dtype=torch.float32)
        tokens, _ = self.patch_embed(image, waves)
        height = image.shape[-2] // 16
        width = image.shape[-1] // 16
        pos = self._pos_for(height, width, tokens.dtype)
        x = tokens + pos[:, 1:]
        cls = (self.cls_token + pos[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        spatial = self.norm(x[:, 1:])
        return spatial.transpose(1, 2).reshape(x.shape[0], -1, height, width)
