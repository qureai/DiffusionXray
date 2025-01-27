"""Classes and functions for neural networks."""
import math
from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import (
    Tensor,
    einsum,
    nn,
)

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.init as init


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        # Ensure the hidden_dim and output_dim match your model's dimensions
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)  # Ensure this matches the input dimension
        
        

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        context_flat = context.view(b, c, -1).permute(0, 2, 1)

        q = self.to_q(x_flat)
        k = self.to_k(context_flat)
        v = self.to_v(context_flat)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out.permute(0, 2, 1).view(b, c, h, w)  # Ensure output shape matches the input shape

class Residual(nn.Module):
    """Wrapper for residual connection of a function."""

    def __init__(self, fn: nn.Module) -> None:
        """Initialize residual connection."""
        super().__init__()
        self.fn = fn
        

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Pass a tensor through the module."""
        return self.fn(x, *args, **kwargs) + x


def get_upsample_conv(dim: int) -> nn.ConvTranspose2d:
    """Initialize transposed convolution layer."""
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def get_downsample_conv(dim: int) -> nn.Conv2d:
    """Initialize convolution layer."""
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    """Class for sinusoidal embeddings."""

    def __init__(self, dim: int) -> None:
        """Initialize SinusoidalPositionEmbeddings."""
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """Pass a tensor through the module."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Neural block with convolutions, norm and activations."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        """Initialize block."""
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Residual block."""

    def __init__(
        self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8
    ) -> None:
        """Initialize a residual block."""
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        h = self.block1(x)

        time_emb = self.mlp(t)
        h += rearrange(time_emb, 'b c -> b c 1 1')

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    """Attention module."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """Initialize Attention."""
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """Linear attention module."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """Initialize linear attention module."""
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """PreNorm Module."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initialize PreNorm."""
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    """U-net architecture."""

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Tuple = (1, 2, 4, 8),
        num_attention_layer: int = 5,
        channels: int = 3,
        block_groups: int = 8,
    ) -> None:
        """Initialize U-net."""
        super().__init__()

        self.channels = channels

        init_dim = init_dim if init_dim is not None else dim // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        #self.init_conv_mask = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.init_conv_mask = nn.Conv2d(1, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=block_groups)

        self.cross_attentions = nn.ModuleList([])
        for dim_out in dims[1:]:
            self.cross_attentions.append(CrossAttention(dim_out))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            has_att = ind >= (num_resolutions - num_attention_layer + 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                        if has_att
                        else nn.Identity(),
                        get_downsample_conv(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        if num_attention_layer >= 1:
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            has_att = ind >= (num_resolutions - num_attention_layer + 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 3, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                        if has_att
                        else nn.Identity(),
                        get_upsample_conv(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = out_dim if out_dim is not None else channels
        self.final_conv = nn.Sequential(
            Residual(
                nn.Sequential(
                    Block(dim, dim, groups=block_groups),
                    Block(dim, dim, groups=block_groups),
                )
            ),
            nn.Conv2d(dim, out_dim, 1),
        )

    def forward(self, x: Tensor, xm: Tensor, t: Tensor = None) -> Tensor:
        """Pass a tensor through the module."""
        x = self.init_conv(x)
        xm = self.init_conv_mask(xm)
        t = self.time_mlp(t)

        h = []

        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
                    
            x = block2(x, t)
                    
            x = attn(x)

            if i < len(self.cross_attentions):
                x = x + self.cross_attentions[i](x, xm)
                    
            h.append(x)
            x = downsample(x)

            if i < len(self.downs) - 1:  # Don't downsample xm on the last iteration
                xm = downsample(xm)
        

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            if i < len(self.cross_attentions):
                x = x + self.cross_attentions[len(self.cross_attentions) - 1 - i](x, xm)
            x = upsample(x)

            if i < len(self.ups) - 1:  # Don't upsample xm on the last iteration
                xm = upsample(xm)

        return self.final_conv(x)


# # Initialize the model
# model = Unet(dim=64)  # Adjust the dimensions and other parameters as needed

# # Print the model architecture
# print("Model Architecture:\n", model)

# # Calculate and print the number of parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"\nTotal number of parameters: {total_params:,}")