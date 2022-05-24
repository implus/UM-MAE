import logging
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)

import sys
sys.path.append('..')

from models_mae_swin import PatchEmbed, Mlp, window_partition, window_reverse

class PatchMerge(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.norm = norm_layer(in_chans)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, hw): # pos_hw are absolute positions of h and w
        N, L, C = x.shape
        H, W = hw
        assert L == H * W
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        #assert self.patch_size[0] == 1 or self.patch_size[0] == 2
        assert self.patch_size[0] == 2
            
        return x, (H, W)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, posmlp_dim=32, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.posmlp = nn.Sequential(
            nn.Conv2d(2, posmlp_dim, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(posmlp_dim, num_heads, 1)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        ws = window_size[0]
        assert window_size[0] == window_size[1]
        pos_h = torch.arange(0, ws)[None, None, :, None].repeat(1, 1, 1, ws).float()
        pos_w = torch.arange(0, ws)[None, None, None, :].repeat(1, 1, ws, 1).float()
        self.pos_hw = torch.cat((pos_h, pos_w), dim=1) 

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (B*num_windows, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B*nWindow, nHead, Wh*Ww, Wh*Ww)

        # pos_hw (B, H, W, 2); during finetuning, B == 1 to save computation and storage
        assert self.window_size[0] == self.window_size[1]
        pos_windows = self.pos_hw.to(x.device).flatten(2) # 1, 2, Wh*Ww
        pos_input = pos_windows.unsqueeze(2) - pos_windows.unsqueeze(3) # 1, 2, Wh*Ww, Wh*Ww
        # log-spaced coords
        pos_input = torch.sign(pos_input) * torch.log(1. + pos_input.abs())
        relative_position_bias = self.posmlp(pos_input) # 1, nH, WW, WW

        # B*nW, nH, WW, WW
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}' 


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, posmlp_dim=32,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, posmlp_dim=posmlp_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.downsample = downsample

    def forward(self, x, hw): # pos_hw (B, H, W, 2)
        #H, W = self.input_resolution
        H, W = hw
        if self.downsample:
            x, (H, W) = self.downsample(x, hw)
        B, L, C = x.shape
        assert L == H * W, "input dimension wrong"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad for downstream tasks
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # pay attention
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            #pos_hw = torch.roll(pos_hw, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            #pos_hw = torch.roll(pos_hw, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, (H, W)


@BACKBONES.register_module()
class SwinDet(nn.Module):

    def __init__(self, pretrained=None, img_size=1024, num_classes=80, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=16, # 16 for finetune
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 posmlp_dim=32, out_indices=[1, 3, 9, 11], fpn_out_dim=768):
        
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer)
        patches_resolution = self.patch_embed.patches_resolution

        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        idx = 0
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                layer = SwinBlock(dim=embed_dim*(2**i_layer), 
                                 input_resolution=(
                                     patches_resolution[0] // (2**(i_layer)),
                                     patches_resolution[1] // (2**(i_layer))
                                 ),
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 shift_size=0 if (dep % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                                 posmlp_dim=posmlp_dim,
                                 drop_path=dpr[idx],
                                 downsample=PatchMerge(
                                     patch_size=2, 
                                     in_chans=embed_dim*(2**(i_layer - 1)), 
                                     embed_dim=embed_dim*(2**i_layer),
                                     norm_layer=norm_layer
                                 ) if downsample_flag else None,
                )
                self.blocks.append(layer)
                idx += 1

        self.flags = out_indices
        
        self.up1 = nn.Sequential(
            nn.LayerNorm(self.embed_dim*(2**0)),
            nn.Linear(self.embed_dim*(2**0), fpn_out_dim)
        )
        self.up2 = nn.Sequential(
            nn.LayerNorm(self.embed_dim*(2**1)),
            nn.Linear(self.embed_dim*(2**1), fpn_out_dim)
        )
        self.up3 = nn.Sequential(
            nn.LayerNorm(self.embed_dim*(2**2)),
            nn.Linear(self.embed_dim*(2**2), fpn_out_dim)
        )
        self.up4 = nn.Sequential(
            nn.LayerNorm(self.embed_dim*(2**3)),
            nn.Linear(self.embed_dim*(2**3), fpn_out_dim)
        )
        
        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            self.custom_load_checkpoint(pretrained, map_location='cpu', strict=False, logger=logger)

    def custom_load_checkpoint(self, filename, map_location, strict, logger):
        from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict
        import re
        from collections import OrderedDict
        checkpoint = _load_checkpoint(filename, map_location, logger)
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {filename}')

        # get state_dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: # to support other framework (i.e., mae)
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        revise_keys=[(r'^module\.', '')]
        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                for k, v in state_dict.items()})
        # Keep metadata in state_dict
        state_dict._metadata = metadata

        # load state_dict
        load_state_dict(self, state_dict, strict, logger)
        return checkpoint

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        B, _, H, W = x.size()
        outs = []
        BHW = []
        x = self.patch_embed(x)
        hw = (H//self.patch_size, W//self.patch_size)

        for index, blk in enumerate(self.blocks):
            x, hw = blk(x, hw)
            B = x.size(0)
            H, W = hw
            if index in self.flags:
                outs.append(x) 
                BHW.append((B, H, W))

        ops = [self.up1, self.up2, self.up3, self.up4]
        for i in range(len(ops)):
            B, H, W = BHW[i]
            outs[i] = ops[i](outs[i]).permute(0, 2, 1).reshape(B, -1, H, W)
        
        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        return outs

