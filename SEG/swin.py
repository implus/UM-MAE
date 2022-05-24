# --------------------------------------------------------
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

import sys
sys.path.append('..')

from models_mae_swin import PatchEmbed, SwinBlock, PatchMerge

@BACKBONES.register_module()
class Swin(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=256, num_classes=80, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=16, # 16 for finetune
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 out_indices=[1, 3, 9, 11], fpn_out_dim=768): 
        super().__init__()

        self.out_indices = out_indices

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer)
        patches_resolution = self.patch_embed.patches_resolution

        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)

        pos_h = torch.arange(0, self.embed_h)[None, :, None, None].repeat(1, 1, self.embed_w, 1).float()
        pos_w = torch.arange(0, self.embed_w)[None, None, :, None].repeat(1, self.embed_h, 1, 1).float()
        self.pos_hw = torch.cat((pos_h, pos_w), dim=-1) #(1, H, W, 2)

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

        self.fpn1 = nn.Conv2d(embed_dim*(2**0), fpn_out_dim, 1)
        self.fpn2 = nn.Conv2d(embed_dim*(2**1), fpn_out_dim, 1)
        self.fpn3 = nn.Conv2d(embed_dim*(2**2), fpn_out_dim, 1)
        self.fpn4 = nn.Conv2d(embed_dim*(2**3), fpn_out_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward_features(self, x):
        # embed patches
        x = self.patch_embed(x)

        pos_hw = self.pos_hw.to(x.device)

        features = []
        for i, blk in enumerate(self.blocks):
            x, pos_hw = blk(x, pos_hw)
            B = x.size(0)
            H, W = pos_hw.size(1), pos_hw.size(2)
            if i in self.out_indices:
                xp = x.permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x
