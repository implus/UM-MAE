# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from timm.models.layers import DropPath, to_2tuple

from models_mae_swin import SwinBlock, PatchEmbed, PatchMerge

########################## for finetuning ##############################
class Swin(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, num_classes=1000, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=16, # 16 for finetune
                 posmlp_dim=32,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, global_pool=True,
                 with_cp=False
                 ):
        super().__init__()

        self.with_cp = with_cp

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
                                 posmlp_dim=posmlp_dim,
                                 drop_path=dpr[idx],
                                 downsample=PatchMerge(
                                     patch_size=2,
                                     in_chans=embed_dim*(2**(i_layer - 1)),
                                     embed_dim=embed_dim*(2**i_layer),
                                     norm_layer=norm_layer
                                 ) if downsample_flag else None,
                                 with_cp=with_cp
                )
                self.blocks.append(layer)
                idx += 1
        encoder_out_dim = embed_dim*(2**(self.num_layers-1))

        self.fc_norm = norm_layer(encoder_out_dim)
        self.head = nn.Linear(encoder_out_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        pos_hw = self.pos_hw.to(x.device) 

        # apply Transformer blocks
        for blk in self.blocks:
            x, pos_hw = blk(x, pos_hw)

        x = x.mean(dim=1) # global pool
        x = self.fc_norm(x)
        x = self.head(x)

        return x


def swin_tiny_256(**kwargs):
    model = Swin(img_size=256, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=16, # 16 for finetune
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def swin_large_256(**kwargs):
    model = Swin(img_size=256, patch_size=4, in_chans=3, 
                 embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                 posmlp_dim=64, mlp_ratio=4, window_size=16, # 16 for finetune
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

