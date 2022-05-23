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

from models_mae_pvt import PVTBlock, PatchMerge, PatchEmbed


########################## for finetuning ##############################
class PVT(nn.Module):
    """ PVT for finetuning
    """
    def __init__(self, img_size=224, num_classes=1000, patch_size=4, in_chans=3, 
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], # [8, 4, 2, 1] for finetune
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, global_pool=True):
        super().__init__()

        self.patch_size = patch_size

        self.embed_dims = embed_dims
            
        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches

        self.num_layers = len(depths)

        # during finetuning we let the pos_embed learn
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]), requires_grad=True)  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        idx = 0
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                layer = PVTBlock(dim=embed_dims[i_layer], 
                                 num_heads=num_heads[i_layer],
                                 sr_ratio=sr_ratios[i_layer],
                                 mlp_ratio=mlp_ratios[i_layer],
                                 qkv_bias=True, qk_scale=None,
                                 drop_path=dpr[idx],
                                 downsample=PatchMerge(
                                     patch_size=2, 
                                     in_chans=embed_dims[i_layer - 1], 
                                     embed_dim=embed_dims[i_layer]
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
                idx += 1

        self.fc_norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

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
        _, _, H, W = x.size()
        # embed patches
        x = self.patch_embed(x)
        H, W = H//self.patch_size, W//self.patch_size

        # add position embedding
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x, (H, W) = blk(x, H, W)

        x = x.mean(dim=1) # global pool
        x = self.fc_norm(x)
        x = self.head(x)

        return x


def pvt_small_256(**kwargs):
    model = PVT(img_size=256, patch_size=4, in_chans=3, 
        embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], # [8, 4, 2, 1] for finetune
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
