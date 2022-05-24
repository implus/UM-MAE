import logging
from pdb import post_mortem
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_
from .base.vit import TIMMVisionTransformer
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.models.builder import BACKBONES
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)

import sys
sys.path.append('..')

from models_mae_pvt import PatchEmbed, PVTBlock, PatchMerge

@BACKBONES.register_module()
class PVTDet(nn.Module):

    def __init__(self, pretrained=None, pretrain_size=256, img_size=1024, num_classes=80, patch_size=4, in_chans=3, 
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], drop_path_rate=0.1, 
                 norm_layer=None, out_indices=[2, 6, 12, 15], fpn_out_dim=768): 
        super().__init__()
        self.patch_size = patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dims = embed_dims  # num_features for consistency with other models
        self.pretrain_size = pretrain_size

        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])

        self.num_layers = len(depths)

        # during finetuning we let the pos_embed learn
        self.pos_embed = nn.Parameter(torch.zeros(1, (pretrain_size//patch_size)**2, embed_dims[0]), requires_grad=True)  

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

        self.flags = out_indices
        
        self.up1 = nn.Sequential(
            nn.LayerNorm(embed_dims[0]),
            nn.Linear(embed_dims[0], fpn_out_dim)
        )
        self.up2 = nn.Sequential(
            nn.LayerNorm(embed_dims[1]),
            nn.Linear(embed_dims[1], fpn_out_dim)
        )
        self.up3 = nn.Sequential(
            nn.LayerNorm(embed_dims[2]),
            nn.Linear(embed_dims[2], fpn_out_dim)
        )
        self.up4 = nn.Sequential(
            nn.LayerNorm(embed_dims[3]),
            nn.Linear(embed_dims[3], fpn_out_dim)
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
    
    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size // self.patch_size,
                                      self.pretrain_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self, x):
        B, _, H, W = x.size()
        outs = []
        BHW = []
        x = self.patch_embed(x)
        H, W = H//self.patch_size, W//self.patch_size

        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x = x + pos_embed

        for index, blk in enumerate(self.blocks):
            x, (H, W) = blk(x, H, W)
            if index in self.flags:
                outs.append(x)
                BHW.append((B, H, W))

        ops = [self.up1, self.up2, self.up3, self.up4]
        for i in range(len(ops)):
            B, H, W = BHW[i]
            outs[i] = ops[i](outs[i]).permute(0, 2, 1).reshape(B, -1, H, W)
        
        return tuple(outs)

    def forward(self, x):
        outs = self.forward_features(x)
        return outs

