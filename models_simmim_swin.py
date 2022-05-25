# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# SimMIM: https://github.com/microsoft/SimMIM
# --------------------------------------------------------

from functools import partial

import math
from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange

from models_mae_swin import PatchEmbed, SwinBlock, PatchMerge

class SimMIMSwin(nn.Module):
    """ SimMIM with Swin Transformer backbone
    """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, stride=16,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=8, # 16 for finetune
                 posmlp_dim=32,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.stride = stride

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        # --------------------------------------------------------------------------
        # simmim encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        self.patch_size = patch_size

        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)

        pos_h = torch.arange(0, self.embed_h)[None, :, None, None].repeat(1, 1, self.embed_w, 1).float()
        pos_w = torch.arange(0, self.embed_w)[None, None, :, None].repeat(1, self.embed_h, 1, 1).float()
        self.pos_hw = torch.cat((pos_h, pos_w), dim=-1) #(1, H, W, 2)

        self.blocks = nn.ModuleList()
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
                                 drop_path=0.,
                                 downsample=PatchMerge(
                                     patch_size=2, 
                                     in_chans=embed_dim*(2**(i_layer - 1)), 
                                     embed_dim=embed_dim*(2**i_layer),
                                     norm_layer=norm_layer
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
        encoder_out_dim = embed_dim*(2**(self.num_layers-1))
        self.norm = norm_layer(encoder_out_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # simmim decoder specifics
        self.decoder_pred = nn.Linear(encoder_out_dim, 4 * stride**2 * in_chans, bias=True)
        self.decoder_shuffle = nn.PixelShuffle(2)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def unpatchify(self, x, stride=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = stride
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def patchify(self, imgs, stride=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = stride
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x, mask):
        N, _, H, W = x.size()
        # embed patches
        x = self.patch_embed(x)
        H, W = H//self.patch_size, W//self.patch_size

        L = mask.size(1)
        M = int(L**0.5)
        scale = self.embed_h // M
        mask = mask.reshape(N, M, M)
        mask = mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2)

        N, L, _ = x.size()
        mask_tokens = self.mask_token.expand(N, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        pos_hw = self.pos_hw.to(x.device)

        # apply Transformer blocks
        for blk in self.blocks:
            x, pos_hw = blk(x, pos_hw)
        x = self.norm(x)

        return x 

    def forward_decoder(self, x):
        x = self.decoder_pred(x) # 4 * p^2*3
        N, L, D = x.shape
        M = int(L**0.5)
        x = self.decoder_shuffle(x.permute(0, 2, 1).reshape(N, D, M, M)).flatten(2)
        x = x.permute(0, 2, 1)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, mask, p*p*3] 
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs, self.stride)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5 # (N, L, p*p*3)

        # add new type
        pred = pred[mask].reshape(N, -1, D)

        # Following SimMIM, use L1 loss for reconstruction
        loss = (pred - target).abs()
        loss = loss.mean()
        return loss

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask) # returned mask may change
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def simmim_swin_tiny_256(**kwargs):
    model = SimMIMSwin(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        mlp_ratio=4, window_size=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
