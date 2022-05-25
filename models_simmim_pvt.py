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
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block 
from timm.models.layers import DropPath, to_2tuple

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange

from models_mae_pvt import PatchEmbed, PatchMerge, PVTBlock


class SimMIMPVT(nn.Module):
    """ SimMIM with Pyramid Vision Transformer backbone
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, stride=16,
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.stride = stride
        self.kernel_stride = stride // patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
            
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]), requires_grad=False)  # fixed sin-cos embedding
        self.kernel = torch.ones(embed_dims[0], 1, 2, 2)

        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                layer = PVTBlock(dim=embed_dims[i_layer], 
                                 num_heads=num_heads[i_layer],
                                 sr_ratio=sr_ratios[i_layer],
                                 mlp_ratio=mlp_ratios[i_layer],
                                 qkv_bias=True, qk_scale=None,
                                 drop_path=0.,
                                 downsample=PatchMerge(
                                     patch_size=2,
                                     in_chans=embed_dims[i_layer - 1], 
                                     embed_dim=embed_dims[i_layer]
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_pred = nn.Linear(embed_dims[-1], 4 * stride**2 * in_chans, bias=True)
        self.decoder_shuffle = nn.PixelShuffle(2)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

        # add position embedding
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x, (H, W) = blk(x, H, W)
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

        #loss = (pred - target) ** 2
        loss = (pred - target).abs()
        loss = loss.mean()
        return loss

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask) # returned mask may change
        # pred, mask_num = self.forward_decoder(latent, mask)  # [N, L, p*p*3]
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def simmim_pvt_small_256(**kwargs):
    model = SimMIMPVT(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], # [8, 4, 2, 1] for finetune
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




