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

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio

        self.sr = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = F.avg_pool2d(x_, self.sr_ratio) 
        x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.downsample = downsample

    def forward(self, x, H, W):
        if self.downsample:
            x, (H, W) = self.downsample(x, H, W)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, (H, W)


class PatchMerge(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        N, L, C = x.shape
        assert L == H * W
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class MaskedAutoencoderPVT(nn.Module):
    """ Masked Autoencoder with PVT backbone
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, stride=16,
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[4, 2, 1, 1], # [8, 4, 2, 1] for finetune
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16, 
                 decoder_mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 vis_mask_ratio=0.):
        super().__init__()

        self.embed_dims = embed_dims
        self.stride = stride
        self.kernel_stride = stride // patch_size

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
            print('vis_mask_token is learnable')
            
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches

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
                                     patch_size=2, # if i_layer < self.num_layers - 1 else 1, 
                                     in_chans=embed_dims[i_layer - 1], 
                                     embed_dim=embed_dims[i_layer]
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dims[-1], 4 * decoder_embed_dim, bias=True)
        self.decoder_expand = nn.PixelShuffle(2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_num_patches = (img_size // stride) ** 2
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.decoder_num_patches, decoder_embed_dim), requires_grad=False)  

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, stride**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.decoder_num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, 'vis_mask_token'):
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

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
        N = x.size(0)
        # embed patches
        x = self.patch_embed(x)

        # secondary mask 
        L = mask.size(1)
        vis_cnt = L - len(mask[0].nonzero())
        vis_final_cnt = int(vis_cnt * (1. - self.vis_mask_ratio)) # final visible
        noise = torch.rand(N, L, device=x.device)
        mask_noise = mask.float() + noise
        ids_shuffle = torch.argsort(mask_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        new_mask = torch.ones([N, L], device=x.device)
        new_mask[:, :vis_final_cnt] = 0
        new_mask = torch.gather(new_mask, dim=1, index=ids_restore).to(torch.bool)

        # and amplify mask (N, L) and new_mask (N, L), new_mask has more 1 (mask)
        M = int(L ** 0.5)
        scale = self.embed_h // M 
        mask = mask.reshape(N, M, M)
        mask = mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2).unsqueeze(1).contiguous()
        new_mask = new_mask.reshape(N, M, M)
        new_mask = new_mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2).unsqueeze(1).contiguous()

        # add vis_mask_token
        if hasattr(self, 'vis_mask_token'):
            token_mask = (~mask).int() - (~new_mask).int()
            vis_mask_token = self.vis_mask_token.expand(N, self.patch_embed.num_patches, -1)
            vis_mask_token = vis_mask_token.reshape(N, self.embed_h, self.embed_w, self.embed_dims[0]).permute(0, 3, 1, 2) # N C H W
            vis_mask_token = vis_mask_token * token_mask
        else:
            vis_mask_token = 0

        # prepare variables
        K = self.kernel_stride
        H, W = self.embed_h, self.embed_w
        self.kernel = self.kernel.to(x.device)

        # x to image shape (N, L, D) -> (N, C, H//2, W//2)
        x = x.reshape(N, self.embed_h, self.embed_w, self.embed_dims[0]).permute(0, 3, 1, 2) # N C H W
        x = x * (~new_mask) + vis_mask_token
        x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=K*2, p2=K*2)
        x = F.conv2d(x, self.kernel, dilation=K, groups=self.embed_dims[0])
        x = rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//(K*2), w=W//(K*2))

        # pos_embed to image shape (N, L, D) -> (N, C, H//2, W//2)
        ipe = self.pos_embed.expand(N, -1, -1)
        ipe = ipe.reshape(N, self.embed_h, self.embed_w, self.embed_dims[0]).permute(0, 3, 1, 2)
        ipe = ipe * (~mask) # attention mask here
        ipe = rearrange(ipe, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=K*2, p2=K*2)
        ipe = F.conv2d(ipe, self.kernel, dilation=K, groups=self.embed_dims[0])
        ipe = rearrange(ipe, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//(K*2), w=W//(K*2))

        # add position embedding
        x = x + ipe

        _, _, H, W = x.size()
        # reverse x to (N, L, C)
        x = x.permute(0, 2, 3, 1).reshape(N, -1, self.embed_dims[0])

        # apply Transformer blocks
        for blk in self.blocks:
            x, (H, W) = blk(x, H, W)
        x = self.norm(x)

        return x 

    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        x_vis = x
        N, L, nD = x_vis.shape
        M = int(L**0.5)
        x_vis = self.decoder_expand(x_vis.permute(0, 2, 1).reshape(-1, nD, M, M)).flatten(2)
        x_vis = x_vis.permute(0, 2, 1)
        _, _, D = x_vis.shape

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed.expand(N, -1, -1)
        pos_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_mask = expand_pos_embed[mask].reshape(N, -1, D)

        x = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x, pos_mask.shape[1]

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

        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask) # returned mask may change
        pred, mask_num = self.forward_decoder(latent, mask)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred[:, -mask_num:], mask)
        return loss, pred, mask

def mae_pvt_small_256_dec512d2b(**kwargs):
    model = MaskedAutoencoderPVT(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], sr_ratios=[4, 2, 1, 1], # [8, 4, 2, 1] for finetune
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        decoder_mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_pvt_small_256 = mae_pvt_small_256_dec512d2b # decoder: 512 dim, 2 blocks

