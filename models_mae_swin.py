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
from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange

import torch.utils.checkpoint as cp


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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos_hw, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B*nWindow, nHead, Wh*Ww, Wh*Ww)

        # pos_hw (B, H, W, 2); during finetuning, B == 1 to save computation and storage
        assert self.window_size[0] == self.window_size[1]
        pos_windows = window_partition(pos_hw, self.window_size[0])  # B*nWindow, window_size, window_size, 2
        pos_windows = pos_windows.permute(0, 3, 1, 2).flatten(2)  # B*nW, 2, Wh*Ww
        pos_input = pos_windows.unsqueeze(2) - pos_windows.unsqueeze(3) # B*nW, 2, Wh*Ww, Wh*Ww
        # log-spaced coords
        pos_input = torch.sign(pos_input) * torch.log(1. + pos_input.abs())
        relative_position_bias = self.posmlp(pos_input) # B*nW, nH, WW, WW

        if pos_hw.size(0) == 1: # for finetuning B == 1
            nW, nH, WW, WW = relative_position_bias.size()
            B = B_ // nW
            relative_position_bias = relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1, -1)
            relative_position_bias = relative_position_bias.reshape(-1, nH, WW, WW)

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


class PatchMerge(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.norm = norm_layer(in_chans)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, pos_hw): # pos_hw are absolute positions of h and w
        N, L, C = x.shape
        _, H, W, _ = pos_hw.shape
        assert L == H * W
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        assert self.patch_size[0] == 2
        if self.patch_size[0] == 2:
            # shrink position values and scales
            pos_hw = pos_hw[:, 0::2, 0::2, :] / 2.
            
        return x, (H, W), pos_hw


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, posmlp_dim=32,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
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

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

        self.attn_mask = attn_mask
        self.downsample = downsample

    def forward(self, x, pos_hw): # pos_hw (B, H, W, 2)
        def _inner_forward(x, pos_hw):
            #H, W = self.input_resolution
            H, W = pos_hw.size(1), pos_hw.size(2)
            if self.downsample:
                x, (H, W), pos_hw = self.downsample(x, pos_hw)
            B, L, C = x.shape
            assert L == H * W, "input dimension wrong"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                pos_hw = torch.roll(pos_hw, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            if self.attn_mask is not None:
                self.attn_mask = self.attn_mask.to(x.device)
            attn_windows = self.attn(x_windows, pos_hw, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                pos_hw = torch.roll(pos_hw, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            x = x.view(B, H * W, C)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x, pos_hw

        if self.with_cp and x.requires_grad:
            x, pos_hw = cp.checkpoint(_inner_forward, x, pos_hw)
        else:
            x, pos_hw = _inner_forward(x, pos_hw)

        return x, pos_hw


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
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
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class MaskedAutoencoderSwin(nn.Module):
    """ Masked Autoencoder with Swin backbone
    """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, stride=16,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4, window_size=8, # 16 for finetune
                 posmlp_dim=32,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16, 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, vis_mask_ratio=0.):
        super().__init__()

        self.embed_dim = embed_dim
        self.stride = stride
        self.kernel_stride = stride // patch_size

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            print('vis_mask_token is learnable')
            
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer)
        patches_resolution = self.patch_embed.patches_resolution

        self.embed_h = self.embed_w = int(self.patch_embed.num_patches ** 0.5)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)


        pos_h = torch.arange(0, self.embed_h)[None, None, :, None].repeat(1, 1, 1, self.embed_w).float()
        pos_w = torch.arange(0, self.embed_w)[None, None, None, :].repeat(1, 1, self.embed_h, 1).float()
        self.pos_hw = torch.cat((pos_h, pos_w), dim=1) #(1, 2, H, W)

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.kernel = torch.ones(embed_dim, 1, 2, 2)

        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                # add = 1 if i_layer < self.num_layers - 1 else 0 # align with downsample's patch_size
                add = 1
                layer = SwinBlock(dim=embed_dim*(2**i_layer), 
                                 input_resolution=(
                                     patches_resolution[0] // (2**(i_layer+add)),
                                     patches_resolution[1] // (2**(i_layer+add))
                                 ),
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 shift_size=0 if (dep % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, 
                                 posmlp_dim=posmlp_dim,
                                 drop_path=0.,
                                 downsample=PatchMerge(
                                     patch_size=2, # if i_layer < self.num_layers - 1 else 1, 
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
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(encoder_out_dim, 4 * decoder_embed_dim, bias=True)
        self.decoder_expand = nn.PixelShuffle(2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_num_patches = (img_size // stride) ** 2
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.decoder_num_patches, decoder_embed_dim), requires_grad=False)  

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, stride**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
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
        # p = self.patch_embed.patch_size[0]
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
        mask = mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2).unsqueeze(1).contiguous() # (N, 1, H, W)
        new_mask = new_mask.reshape(N, M, M)
        new_mask = new_mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2).unsqueeze(1).contiguous()

        # add vis_mask_token
        if hasattr(self, 'vis_mask_token'):
            token_mask = (~mask).int() - (~new_mask).int()
            vis_mask_token = self.patch_embed.norm(self.vis_mask_token)
            vis_mask_token = vis_mask_token.expand(N, self.patch_embed.num_patches, -1)
            vis_mask_token = vis_mask_token.reshape(N, self.embed_h, self.embed_w, self.embed_dim).permute(0, 3, 1, 2) # N C H W
            vis_mask_token = vis_mask_token * token_mask
        else:
            vis_mask_token = 0

        # prepare variables
        K = self.kernel_stride
        H, W = self.embed_h, self.embed_w
        self.kernel = self.kernel.to(x.device)

        # x to image shape (N, L, D) -> (N, C, H//2, W//2)
        x = x.reshape(N, self.embed_h, self.embed_w, self.embed_dim).permute(0, 3, 1, 2) # N C H W
        x = x * (~new_mask) + vis_mask_token
        x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=K*2, p2=K*2)
        x = F.conv2d(x, self.kernel, dilation=K, groups=self.embed_dim)
        x = rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//(K*2), w=W//(K*2))

        # we have pos_hw (N, H, W, 2) table to go through the network
        pos_hw = self.pos_hw.repeat(N, 1, 1, 1).to(x.device)
        pos_hw = pos_hw * (~mask)
        pos_hw = rearrange(pos_hw, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=K*2, p2=K*2)
        pos_hw = F.conv2d(pos_hw, self.kernel[:2], dilation=K, groups=2)
        pos_hw = rearrange(pos_hw, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//(K*2), w=W//(K*2))
        pos_hw = pos_hw.permute(0, 2, 3, 1) # (N, H//2, W//2, 2)

        x = x.permute(0, 2, 3, 1).reshape(N, -1, self.embed_dim)

        # apply Transformer blocks
        for blk in self.blocks:
            x, pos_hw = blk(x, pos_hw)
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


def mae_swin_tiny_256_dec512d2b(**kwargs):
    model = MaskedAutoencoderSwin(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        mlp_ratio=4, window_size=8, # 16 for finetune
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_swin_large_256_dec512d2b(**kwargs):
    model = MaskedAutoencoderSwin(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
        mlp_ratio=4, window_size=8, # 16 for finetune
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_swin_large_256_dec512d8b64pmd(**kwargs):
    model = MaskedAutoencoderSwin(
        img_size=256, patch_size=4, in_chans=3, stride=16,
        embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
        posmlp_dim=64, mlp_ratio=4, window_size=8, # 16 for finetune
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_swin_tiny_256 = mae_swin_tiny_256_dec512d2b # decoder: 512 dim, 2 blocks
mae_swin_large_256 = mae_swin_large_256_dec512d8b64pmd


