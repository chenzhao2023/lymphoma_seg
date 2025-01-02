from timm.models.layers import to_3tuple
from monai.utils import optional_import
import torch.nn.functional as F
from typing import Sequence, Tuple, Union
from monai.networks.blocks import UnetrBasicBlock, UnetResBlock, UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock, get_conv_layer, UnetBasicBlock
import torch.nn as nn
import torch.nn.functional as nnf
import torch
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


def window_partition(x, window_size):
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2],
               window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x


class SWTransCrossAttention(nn.Module):
    """
        SWTransCrossAttention implements a multi-scale cross-modal attention mechanism with shifted windowing.
        Attributes:
                dim (int): Feature dimension.
                depth (int): Number of attention blocks in the module.
                num_heads (int): Number of attention heads in each block.
                window_size (tuple): Size of the attention window (D, H, W).
                mlp_ratio (float): Expansion ratio for the feedforward network.
                qkv_bias (bool): Whether to include bias terms for QKV projections.
                qk_scale (float): Scaling factor for QK projections (default: None).
                drop (float): Dropout rate for linear projections.
                attn_drop (float): Dropout rate for attention weights.
                drop_path (float or list): Stochastic depth rate for each block.
                norm_layer (nn.Module): Normalization layer used throughout the module.
                downsample (nn.Module): Optional downsampling layer applied after attention.
                use_checkpoint (bool): Enable gradient checkpointing to reduce memory usage.
                pat_merg_rf (int): Reduction factor for patch merging.
                pos_embed_method (str): Method for positional embedding ('relative', 'absolute', etc.).
                concatenated_input (bool): Whether the input is concatenated from multiple modalities.
        """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        self.dim = dim
        # Define a series of multi-scale cross-modal attention blocks
        self.blocks = nn.ModuleList([
            Multi_Scale_Cross_Modal_Attention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pos_embed_method=pos_embed_method,
                concatenated_input=concatenated_input)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf,
                                         concatenated_input=concatenated_input)
        else:
            self.downsample = None

    def forward(self, x, x_1, H, W, T):
        """
                Forward pass for the SWTransCrossAttention module.

                Args:
                    x (Tensor): First input feature map [batch, tokens, channels].
                    x_1 (Tensor): Second input feature map [batch, tokens, channels].
                    H, W, T (int): Spatial dimensions of the feature maps.
                Returns:
                    Tuple: Outputs after attention and optional downsampling, including:
                        - x: Updated feature map from the first input.
                        - x_1: Updated feature map from the second input.
                        - H, W, T: Updated spatial dimensions after processing.
                        - f: Cross-modal features.
                        - x_down, x_1_down: Downsampled versions of x and x_1 (if downsample is used).
                        - Wh, Ww, Wt: Spatial dimensions after downsampling.
                        - f_down: Downsampled cross-modal features (if downsample is used).
        """
        # Compute padded spatial dimensions for shifted windowing
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]

        # Initialize a mask for shifted window attention
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0

        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1
        # Create attention masks for shifted windows
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Iterate through each attention block
        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T

            x, x_1, f = blk(x, x_1, attn_mask)
            # Apply downsampling if defined
            if self.downsample is not None:
                x_down = self.downsample(x, H, W, T)
                x_1_down = self.downsample(x_1, H, W, T)
                f_down = self.downsample(f, H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return x, x_1, H, W, T, f, x_down, x_1_down, Wh, Ww, Wt, f_down
            else:
                return x, x_1, H, W, T, f, x, x_1, H, W, T, f


class Multi_Scale_Cross_Modal_Attention(nn.Module):
    """
        Multi_Scale_Cross_Modal_Attention is a module that implements multi-scale
        cross-modal attention for feature fusion and extraction from different modalities.
        Attributes:
            dim (int): Input feature dimension.
            num_heads (int): Number of attention heads for the attention mechanism.
            window_size (tuple): Size of the attention window in 3D (D, H, W).
            shift_size (tuple): Shift size for shifted window attention (D, H, W).
            mlp_ratio (float): Expansion ratio for the feedforward network.
            qkv_bias (bool): Whether to include bias terms for QKV projections.
            qk_scale (float): Scaling factor for QK projections (default: None).
            drop (float): Dropout rate for linear projections.
            attn_drop (float): Dropout rate for attention weights.
            drop_path (float): Dropout rate for stochastic depth.
            act_layer (nn.Module): Activation function used in the MLP (default: GELU).
            norm_layer (nn.Module): Normalization layer applied in the attention and MLP blocks.
            pos_embed_method (str): Method for positional embedding ('relative' or others).
            concatenated_input (bool): Whether input features are concatenated from multiple modalities.
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pos_embed_method='relative', concatenated_input=True):
        super().__init__()
        if concatenated_input:
            self.dim = dim * 2
        else:
            self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size"

        # Layer normalization and attention
        self.norm1 = norm_layer(self.dim)
        self.attn = CrossModalAttention(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pos_embed_method=pos_embed_method)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None
        self.multi_scale_feature_extractor = MultiScaleFeatureExtractor(in_channels=self.dim, out_channels=self.dim)
        self.modal_fusion = ModalFusion(in_channels=self.dim)
        self.gate_fusion = GateFusion(in_channels=self.dim)

    def forward(self, x, x_1, mask_matrix):
        """
        Forward pass for multi-scale cross-modal attention.
        Args:
            x (Tensor): Input features from modality 1, shape [B, L, C].
            x_1 (Tensor): Input features from modality 2, shape [B, L, C].
            mask_matrix (Tensor): Attention mask matrix for shifted window attention.
        Returns:
            Tuple: Processed feature maps and fused features from both modalities.
        """

        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape

        assert L == H * W * T, "input feature has wrong size"

        # Shortcut for residual connections
        shortcut = x
        x = self.norm1(x)
        x_1 = self.norm1(x_1)

        x_ms_feats = self.multi_scale_feature_extractor(x.view(B, C, H, W, T))
        x_1_ms_feats = self.multi_scale_feature_extractor(x_1.view(B, C, H, W, T))
        x_ms_out = []
        x_1_ms_out = []
        modal_out = []
        # Process features at each scale
        for i in range(len(x_ms_feats)):
            x_ms_feats[i] = x_ms_feats[i].permute(0, 2, 3, 4, 1).contiguous()
            x_1_ms_feats[i] = x_1_ms_feats[i].permute(0, 2, 3, 4, 1).contiguous()

            # Apply padding for window partitioning
            pad_l = pad_t = pad_f = 0
            pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
            pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
            pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
            x_ms_feats[i] = nnf.pad(x_ms_feats[i], (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
            x_1_ms_feats[i] = nnf.pad(x_1_ms_feats[i], (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))

            _, Hp, Wp, Tp, _ = x_ms_feats[i].shape

            # cyclic shift
            if min(self.shift_size) > 0:
                shifted_x = torch.roll(x_ms_feats[i],
                                       shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                       dims=(1, 2, 3))
                shifted_x_1 = torch.roll(x_1_ms_feats[i],
                                         shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                         dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x = x_ms_feats[i]
                shifted_x_1 = x_1_ms_feats[i]
                attn_mask = None

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                       C)  # nW*B, window_size*window_size, C

            x_1_windows = window_partition(shifted_x_1, self.window_size)  # nW*B, window_size, window_size, C
            x_1_windows = x_1_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                           C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows, attn_windows_x_1 = self.attn(x_windows, x_1_windows,
                                                       mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' C
            attn_windows_x_1 = attn_windows_x_1.view(-1, self.window_size[0], self.window_size[1], self.window_size[2],
                                                     C)
            shifted_x_1 = window_reverse(attn_windows_x_1, self.window_size, Hp, Wp, Tp)  # B H' W' C
            # reverse cyclic shift
            if min(self.shift_size) > 0:
                x_ms_feats[i] = torch.roll(shifted_x,
                                           shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                           dims=(1, 2, 3))
                x_1_ms_feats[i] = torch.roll(shifted_x_1,
                                             shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                             dims=(1, 2, 3))

            else:
                x_ms_feats[i] = shifted_x
                x_1_ms_feats[i] = shifted_x_1

            if pad_r > 0 or pad_b > 0:
                x_ms_feats[i] = x_ms_feats[i][:, :H, :W, :T, :].contiguous()
                x_1_ms_feats[i] = x_1_ms_feats[i][:, :H, :W, :T, :].contiguous()

            x_ms_feats[i] = x_ms_feats[i].view(B, H * W * T, C)
            x_1_ms_feats[i] = x_1_ms_feats[i].view(B, H * W * T, C)

            fusion_out = self.modal_fusion(x_ms_feats[i], x_1_ms_feats[i])

            x_ms_feats[i] = shortcut + self.drop_path(x_ms_feats[i])
            x_1_ms_feats[i] = shortcut + self.drop_path(x_1_ms_feats[i])

            x_ms_feats[i] = x_ms_feats[i] + self.drop_path(self.mlp(self.norm2(x_ms_feats[i])))
            x_1_ms_feats[i] = x_1_ms_feats[i] + self.drop_path(self.mlp(self.norm2(x_1_ms_feats[i])))
            x_ms_out.append(x_ms_feats[i])
            x_1_ms_out.append(x_1_ms_feats[i])
            modal_out.append(fusion_out)

        x_scale_1, x_scale_3, x_scale_5 = x_ms_out
        x_1_scale_1, x_1_scale_3, x_1_scale_5 = x_1_ms_out
        f_out1, f_out3, f_out5 = modal_out
        x = self.gate_fusion(x_scale_1, x_scale_3, x_scale_5)
        x_1 = self.gate_fusion(x_1_scale_1, x_1_scale_3, x_1_scale_5)
        f = self.gate_fusion(f_out1, f_out3, f_out5)
        return x, x_1, f


class CrossModalAttention(nn.Module):
    """
        CrossModalAttention implements a cross-modal attention mechanism for feature interaction
        between two different modalities.
        Attributes:
            dim (int): Input feature dimension.
            window_size (tuple): Size of the attention window in 3D (D, H, W).
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias terms in QKV projections.
            qk_scale (float): Scaling factor for QK projections (default: None, uses 1/sqrt(head_dim)).
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for the output projection.
            pos_embed_method (str): Method for positional embedding ('relative' or 'rotary').
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pos_embed_method='relative'):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  #

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed_method = pos_embed_method
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sinposembed = RelativeSinPosEmbed()

    def forward(self, x, x_1, mask=None):

        # Compute QKV for both modalities
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_mod1 = self.qkv(x_1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv_mod1[1], qkv_mod1[2]  # make torchscript happy (cannot use tensor as tuple)
        q_mod1, k_mod1, v_mod1 = qkv_mod1[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Scale Q for numerical stability
        q = q * self.scale
        q_mod1 = q_mod1 * self.scale
        # Apply rotary positional embedding (if specified)
        if self.pos_embed_method == 'rotary':
            pos_embed = self.sinposembed(q)
            cos_pos = pos_embed[..., 1::2].repeat(1, 1, 1, 2).cuda()
            sin_pos = pos_embed[..., ::2].repeat(1, 1, 1, 2).cuda()
            qw2 = torch.stack([-q[..., 1::2], q[..., ::2]], 4)
            qw2 = torch.reshape(qw2, q.shape)
            q = q * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-k[..., 1::2], k[..., ::2]], 4)
            kw2 = torch.reshape(kw2, k.shape)
            k = k * cos_pos + kw2 * sin_pos

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1))
        attn_mod1 = (q_mod1 @ k_mod1.transpose(-2, -1))

        if self.pos_embed_method == 'relative':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)
            attn_mod1 = attn_mod1 + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn_mod1 = attn_mod1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_mod1 = attn_mod1.view(-1, self.num_heads, N, N)
            attn_mod1 = self.softmax(attn_mod1)
        else:
            attn = self.softmax(attn)
            attn_mod1 = self.softmax(attn_mod1)

        attn = self.attn_drop(attn)
        attn_mod1 = self.attn_drop(attn_mod1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # mod 1 dot mod2 * mod2
        x_1 = (attn_mod1 @ v_mod1).transpose(1, 2).reshape(B_, N, C)  # mod 2 dot mod 1 * mod1

        x = self.proj(x)
        x_1 = self.proj(x_1)

        x = self.proj_drop(x)
        x_1 = self.proj_drop(x_1)

        return x, x_1


class RelativeSinPosEmbed(nn.Module):

    def __init__(self, ):
        super(RelativeSinPosEmbed, self).__init__()

    def forward(self, attn):
        batch_sz, _, n_patches, emb_dim = attn.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, emb_dim // 2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / emb_dim)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings.view(n_patches, emb_dim), (1, 1, n_patches, emb_dim))

        return embeddings


rearrange, _ = optional_import("einops", name="rearrange")


class SWINUnetrUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        out = torch.cat((inp, skip), dim=1)
        out = self.conv_block(out)
        out = self.transp_conv(out)

        return out


class SWINUnetrBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        out = torch.cat((inp, skip), dim=1)
        out = self.conv_block(out)

        return out


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2, concatenated_input=False):
        super().__init__()
        if concatenated_input:
            self.dim = dim * 2
        else:
            self.dim = dim
        self.reduction = nn.Linear(8 * self.dim, (8 // reduce_factor) * self.dim, bias=False)
        self.norm = norm_layer(8 * self.dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, ):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden // 2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings


class RelativeSinPosEmbed(nn.Module):

    def __init__(self, ):
        super(RelativeSinPosEmbed, self).__init__()

    def forward(self, attn):
        batch_sz, _, n_patches, emb_dim = attn.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, emb_dim // 2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / emb_dim)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings.view(n_patches, emb_dim), (1, 1, n_patches, emb_dim))

        return embeddings


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  #

        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class SW_Transformer_Encoder(nn.Module):

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24, 48],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        """
        SW_Transformer_Encoder is a custom Swin-based Transformer encoder designed for multi-modal input data.
        It extracts hierarchical features from 3D volumetric data using multiple layers, each with multi-modal 
        attention mechanisms and multi-scale feature processing.
        Attributes:
            pretrain_img_size: Predefined size for input image dimensions.
            patch_size: Size of patches used for tokenization.
            in_chans: Number of input channels.
            embed_dim: Dimension of feature embeddings.
            depths: Number of blocks in each transformer stage.
            num_heads: Number of attention heads per stage.
            window_size: Size of attention windows.
            mlp_ratio: Expansion ratio for the feedforward network.
            qkv_bias: Whether to include bias terms for QKV projections.
            drop_rate: Dropout rate for embedding projection.
            attn_drop_rate: Dropout rate for attention weights.
            drop_path_rate: Stochastic depth rate for each block.
            norm_layer: Normalization layer used throughout the model.
            ape: Whether to use absolute positional embedding.
            spe: Whether to use sinusoidal positional embedding.
            patch_norm: Whether to normalize patch embeddings.
            out_indices: Indices of output feature maps.
            frozen_stages: Number of stages to freeze during training.
            use_checkpoint: Enable gradient checkpointing to save memory.
            pat_merg_rf: Reduction factor for patch merging.
            pos_embed_method: Method for positional embedding.
        """

        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Initialize patch embedding modules for PET and CT modalities
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Initialize positional embeddings if required
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Define stochastic depth rates for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        # Define the first multi-scale cross-modal attention layer
        self.msif_0 = SWTransCrossAttention(dim=embed_dim,
                                            depth=depths[0],
                                            num_heads=num_heads[0],
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=dpr[0],
                                            norm_layer=norm_layer,
                                            downsample=PatchMerging,
                                            use_checkpoint=use_checkpoint,
                                            pat_merg_rf=pat_merg_rf,
                                            pos_embed_method=pos_embed_method,
                                            concatenated_input=False)
        # Define the remaining multi-scale cross-modal attention layers
        for i_layer in range(self.num_layers):
            layer = SWTransCrossAttention(dim=int((embed_dim) * 2 ** i_layer),
                                          depth=depths[i_layer],
                                          num_heads=num_heads[i_layer],
                                          window_size=window_size,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          drop=drop_rate,
                                          attn_drop=attn_drop_rate,
                                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                          norm_layer=norm_layer,
                                          downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                          use_checkpoint=use_checkpoint,
                                          pat_merg_rf=pat_merg_rf,
                                          pos_embed_method=pos_embed_method)
            self.layers.append(layer)
            # Add patch merging for this stage
            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim * 2) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):

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
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):

        outs = []
        # Process the first and second input channels separately
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # Patch embeddings for PET and CT modalities
        x_0 = self.patch_embed_mod0(x_0)

        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:

            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = x_0 + absolute_pos_embed_x0

            x_0 = x_0.flatten(2)
            x_0 = x_0.transpose(1, 2)

        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)

        x_0 = self.pos_drop(x_0)

        x_1 = self.patch_embed_mod1(x_1)  # B C, W, H ,D
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)

        # Fuse the two modalities using the first multi-scale cross-modal attention layer
        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, f, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, f_small = \
            self.msif_0(x_0, x_1, Wh_x0, Ww_x0, Wt_x0)

        out = x_0 + x_1 + f
        out = self.proj_out(out, normalize)
        out_t = out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim)

        out_t = out_t.permute(0, 4, 1, 2, 3)
        out_t = out_t.contiguous()

        outs.append(out_t)

        # Reduce the resolution of the feature maps using patch merging
        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        layer0 = self.layers[0]

        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, f, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, f_1_small = layer0(
            x_0, x_1, int(Wh_x0 / 2), int(Ww_x0 / 2),
            int(Wt_x0 / 2))

        x_0_out = x_out_x0_l0
        x_1_out = x_out_x1_l0

        out_x0_x1_l0 = x_0_out + x_1_out + f
        x_out_l0 = self.proj_out(out_x0_x1_l0, normalize)

        out = x_out_l0.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1

        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0 / 2), int(Ww_x0 / 2), int(Wt_x0 / 2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1 / 2), int(Ww_x1 / 2), int(Wt_x1 / 2))

        layer = self.layers[1]

        x_out_x0_l1, x_out_x1_l1, H_x0_1, W_x0_1, T_x0_1, f, x_0_small, x_1_small, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1, f_1_small = layer(
            x_0, x_1, int(Wh_x0_1_l0),
            int(Ww_x0_1_l0), int(Wt_x0_1_l0))

        x_0_out = x_out_x0_l1
        x_1_out = x_out_x1_l1

        out_x0_x1_l1 = x_0_out + x_1_out + f
        x_out_l1 = self.proj_out(out_x0_x1_l1, normalize)

        out = x_out_l1.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2,
                                                                                         3).contiguous()
        outs.append(out)
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))

        layer = self.layers[2]

        x_out_x0_l2, x_out_x1_l2, H_x0_1, W_x0_1, T_x0_1, f, x_0_small, x_1_small, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2, f_1_small = layer(
            x_0, x_1, int(Wh_x0_1_l1),
            int(Ww_x0_1_l1), int(Wt_x0_1_l1))

        x_0_out = x_out_x0_l2
        x_1_out = x_out_x1_l2

        out_x0_x1_l2 = x_0_out + x_1_out + f
        x_out_l2 = self.proj_out(out_x0_x1_l2, normalize)

        out = x_out_l2.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                         3).contiguous()
        outs.append(out)
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))

        layer = self.layers[3]
        x_out_x0_l3, x_out_x1_l3, H_x0_1, W_x0_1, T_x0_1, f, x_0_small, x_1_small, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3, f_1_small = layer(
            x_0, x_1, int(Wh_x0_1_l2),
            int(Ww_x0_1_l2), int(Wt_x0_1_l2))

        x_0_out = x_out_x0_l3
        x_1_out = x_out_x1_l3
        out_x0_x1_l3 = x_0_out + x_1_out + f

        out_x0_x1_l3 = self.proj_out(out_x0_x1_l3, normalize)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                          3).contiguous()
        outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SW_Transformer_Encoder, self).train(mode)
        self._freeze_stages()


class OurMethod(nn.Module):
    def __init__(self, config):
        super(OurMethod, self).__init__()
        # Configuration flags for convolutional and transformer-based skip connections
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        # SW-Transformer encoder for extracting global features
        self.transformer_encoder = SW_Transformer_Encoder(patch_size=config.patch_size,
                                                          in_chans=int(config.in_chans / 2),
                                                          embed_dim=config.embed_dim,
                                                          depths=config.depths,
                                                          num_heads=config.num_heads,
                                                          window_size=config.window_size,
                                                          mlp_ratio=config.mlp_ratio,
                                                          qkv_bias=config.qkv_bias,
                                                          drop_rate=config.drop_rate,
                                                          drop_path_rate=config.drop_path_rate,
                                                          ape=config.ape,
                                                          spe=config.spe,
                                                          patch_norm=config.patch_norm,
                                                          use_checkpoint=config.use_checkpoint,
                                                          out_indices=config.out_indices,
                                                          pat_merg_rf=config.pat_merg_rf,
                                                          pos_embed_method=config.pos_embed_method,
                                                          )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim * 1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )
        # Decoder blocks for reconstructing the output from hierarchical features
        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=1)  # type: ignore

    def forward(self, x):
        """
        Forward pass for the network.
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, depth, height, width].
        Returns:
            logits (Tensor): Output logits for segmentation.
        """
        # x: [batch, channel=2, depth, height, width]
        # Extract global features
        out = self.transformer_encoder(x)  # (B, n_patch, hidden)

        # stage 4 features
        enc5 = self.res_botneck(out[-1])  # B, 5,5,5,2048
        dec4 = self.decoder5(enc5)  # B, 10,10,10,1024
        enc4 = self.encoder4(out[-2])  # skip features should be twice the di

        # stage 3 features
        dec3 = self.decoder4(dec4, enc4)
        enc3 = self.encoder3(out[-3])  # skip features

        # stage 2 features
        dec2 = self.decoder3(dec3, enc3)
        enc2 = self.encoder2(out[-4])  # skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()

        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)

        return [feat1, feat3, feat5]


class ModalFusion(nn.Module):
    def __init__(self, in_channels):
        super(ModalFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1).contiguous()
        x2 = x2.permute(0, 2, 1).contiguous()

        gate_weights1 = self.gate(x1)
        gate_weights2 = self.gate(x2)

        fused = gate_weights1 * x1 + gate_weights2 * x2

        fused = fused.permute(0, 2, 1).contiguous()

        return fused


class GateFusion(nn.Module):
    def __init__(self, in_channels):
        super(GateFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, scale1, scale3, scale5):
        scale1 = scale1.permute(0, 2, 1).contiguous()
        scale3 = scale3.permute(0, 2, 1).contiguous()
        scale5 = scale5.permute(0, 2, 1).contiguous()

        gate_weights1 = self.gate(scale1)
        gate_weights3 = self.gate(scale3)
        gate_weights5 = self.gate(scale5)

        fused = gate_weights1 * scale1 + gate_weights3 * scale3 + gate_weights5 * scale5

        fused = fused.permute(0, 2, 1).contiguous()

        return fused


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
