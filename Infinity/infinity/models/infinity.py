"""
Definition of Infinity transformer model.
"""

import math
import random
import time
import json
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np
# from torch.nn.attention.flex_attention import flex_attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_fused_op_installed, AdaLNBeforeHead, CrossAttnBlock, SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid
from infinity.models.fastvar_basic import FastVARCrossAttnBlock
from infinity.models.fastvar_utils import set_fastvar_compute_merge_enabled, set_spacevar_config

from infinity.models.fastvar_utils import build_cfg_diff_select_indices, get_num_remain_for_scale
from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

import cv2
import torchvision.transforms as transforms

from infinity.token_entropy_experiment.visualization import (
    compute_token_stats,
    decode_codes_to_bgr_uint8,
    ensure_dir,
    finalize_run_summary,
    save_scale_visualization,
    write_run_config,
)





def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))
def fourior_plot(x):
    # 直接打印中间特征的傅里叶频谱
    x = x[:,:,0,:,:]
    B,C,H,W = x.shape
    x_patch = x
    f = torch.fft.fft2(x_patch)
    f = f.abs() + 1e-6
    f = f.log()
    latent = shift(f).mean(dim=(0, 1))
    latent = latent.diag()[int(H / 2):]  # only use the half-diagonal components
    latent = latent - latent[0]  # visualize 'relative' log amplitudes
    # (i.e., low-freq amp - high freq amp)
    print(list(latent.cpu().numpy()))


def decode_summed_codes_to_bgr_uint8(summed_codes, vae):
    with torch.no_grad():
        img = vae.decode(summed_codes.squeeze(-3))
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    return img[0].detach().cpu().numpy()


def save_intermediate_results(summed_codes,vae):
    transform = transforms.Compose([
        transforms.ToPILImage(),  #
        transforms.Resize(256),  #
        transforms.ToTensor()
    ])
    img = vae.decode(summed_codes.squeeze(-3)).detach().cpu()
    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    generated_image = img[0]
    # save_file = '/data2/guohang/Infinite/intermediate_scale_img.png'
    # cv2.imwrite(save_file, generated_image.cpu().numpy())
    generated_image =  generated_image.cpu().numpy()
    generated_image = transform(generated_image[:, :, ::-1])
    return generated_image


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h


SHARED_UPDATE_LAYER_RANGES = {
    5: [(16, 27)],
    6: [(16, 27)],
    7: [(16, 27)],
    8: [(16, 27)],
    9: [(16, 27)],
    10: [(16, 27)],
}


def in_layer_ranges(layer_idx: int, layer_ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= layer_idx <= end for start, end in layer_ranges)


def should_use_shared_update(scale_ind: int, layer_idx: int, cfg: float, policy: Dict[int, List[Tuple[int, int]]]) -> bool:
    if cfg == 1:
        return False
    if scale_ind not in policy:
        return False
    return in_layer_ranges(layer_idx, policy[scale_ind])


class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> flash_attn_func.__code__.co_varnames={flash_attn_func.__code__.co_varnames} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5 ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(f'======= WARNING: selecting_idx not specified, set to 1/{num_classes} @ {dist.get_device()} =======')
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                #fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (FastVARCrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"self.num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}, depth={depth}, block_chunks={block_chunks}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash if self.t2i else b.attn.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
    

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "infinity_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
            return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        SelfAttnBlock.forward, FastVARCrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        use_shared_update=True,
        shared_update_policy=None,
        warmup_runs=0,
        print_timing=True,
        fixed_history_cache_len=-1,  # -1通常表示“未设置/默认关闭/False”，即不启用fixed history cache
        force_cond_only_branch=False,
        skip_final_two_scales=True,
        cond_only_start_scale=-1,
        # EXPERIMENT FLAG: do one CFG fusion at boundary, then continue cond-only.
        enable_layerwise_cond_only_collapse=False,
        cond_only_start_layer=-1,
        enable_adaptive_cond_only_collapse=False,
        adaptive_stats_only=False,
        adaptive_probe_scale=8,
        adaptive_probe_layer=5,
        adaptive_late_collapse_layer=20,
        adaptive_top_p_percent=10.0,
        adaptive_tau_A=1.0,
        adaptive_tau_g=0.0,
        adaptive_tau_d=0.0,
        enable_cond_only_token_reuse=False,
        cond_only_token_reuse_source='cfg_diff',
        cond_only_token_reuse_policy='freeze_once',
        visualize_token_entropy=False,
        token_entropy_output_dir='',
        token_entropy_overlay_alpha=0.45,
        token_entropy_save_raw=False,
        token_entropy_save_mode='all',
        scale_decode_output_dir='',
        scale_decode_max_scales=10,
        branch_decode_output_dir='',
        branch_decode_scale_index=8,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        if shared_update_policy is None:
            shared_update_policy = SHARED_UPDATE_LAYER_RANGES
        set_fastvar_compute_merge_enabled(bool(getattr(self, 'enable_fastvar_compute_merge', True)))
        set_spacevar_config(
            enabled=bool(getattr(self, 'enable_spacevar_compute_merge', False)),
            ratio_by_scale=getattr(self, 'fastvar_ratio_by_scale', None),
        )
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        if force_cond_only_branch:
            use_shared_update = False
        if cond_only_start_scale >= 0:
            use_shared_update = False
        if enable_layerwise_cond_only_collapse:
            use_shared_update = False
        if enable_adaptive_cond_only_collapse or adaptive_stats_only:
            use_shared_update = False

        scale_decode_enabled = bool(scale_decode_output_dir)
        scale_decode_summaries = []
        if scale_decode_enabled:
            ensure_dir(scale_decode_output_dir)

        branch_decode_enabled = bool(branch_decode_output_dir)
        branch_decode_summaries = []
        if branch_decode_enabled:
            ensure_dir(branch_decode_output_dir)

        entropy_vis_enabled = bool(visualize_token_entropy)
        entropy_vis_output_dir = token_entropy_output_dir
        entropy_vis_summaries = []
        if entropy_vis_enabled:
            if not entropy_vis_output_dir:
                raise ValueError('token_entropy_output_dir must be provided when visualize_token_entropy=1')
            ensure_dir(entropy_vis_output_dir)
            write_run_config(entropy_vis_output_dir, {
                'visualize_token_entropy': bool(visualize_token_entropy),
                'token_entropy_output_dir': token_entropy_output_dir,
                'token_entropy_overlay_alpha': float(token_entropy_overlay_alpha),
                'token_entropy_save_raw': bool(token_entropy_save_raw),
                'token_entropy_save_mode': str(token_entropy_save_mode),
                'skip_final_two_scales': bool(skip_final_two_scales),
                'force_cond_only_branch': bool(force_cond_only_branch),
                'cond_only_start_scale': int(cond_only_start_scale),
                'enable_layerwise_cond_only_collapse': bool(enable_layerwise_cond_only_collapse),
                'cond_only_start_layer': int(cond_only_start_layer),
                'enable_adaptive_cond_only_collapse': bool(enable_adaptive_cond_only_collapse),
                'adaptive_stats_only': bool(adaptive_stats_only),
                'adaptive_probe_scale': int(adaptive_probe_scale),
                'adaptive_probe_layer': int(adaptive_probe_layer),
                'adaptive_late_collapse_layer': int(adaptive_late_collapse_layer),
                'adaptive_top_p_percent': float(adaptive_top_p_percent),
                'adaptive_tau_A': float(adaptive_tau_A),
                'adaptive_tau_g': float(adaptive_tau_g),
                'adaptive_tau_d': float(adaptive_tau_d),
                'adaptive_collapse_once_only': True,
                'adaptive_pruning_keep_ratio': '24r015_32r025_40r035',
                'use_shared_update': bool(use_shared_update),
                'enable_fastvar_compute_merge': bool(getattr(self, 'enable_fastvar_compute_merge', True)),
                'enable_spacevar_compute_merge': bool(getattr(self, 'enable_spacevar_compute_merge', False)),
                'pruning_mode': (
                    'fastvar' if bool(getattr(self, 'enable_fastvar_compute_merge', True))
                    else ('spacevar' if bool(getattr(self, 'enable_spacevar_compute_merge', False)) else 'disabled')
                ),
                'fastvar_keep_ratio_by_scale': dict(getattr(self, 'fastvar_ratio_by_scale', {})),
            })

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        kv_compact_cond_raw, lens_cond_raw, cu_seqlens_k_cond_raw, max_seqlen_k_cond_raw = label_B_or_BLT
        kv_compact_cond, cu_seqlens_k_cond, max_seqlen_k_cond = None, None, None
        if use_shared_update or enable_layerwise_cond_only_collapse or enable_adaptive_cond_only_collapse or adaptive_stats_only or enable_cond_only_token_reuse:
            kv_compact_cond = self.text_norm(kv_compact_cond_raw)
            cu_seqlens_k_cond = cu_seqlens_k_cond_raw
            max_seqlen_k_cond = max_seqlen_k_cond_raw
        if any(np.array(cfg_list) != 1) and not force_cond_only_branch:
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [10, 2048]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        if use_shared_update or enable_layerwise_cond_only_collapse or enable_adaptive_cond_only_collapse or adaptive_stats_only or enable_cond_only_token_reuse:
            cond_only_ca_kv = self.text_proj_for_ca(kv_compact_cond), cu_seqlens_k_cond, max_seqlen_k_cond
        else:
            cond_only_ca_kv = None
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        cond_only_cond_BD_or_gss = cond_BD_or_gss[:B] if (use_shared_update or enable_layerwise_cond_only_collapse or enable_adaptive_cond_only_collapse or adaptive_stats_only or enable_cond_only_token_reuse) else None
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(True, history_cache_max_len=fixed_history_cache_len)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, FastVARCrossAttnBlock) else module.attn).kv_caching(True, history_cache_max_len=fixed_history_cache_len)

        def collapse_to_cond_only_state(
            x_BLC: torch.Tensor,
            cfg: float,
            apply_cfg_fusion: bool = False,
            collapse_from_layer_idx: int = 0,
        ) -> torch.Tensor:
            # EXPERIMENT FLAG: optional boundary CFG fusion before collapsing to cond-only.
            if apply_cfg_fusion and x_BLC.shape[0] > B:
                x_BLC = (cfg * x_BLC[:B] + (1-cfg) * x_BLC[B:]).contiguous()
            else:
                x_BLC = x_BLC[:B].contiguous()
            modules = self.unregistered_blocks if inference_mode else [module for block_chunk_ in self.block_chunks for module in block_chunk_.module.module]
            for module_idx, blk in enumerate(modules):
                if module_idx < collapse_from_layer_idx:
                    continue
                sa_module = blk.sa if isinstance(blk, FastVARCrossAttnBlock) else blk.attn
                if sa_module.cached_k is not None:
                    sa_module.cached_k = sa_module.cached_k[:B].contiguous()
                if sa_module.cached_v is not None:
                    sa_module.cached_v = sa_module.cached_v[:B].contiguous()
            return x_BLC

        def maybe_prepare_cond_only_token_reuse(x_BLC: torch.Tensor, x_shape, scale_ind: int, layer_idx: int, current_bs: int, cfg: float):
            if not enable_cond_only_token_reuse:
                return None
            if cond_only_token_reuse_source != 'cfg_diff':
                return None
            if cond_only_token_reuse_policy != 'freeze_once':
                return None
            if current_bs <= B:
                return None
            if x_BLC.shape[0] <= B or (x_BLC.shape[0] % 2) != 0:
                print(
                    f'[cond-only-token-reuse] skip scale={scale_ind} ({x_shape[1]}x{x_shape[2]}) '
                    f'layer={layer_idx} because dual-branch state is unavailable: batch={x_BLC.shape[0]}'
                )
                return None
            if cfg == 1:
                return None
            if use_shared_update:
                return None
            _, num_remain = get_num_remain_for_scale(x_shape)
            if num_remain is None:
                return None
            select_indices = build_cfg_diff_select_indices(x_BLC[:current_bs], num_remain, x_shape)
            print(
                f'[cond-only-token-reuse] prepare scale={scale_ind} ({x_shape[1]}x{x_shape[2]}) '
                f'layer={layer_idx} keep={num_remain}/{x_shape[1] * x_shape[2]}'
            )
            return select_indices

        def compute_adaptive_collapse_stats(x_BLC: torch.Tensor, cfg: float, tau: float):
            if x_BLC.shape[0] <= B or (x_BLC.shape[0] % 2) != 0:
                return None
            with torch.no_grad():
                full_logits = self.get_logits(x_BLC, cond_BD).float().mul(1 / tau)
                cond_logits = full_logits[:B]
                uncond_logits = full_logits[B:]
                cfg_logits = (cfg * cond_logits + (1 - cfg) * uncond_logits).float()
                if self.use_bit_label:
                    Bq, Lq, Vq = cond_logits.shape
                    if Vq % 2 != 0:
                        raise ValueError(f'bit-label logits last dim must be even, got {Vq}')
                    Dq = Vq // 2
                    cond_view = cond_logits.view(Bq, Lq, Dq, 2)
                    uncond_view = uncond_logits.view(Bq, Lq, Dq, 2)
                    cfg_view = cfg_logits.view(Bq, Lq, Dq, 2)
                    cond_probs = torch.softmax(cond_view, dim=-1)
                    uncond_probs = torch.softmax(uncond_view, dim=-1)
                    cfg_probs = torch.softmax(cfg_view, dim=-1)
                    cond_entropy = -(cond_probs * torch.log(cond_probs.clamp_min(1e-12))).sum(dim=-1).mean(dim=-1)
                    uncond_entropy = -(uncond_probs * torch.log(uncond_probs.clamp_min(1e-12))).sum(dim=-1).mean(dim=-1)
                    m_probs = 0.5 * (cfg_probs + cond_probs)
                    cfg_kl = (cfg_probs * (torch.log(cfg_probs.clamp_min(1e-12)) - torch.log(m_probs.clamp_min(1e-12)))).sum(dim=-1)
                    cond_kl = (cond_probs * (torch.log(cond_probs.clamp_min(1e-12)) - torch.log(m_probs.clamp_min(1e-12)))).sum(dim=-1)
                    diff = (0.5 * (cfg_kl + cond_kl)).mean(dim=-1)
                else:
                    cond_probs = torch.softmax(cond_logits, dim=-1)
                    uncond_probs = torch.softmax(uncond_logits, dim=-1)
                    cfg_probs = torch.softmax(cfg_logits, dim=-1)
                    cond_entropy = -(cond_probs * torch.log(cond_probs.clamp_min(1e-12))).sum(dim=-1)
                    uncond_entropy = -(uncond_probs * torch.log(uncond_probs.clamp_min(1e-12))).sum(dim=-1)
                    m_probs = 0.5 * (cfg_probs + cond_probs)
                    cfg_kl = (cfg_probs * (torch.log(cfg_probs.clamp_min(1e-12)) - torch.log(m_probs.clamp_min(1e-12)))).sum(dim=-1)
                    cond_kl = (cond_probs * (torch.log(cond_probs.clamp_min(1e-12)) - torch.log(m_probs.clamp_min(1e-12)))).sum(dim=-1)
                    diff = 0.5 * (cfg_kl + cond_kl)
                gap = (uncond_entropy - cond_entropy)[0]
                diff = diff[0]
                q05, q25, q50, q75, q95 = torch.quantile(gap.float(), torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=gap.device))
                iqr = q75 - q25
                gap_mean = gap.float().mean()
                gap_std = gap.float().std(unbiased=False)
                gap_min = gap.float().min()
                gap_max = gap.float().max()
                gap_pos_ratio = (gap > 0).float().mean()
                gap_neg_ratio = (gap < 0).float().mean()
                S_g = (q95 - q50) / (iqr + 1e-6)
                top_p = max(0.0, min(float(adaptive_top_p_percent), 100.0))
                top_k = max(1, int(math.ceil(gap.numel() * top_p / 100.0)))
                gap_top_idx = torch.topk(gap, k=top_k, largest=True).indices
                diff_top_idx = torch.topk(diff, k=top_k, largest=True).indices
                overlap = torch.isin(gap_top_idx, diff_top_idx).float().mean()
                A = S_g
                diff_q95 = torch.quantile(diff.float(), 0.95)
                return {
                    'Q05_gap': float(q05.detach().cpu()),
                    'Q25_gap': float(q25.detach().cpu()),
                    'Q50_gap': float(q50.detach().cpu()),
                    'Q75_gap': float(q75.detach().cpu()),
                    'Q95_gap': float(q95.detach().cpu()),
                    'IQR_gap': float(iqr.detach().cpu()),
                    'mean_gap': float(gap_mean.detach().cpu()),
                    'std_gap': float(gap_std.detach().cpu()),
                    'min_gap': float(gap_min.detach().cpu()),
                    'max_gap': float(gap_max.detach().cpu()),
                    'positive_gap_ratio': float(gap_pos_ratio.detach().cpu()),
                    'negative_gap_ratio': float(gap_neg_ratio.detach().cpu()),
                    'S_g': float(S_g.detach().cpu()),
                    'Q95_diff': float(diff_q95.detach().cpu()),
                    'overlap_O': float(overlap.detach().cpu()),
                    'collapse_tendency_A': float(A.detach().cpu()),
                    'collapse_score_source': 'signed_gap_S_g',
                    'top_k': int(top_k),
                }
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')


        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        _scale_time_ms = {}  # per-scale timing dict
        shared_update_applied = {}
        measured_total_ms = 0.0
        effective_scale_labels = []

        for warmup_idx in range(max(0, warmup_runs) + 1):
            is_warmup = warmup_idx < max(0, warmup_runs)
            current_bs = bs
            persistent_cond_only = force_cond_only_branch or (bs == B)
            if not is_warmup:
                _scale_time_ms = {}
                shared_update_applied = {}
                measured_total_ms = 0.0
                effective_scale_labels = []

            cur_L = 0
            accu_BChw, ret = None, []
            idx_Bl_list, idx_Bld_list = [], []
            last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)
            summed_codes = 0
            cond_only_token_reuse_active = False
            cond_only_token_reuse_indices = None
            cond_only_token_reuse_scale = -1
            cond_only_token_reuse_from_layer = -1
            adaptive_decision_made = False
            adaptive_selected_collapse_layer = -1
            adaptive_stats = None

            for si, pn in enumerate(scale_schedule): # si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
                current_bs = B if persistent_cond_only else bs
                # Adaptive collapse is a single global decision; once selected, cond-only persists.
                collapsed_to_cond_only_this_scale = persistent_cond_only
                # For Infinity, we find use 100% pruning ratio at the last two scales is acceptable
                if skip_final_two_scales and (48 == pn[2] or 64 == pn[2]):
                    continue

                cfg = cfg_list[si]
                should_collapse_to_cond_only = (
                    (not enable_adaptive_cond_only_collapse)
                    and (not enable_layerwise_cond_only_collapse)
                    and (not collapsed_to_cond_only_this_scale)
                    and (cond_only_start_scale >= 0)
                    and (si >= cond_only_start_scale)
                )
                if should_collapse_to_cond_only:
                    prepared_reuse_indices = maybe_prepare_cond_only_token_reuse(
                        last_stage,
                        pn,
                        si,
                        0,
                        current_bs,
                        cfg,
                    )
                    if prepared_reuse_indices is not None:
                        cond_only_token_reuse_active = True
                        cond_only_token_reuse_indices = prepared_reuse_indices
                        cond_only_token_reuse_scale = si
                        cond_only_token_reuse_from_layer = 1
                    last_stage = collapse_to_cond_only_state(last_stage, cfg=cfg, apply_cfg_fusion=False, collapse_from_layer_idx=0)
                    current_bs = B
                    persistent_cond_only = True
                    collapsed_to_cond_only_this_scale = True
                scale_force_cond_only = collapsed_to_cond_only_this_scale
                if si >= trunk_scale:
                    break
                cur_L += np.array(pn).prod()
                if not is_warmup:
                    effective_scale_labels.append(f'{pn[1]}x{pn[2]}')

                need_to_pad = 0
                attn_fn = None
                if self.use_flex_attn:
                    attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

                # Per-scale timing
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                shared_update_applied[si] = []
                layer_idx = 0
                for block_idx, b in enumerate(self.block_chunks): # 8
                    # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    
                    for m in b.module: # 4
                        should_layerwise_cond_only_collapse = (
                            (not enable_adaptive_cond_only_collapse)
                            and enable_layerwise_cond_only_collapse
                            and (not persistent_cond_only)
                            and (not collapsed_to_cond_only_this_scale)
                            and (cond_only_start_scale >= 0)
                            and (cond_only_start_layer >= 0)
                            and (si >= cond_only_start_scale)
                            and (layer_idx >= cond_only_start_layer)
                            and (current_bs > B)
                        )
                        if should_layerwise_cond_only_collapse:
                            # EXPERIMENT FLAG: fuse cond/uncond once at boundary, then continue cond-only for this scale only.
                            print(f'[layerwise-cond-only] collapse at scale={si} ({pn[1]}x{pn[2]}) layer={layer_idx} cfg={cfg}')
                            prepared_reuse_indices = maybe_prepare_cond_only_token_reuse(
                                last_stage,
                                pn,
                                si,
                                layer_idx,
                                current_bs,
                                cfg,
                            )
                            if prepared_reuse_indices is not None:
                                cond_only_token_reuse_active = True
                                cond_only_token_reuse_indices = prepared_reuse_indices
                                cond_only_token_reuse_scale = si
                                cond_only_token_reuse_from_layer = layer_idx + 1
                            last_stage = collapse_to_cond_only_state(last_stage, cfg=cfg, apply_cfg_fusion=True, collapse_from_layer_idx=layer_idx)
                            current_bs = B
                            collapsed_to_cond_only_this_scale = True
                        if (
                            (enable_adaptive_cond_only_collapse or adaptive_stats_only)
                            and (not adaptive_decision_made)
                            and (not persistent_cond_only)
                            and (not collapsed_to_cond_only_this_scale)
                            and (si == int(adaptive_probe_scale))
                            and (layer_idx >= int(adaptive_probe_layer))
                            and (current_bs > B)
                        ):
                            adaptive_stats = compute_adaptive_collapse_stats(last_stage, cfg=cfg, tau=tau_list[si])
                            if adaptive_stats_only:
                                adaptive_selected_collapse_layer = -1
                            elif adaptive_stats is None:
                                adaptive_selected_collapse_layer = int(adaptive_late_collapse_layer)
                            else:
                                high_risk = (
                                    adaptive_stats['collapse_tendency_A'] > float(adaptive_tau_A)
                                    and adaptive_stats['Q95_gap'] > float(adaptive_tau_g)
                                    and adaptive_stats['Q95_diff'] > float(adaptive_tau_d)
                                )
                                adaptive_selected_collapse_layer = int(adaptive_probe_layer) if high_risk else int(adaptive_late_collapse_layer)
                            adaptive_decision_made = True
                            mode = 'adaptive-stats-only' if adaptive_stats_only else 'adaptive-cond-only'
                            print(
                                f'[{mode}] decision '
                                f'scale={si} ({pn[1]}x{pn[2]}) probe_layer={layer_idx} '
                                f'selected_layer={adaptive_selected_collapse_layer} '
                                f'stats={json.dumps(adaptive_stats, ensure_ascii=False, sort_keys=True)}'
                            )
                        should_adaptive_cond_only_collapse = (
                            enable_adaptive_cond_only_collapse
                            and adaptive_decision_made
                            and (not persistent_cond_only)
                            and (not collapsed_to_cond_only_this_scale)
                            and (si >= int(adaptive_probe_scale))
                            and (adaptive_selected_collapse_layer >= 0)
                            and (layer_idx >= adaptive_selected_collapse_layer)
                            and (current_bs > B)
                        )
                        if should_adaptive_cond_only_collapse:
                            print(
                                f'[adaptive-cond-only] collapse at scale={si} ({pn[1]}x{pn[2]}) '
                                f'layer={layer_idx} cfg={cfg} selected_layer={adaptive_selected_collapse_layer}'
                            )
                            last_stage = collapse_to_cond_only_state(last_stage, cfg=cfg, apply_cfg_fusion=True, collapse_from_layer_idx=layer_idx)
                            current_bs = B
                            collapsed_to_cond_only_this_scale = True
                        scale_force_cond_only = collapsed_to_cond_only_this_scale
                        layer_use_shared_update = use_shared_update and should_use_shared_update(si, layer_idx, cfg, shared_update_policy)
                        if layer_use_shared_update:
                            x_cond_prev = last_stage[:B]
                            x_uncond_prev = last_stage[B:]
                            sa_module = m.sa if isinstance(m, FastVARCrossAttnBlock) else None
                            prev_cached_k = sa_module.cached_k if sa_module is not None else None
                            prev_cached_v = sa_module.cached_v if sa_module is not None else None
                            if sa_module is not None and prev_cached_k is not None:
                                sa_module.cached_k = prev_cached_k[:B].contiguous()
                                sa_module.cached_v = prev_cached_v[:B].contiguous()
                            x_cond_next = m(
                                x=x_cond_prev,
                                cond_BD=cond_only_cond_BD_or_gss,
                                ca_kv=cond_only_ca_kv,
                                attn_bias_or_two_vector=None,
                                attn_fn=attn_fn,
                                scale_schedule=scale_schedule,
                                rope2d_freqs_grid=self.rope2d_freqs_grid,
                                scale_ind=si,
                                layer_idx=layer_idx,
                                x_shape=pn,
                            )
                            if sa_module is not None:
                                cond_cached_k = sa_module.cached_k
                                cond_cached_v = sa_module.cached_v
                                if prev_cached_k is None:
                                    full_cached_k = torch.cat((cond_cached_k, cond_cached_k.clone()), dim=0)
                                    full_cached_v = torch.cat((cond_cached_v, cond_cached_v.clone()), dim=0)
                                else:
                                    seq_dim = 1 if sa_module.using_flash else 2
                                    prev_len = prev_cached_k.shape[seq_dim]
                                    new_cond_k = cond_cached_k.narrow(seq_dim, prev_len, cond_cached_k.shape[seq_dim] - prev_len)
                                    new_cond_v = cond_cached_v.narrow(seq_dim, prev_len, cond_cached_v.shape[seq_dim] - prev_len)
                                    cond_full_k = torch.cat((prev_cached_k[:B], new_cond_k), dim=seq_dim)
                                    cond_full_v = torch.cat((prev_cached_v[:B], new_cond_v), dim=seq_dim)
                                    uncond_full_k = torch.cat((prev_cached_k[B:], new_cond_k.clone()), dim=seq_dim)
                                    uncond_full_v = torch.cat((prev_cached_v[B:], new_cond_v.clone()), dim=seq_dim)
                                    full_cached_k = torch.cat((cond_full_k, uncond_full_k), dim=0)
                                    full_cached_v = torch.cat((cond_full_v, uncond_full_v), dim=0)
                                sa_module.cached_k = full_cached_k
                                sa_module.cached_v = full_cached_v
                            cond_update = x_cond_next - x_cond_prev
                            x_uncond_next = x_uncond_prev + cond_update
                            last_stage = torch.cat((x_cond_next, x_uncond_next), dim=0)
                            shared_update_applied[si].append(layer_idx)
                        else:
                            module_bs = last_stage.shape[0]
                            use_cond_only_inputs = (
                                cond_only_cond_BD_or_gss is not None
                                and cond_only_ca_kv is not None
                                and (scale_force_cond_only or cond_only_token_reuse_active)
                            )
                            module_cond_BD = cond_only_cond_BD_or_gss if use_cond_only_inputs else cond_BD_or_gss
                            module_ca_kv = cond_only_ca_kv if use_cond_only_inputs else ca_kv
                            external_select_indices = None
                            disable_cache_update_for_reuse = False
                            if (
                                cond_only_token_reuse_active
                                and module_bs == B
                                and si == cond_only_token_reuse_scale
                                and layer_idx >= cond_only_token_reuse_from_layer
                            ):
                                external_select_indices = cond_only_token_reuse_indices
                                disable_cache_update_for_reuse = True
                            last_stage = m(
                                x=last_stage,
                                cond_BD=module_cond_BD,
                                ca_kv=module_ca_kv,
                                attn_bias_or_two_vector=None,
                                attn_fn=attn_fn,
                                scale_schedule=scale_schedule,
                                rope2d_freqs_grid=self.rope2d_freqs_grid,
                                scale_ind=si,
                                layer_idx=layer_idx,
                                x_shape=pn,
                                external_select_indices=external_select_indices,
                                disable_cache_update_for_reuse=disable_cache_update_for_reuse,
                            )
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers) and not scale_force_cond_only:
                            # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                            last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1

                # Per-scale timing end
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                if not is_warmup:
                    _scale_time_ms[f'{pn[1]}x{pn[2]}'] = elapsed_ms
                    measured_total_ms += elapsed_ms
                    if print_timing:
                        print(f'[scale {pn[1]}x{pn[2]}] tokens={pn[1]*pn[2]:5d}  time={elapsed_ms:.1f}ms  scale_force_cond_only={scale_force_cond_only} current_bs={current_bs} last_stage_bs={last_stage.shape[0]}')
                    if shared_update_applied[si]:
                        print(f'[shared-update] scale {pn[1]}x{pn[2]} layers={shared_update_applied[si]}')

                if (cfg != 1) and add_cfg_on_logits and not scale_force_cond_only: #执行的条件是cfg不为1，add_cfg_on_logits为True，且不是cond only分支
                    full_logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                    cond_logits_BlV = full_logits_BlV[:B]
                    uncond_logits_BlV = full_logits_BlV[B:]
                    logits_BlV = cfg * cond_logits_BlV + (1-cfg) * uncond_logits_BlV
                else:#cond only分支
                    cond_for_logits = cond_BD[:current_bs] if current_bs != cond_BD.shape[0] else cond_BD
                    logits_source_BlV = self.get_logits(last_stage[:current_bs], cond_for_logits).mul(1/tau_list[si])
                    cond_logits_BlV = logits_source_BlV[:B]
                    uncond_logits_BlV = logits_source_BlV[B:] if current_bs > B else None
                    logits_BlV = logits_source_BlV
                
                final_logits_for_stats = logits_BlV[:B].detach().clone()
                branch_logits_for_decode = None
                if branch_decode_enabled and (not scale_force_cond_only) and cond_logits_BlV is not None and uncond_logits_BlV is not None:
                    branch_logits_for_decode = {
                        'cond': cond_logits_BlV.detach().clone(),
                        'uncond': uncond_logits_BlV.detach().clone(),
                    }
                
                if self.use_bit_label:
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                else:
                    idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                if vae_type != 0:
                    assert returns_vemb
                    if si < gt_leak:
                        idx_Bld = gt_ls_Bl[si]
                    else:
                        assert pn[0] == 1
                        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                        if self.apply_spatial_patchify: # unpatchify operation
                            idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                            idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                    idx_Bld_list.append(idx_Bld)
                    codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    if branch_decode_enabled and branch_logits_for_decode is not None and (not is_warmup) and (si + 1 == int(branch_decode_scale_index)):
                        scale_label = f'{pn[1]}x{pn[2]}'
                        for branch_name, branch_logits in branch_logits_for_decode.items():
                            branch_tmp_bs, branch_tmp_seq_len = branch_logits.shape[:2]
                            branch_logits = branch_logits.reshape(branch_tmp_bs, -1, 2)
                            branch_idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                                branch_logits,
                                rng=rng,
                                top_k=top_k or self.top_k,
                                top_p=top_p or self.top_p,
                                num_samples=1,
                            )[:, :, 0]
                            branch_idx_Bld = branch_idx_Bld.reshape(branch_tmp_bs, branch_tmp_seq_len, -1)
                            branch_idx_Bld = branch_idx_Bld.reshape(B, pn[1], pn[2], -1)
                            if self.apply_spatial_patchify:
                                branch_idx_Bld = branch_idx_Bld.permute(0, 3, 1, 2)
                                branch_idx_Bld = torch.nn.functional.pixel_shuffle(branch_idx_Bld, 2)
                                branch_idx_Bld = branch_idx_Bld.permute(0, 2, 3, 1)
                            branch_idx_Bld = branch_idx_Bld.unsqueeze(1)
                            branch_codes = vae.quantizer.lfq.indices_to_codes(branch_idx_Bld, label_type='bit_label')
                            branch_summed_codes = summed_codes + F.interpolate(branch_codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up) if si != num_stages_minus_1 else (summed_codes + branch_codes)
                            branch_img_bgr = decode_summed_codes_to_bgr_uint8(branch_summed_codes, vae)
                            out_path = os.path.join(branch_decode_output_dir, f'{branch_name}_scale_{si + 1:02d}_{scale_label}.png')
                            cv2.imwrite(out_path, branch_img_bgr)
                            branch_decode_summaries.append({
                                'branch': branch_name,
                                'scale_index_1based': int(si + 1),
                                'scale_index_0based': int(si),
                                'scale': scale_label,
                                'path': out_path,
                                'decode_expression': 'vae.decode(branch_summed_codes.squeeze(-3))',
                                'cfg_fusion': False,
                            })
                    if entropy_vis_enabled and (not is_warmup):
                        partial_decode_bgr = decode_codes_to_bgr_uint8(vae, summed_codes + F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up) if si != num_stages_minus_1 else (summed_codes + codes))
                        scale_only_bgr = decode_codes_to_bgr_uint8(vae, F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up) if si != num_stages_minus_1 else codes)
                        cond_stats = compute_token_stats(cond_logits_BlV, use_bit_label=self.use_bit_label)
                        uncond_stats = compute_token_stats(uncond_logits_BlV, use_bit_label=self.use_bit_label) if uncond_logits_BlV is not None else None
                        cfg_stats = compute_token_stats(final_logits_for_stats, use_bit_label=self.use_bit_label)
                        entropy_vis_summaries.append(save_scale_visualization(
                            output_dir=entropy_vis_output_dir,
                            scale_idx=si,
                            scale_hw=(pn[1], pn[2]),
                            partial_decode_bgr=partial_decode_bgr,
                            scale_only_bgr=scale_only_bgr,
                            cond_stats=cond_stats,
                            uncond_stats=uncond_stats,
                            cfg_stats=cfg_stats,
                            overlay_alpha=float(token_entropy_overlay_alpha),
                            save_raw=bool(token_entropy_save_raw),
                            save_mode=str(token_entropy_save_mode),
                        ))
                    if si != num_stages_minus_1:
                        summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    else:
                        summed_codes += codes
                    if scale_decode_enabled and (not is_warmup) and (si < int(scale_decode_max_scales)):
                        scale_label = f'{pn[1]}x{pn[2]}'
                        scale_dir = os.path.join(scale_decode_output_dir, f'{si + 1:02d}_{scale_label}')
                        ensure_dir(scale_dir)
                        img_bgr = decode_summed_codes_to_bgr_uint8(summed_codes, vae)
                        out_path = os.path.join(scale_dir, 'summed_codes_decode.png')
                        cv2.imwrite(out_path, img_bgr)
                        scale_decode_summaries.append({
                            'scale_index_1based': int(si + 1),
                            'scale_index_0based': int(si),
                            'scale': scale_label,
                            'path': out_path,
                            'decode_expression': 'vae.decode(summed_codes.squeeze(-3))',
                        })
                    if si != num_stages_minus_1:
                        last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                        last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                        if self.apply_spatial_patchify: # patchify operation
                            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                        last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                    else:
                        summed_codes += codes
                else:
                    if si < gt_leak:
                        idx_Bl = gt_ls_Bl[si]
                    h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                    ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                    if si != num_stages_minus_1:
                        accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
                
                if si != num_stages_minus_1:
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
                    # EXPERIMENT FLAG: layerwise collapse is local to this scale; next scale restores dual-branch hidden state.
                    next_scale_force_cond_only = persistent_cond_only or (
                        (not enable_cond_only_token_reuse)
                        and (not enable_layerwise_cond_only_collapse)
                        and (not enable_adaptive_cond_only_collapse)
                        and (cond_only_start_scale >= 0 and (si + 1) >= cond_only_start_scale)
                    )
                    repeat_factor = 1 if next_scale_force_cond_only else (bs // B)
                    last_stage = last_stage.repeat(repeat_factor, 1, 1)

            if is_warmup:
                for blk in self.unregistered_blocks:
                    if isinstance(blk, FastVARCrossAttnBlock):
                        blk.sa.cached_k = None
                        blk.sa.cached_v = None

            cond_only_token_reuse_active = False
            cond_only_token_reuse_indices = None
            cond_only_token_reuse_scale = -1
            cond_only_token_reuse_from_layer = -1

        if print_timing:
            print(f'[timing] warmup_runs={warmup_runs} total_time={measured_total_ms:.1f}ms scales={effective_scale_labels}')
            timing_summary = {
                'warmup_runs': warmup_runs,
                'total_time_ms': round(float(measured_total_ms), 4),
                'scale_time_ms': {k: round(float(v), 4) for k, v in _scale_time_ms.items()},
                'effective_scales': effective_scale_labels,
                'shared_update_applied': {str(k): v for k, v in shared_update_applied.items() if v},
                'adaptive_cond_only': {
                    'enabled': bool(enable_adaptive_cond_only_collapse),
                    'stats_only': bool(adaptive_stats_only),
                    'probe_scale': int(adaptive_probe_scale),
                    'probe_layer': int(adaptive_probe_layer),
                    'late_collapse_layer': int(adaptive_late_collapse_layer),
                    'selected_collapse_layer': int(adaptive_selected_collapse_layer) if 'adaptive_selected_collapse_layer' in locals() else -1,
                    'stats': adaptive_stats if 'adaptive_stats' in locals() else None,
                },
            }
            print(f'[timing_json]{json.dumps(timing_summary, ensure_ascii=False, sort_keys=True)}')

        if entropy_vis_enabled and entropy_vis_summaries:
            finalize_run_summary(entropy_vis_output_dir, entropy_vis_summaries)

        if scale_decode_enabled and scale_decode_summaries:
            with open(os.path.join(scale_decode_output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(scale_decode_summaries, f, ensure_ascii=False, indent=2)

        if branch_decode_enabled and branch_decode_summaries:
            with open(os.path.join(branch_decode_output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(branch_decode_summaries, f, ensure_ascii=False, indent=2)


        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, FastVARCrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, FastVARCrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
